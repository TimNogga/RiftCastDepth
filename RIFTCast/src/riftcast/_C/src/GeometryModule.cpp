#include <riftcast/GeometryModule.h>

#include <Renderer/Context.h>
#include <Renderer/ContextManager.h>
#include <Renderer/Renderer.h>
#include <Renderer/ShaderManager.h>
#include <riftcast/riftcastkernels.h>

#include <maskcompression/decompress.h>
#include <torchhull/visual_hull.h>
#include <torchhull/gaussian_blur.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

namespace rift
{
class GeometryModule::Impl
{
public:
    Impl();
    ~Impl();

    void init(const uint32_t device_idx, const atcg::ref_ptr<rift::DatasetImporter>& dataloader);
    void renderPrimitiveMaps(const atcg::ref_ptr<atcg::Graph>& mesh, const torch::Tensor& cam_valid_mask);
    void mapPrimitiveMaps(const torch::Tensor& cam_valid);
    void unmapPrimitiveMaps();

    atcg::ref_ptr<atcg::Context> visual_hull_context;
    atcg::ref_ptr<atcg::RendererSystem> visual_hull_renderer;
    atcg::ref_ptr<atcg::ShaderManagerSystem> visual_hull_shader_manager;
    atcg::ref_ptr<atcg::Graph> vh_graph;
    torch::Tensor view_projection_tensor;

    atcg::ref_ptr<rift::DatasetImporter> dataloader;
    uint32_t device_idx;

    std::vector<rift::CameraData> cameras;
    uint32_t primitive_width = 400;
    atcg::ref_ptr<atcg::Shader> primitive_shader;
    std::vector<atcg::ref_ptr<atcg::Framebuffer>> camera_primitive_buffers;
    std::vector<cudaTextureObject_t> host_primitive_texture_handles;
    atcg::DeviceBuffer<cudaTextureObject_t> primitive_texture_handles;
    bool primitive_mapped_vci = false;
};

GeometryModule::Impl::Impl() {}

GeometryModule::Impl::~Impl()
{
    if(visual_hull_renderer) visual_hull_renderer->use();
    if(vh_graph) vh_graph.reset();
    if(primitive_shader) primitive_shader.reset();
    camera_primitive_buffers.clear();
    if(visual_hull_renderer) visual_hull_renderer.reset();
    if(visual_hull_shader_manager) visual_hull_shader_manager.reset();
    if(visual_hull_context) atcg::ContextManager::destroyContext(visual_hull_context);
}

void GeometryModule::Impl::init(const uint32_t device_idx, const atcg::ref_ptr<rift::DatasetImporter>& dataloader)
{
    this->device_idx = device_idx;
    this->dataloader = dataloader;

    torch::Device visual_hull_device(torch::kCUDA, device_idx);
    SET_DEVICE(device_idx);

    at::cuda::CUDAGuard device_guard(device_idx);
    visual_hull_context = atcg::ContextManager::createContext(device_idx);

    visual_hull_renderer       = atcg::make_ref<atcg::RendererSystem>();
    visual_hull_shader_manager = atcg::make_ref<atcg::ShaderManagerSystem>();
    
    visual_hull_renderer->init(4608,
                               5328,
                               visual_hull_context,
                               visual_hull_shader_manager);
    visual_hull_renderer->toggleCulling(false);
    visual_hull_renderer->toggleMSAA(false);

    vh_graph = atcg::Graph::createTriangleMesh();

    view_projection_tensor = dataloader->getViewProjectionTensor().to(visual_hull_device);

    cameras = dataloader->getCameras();
    camera_primitive_buffers.resize(cameras.size());
    host_primitive_texture_handles.resize(cameras.size());
    primitive_texture_handles.create(cameras.size());
    for(uint32_t i = 0; i < cameras.size(); ++i)
    {
        atcg::ref_ptr<atcg::PerspectiveCamera> cam = cameras[i].cam;

        atcg::ref_ptr<atcg::Framebuffer> primitive_buffers =
            atcg::make_ref<atcg::Framebuffer>(primitive_width, (int)(primitive_width / cam->getAspectRatio()));
        atcg::TextureSpecification int_spec;
        int_spec.width                               = primitive_width;
        int_spec.height                              = (int)(primitive_width / cam->getAspectRatio());
        int_spec.format                              = atcg::TextureFormat::RINT;
        atcg::ref_ptr<atcg::Texture2D> primitive_map = atcg::Texture2D::create(int_spec);
        primitive_buffers->attachTexture(primitive_map);
        primitive_buffers->attachDepth();
        primitive_buffers->complete();

        camera_primitive_buffers[i] = primitive_buffers;
    }

    primitive_shader = atcg::make_ref<atcg::Shader>("RIFTCast/src/riftcast/_C/shader/primitive_pass.vs",
                                                    "RIFTCast/src/riftcast/_C/shader/primitive_pass.fs");

    visual_hull_context->deactivate();
}

void GeometryModule::Impl::renderPrimitiveMaps(const atcg::ref_ptr<atcg::Graph>& mesh,
                                               const torch::Tensor& cam_valid_mask)
{
    if(mesh->n_vertices() == 0 || mesh->n_faces() == 0)
    {
        return;
    }

    torch::Tensor host_flags = cam_valid_mask.to(torch::Device(torch::kCPU));
    for(uint32_t id = 0; id < cameras.size(); ++id)
    {
        if(host_flags.index({(int)id}).item<int>() == 0) continue;
        auto vci_camera = cameras[id].cam;

        auto primitive_buffer = camera_primitive_buffers[id];
        primitive_buffer->use();

        visual_hull_renderer->setViewport(0, 0, primitive_buffer->width(), primitive_buffer->height());
        visual_hull_renderer->clear();
        int value = -1;
        primitive_buffer->getColorAttachement(0)->fill(&value);
        visual_hull_renderer->draw(mesh, vci_camera, glm::mat4(1), glm::vec3(1), primitive_shader);
    }

    visual_hull_renderer->finish();
}

void GeometryModule::Impl::mapPrimitiveMaps(const torch::Tensor& cam_valid)
{
    if(primitive_mapped_vci) return;

    auto host_flags = cam_valid.to(torch::kCPU);
    for(uint32_t id = 0; id < cameras.size(); ++id)
    {
        auto primitive_buffer    = camera_primitive_buffers[id];
        atcg::textureArray array = primitive_buffer->getColorAttachement(0)->getTextureArray();

        cudaResourceDesc resDesc = {};
        resDesc.resType          = cudaResourceTypeArray;
        resDesc.res.array.array  = array;

        cudaTextureDesc texDesc  = {};
        texDesc.addressMode[0]   = cudaAddressModeBorder;
        texDesc.addressMode[1]   = cudaAddressModeBorder;
        texDesc.filterMode       = cudaFilterModePoint;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        CUDA_SAFE_CALL(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        host_primitive_texture_handles[id] = texObj;
    }

    primitive_texture_handles.upload(host_primitive_texture_handles.data());

    primitive_mapped_vci = true;
}

void GeometryModule::Impl::unmapPrimitiveMaps()
{
    if(!primitive_mapped_vci) return;

    for(int i = 0; i < (int)cameras.size(); ++i)
    {
        CUDA_SAFE_CALL(cudaDestroyTextureObject(host_primitive_texture_handles[i]));
        camera_primitive_buffers[i]->getColorAttachement()->unmapDevicePointers();
    }

    primitive_mapped_vci = false;
}

GeometryModule::GeometryModule()
{
    impl = std::make_unique<Impl>();
}

GeometryModule::~GeometryModule() {}

void GeometryModule::init(const uint32_t device_idx, const atcg::ref_ptr<rift::DatasetImporter>& dataloader)
{
    impl->init(device_idx, dataloader);
}

GeometryReconstruction
GeometryModule::compute_geometry(const glm::mat4& model, 
                                 const torch::Tensor& cam_valid, 
                                 const std::vector<torch::Tensor>& depths,
                                 const uint32_t frame)
{
    auto forced_cam_valid = cam_valid;
    GeometryReconstruction reconstruction;

    at::cuda::CUDAGuard device_guard(impl->device_idx);
    torch::NoGradGuard no_grad;

    auto torch_stream = at::cuda::getCurrentCUDAStream();
    impl->visual_hull_renderer->use();

    const int PROCESS_H = 1332;
    const int PROCESS_W = 1152;

    try {
        auto loaded_masks = frame == -1 ? impl->dataloader->getMasks(forced_cam_valid) 
                                        : impl->dataloader->getMasks(frame, forced_cam_valid);

        torch::Tensor masks = torch::stack(loaded_masks).contiguous();
        torch::Tensor valid = forced_cam_valid.to(torch::kBool);
        masks = masks.index({valid}).contiguous();

        if(impl->dataloader->enable_smoothing())
        {
            masks = torchhull::gaussian_blur(masks, impl->dataloader->kernel_size(), impl->dataloader->sigma(), true, torch::kFloat32);
        }

        auto host_valid = forced_cam_valid.to(torch::kCPU).to(torch::kBool);
        std::vector<torch::Tensor> valid_depth_tensors;
        std::vector<std::string> valid_cam_names;
        std::vector<int> valid_cam_indices;
        
        for (int i = 0; i < (int)forced_cam_valid.size(0); ++i) {
            if (host_valid[i].item<bool>()) {
                valid_cam_names.push_back(impl->cameras[i].name);
                valid_cam_indices.push_back(i);
                
                if (i < (int)depths.size() && depths[i].defined() && depths[i].numel() > 100) {
                    auto d = depths[i].to(masks.device(), torch::kFloat32);
                    if (d.dim() == 2) d = d.unsqueeze(0).unsqueeze(0); 
                    else if (d.dim() == 3) d = d.permute({2, 0, 1}).unsqueeze(0);
                    d = torch::nn::functional::interpolate(d, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{PROCESS_H, PROCESS_W}).mode(torch::kNearest));
                    d = d.squeeze(0).permute({1, 2, 0});
                    valid_depth_tensors.push_back(d);
                } else {
                    valid_depth_tensors.push_back(torch::zeros({PROCESS_H, PROCESS_W, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(masks.device())));
                }
            }
        }

        auto stacked_depths = torch::stack(valid_depth_tensors).contiguous();
        auto proj_full = impl->view_projection_tensor.index({valid}).contiguous();

        glm::vec3 vol_center = glm::vec3(model[3]);
        float vol_scale = model[0][0];  
        glm::vec3 lower_corner = vol_center - glm::vec3(vol_scale);
        float scale = vol_scale * 2.0f;

        auto [vertices, faces] = torchhull::visual_hull(masks, 
                                                        stacked_depths, 
                                                        proj_full,
                                                        impl->dataloader->level(),
                                                        {lower_corner.x, lower_corner.y, lower_corner.z},
                                                        scale,
                                                        impl->dataloader->partial_masks(),
                                                        "opengl",
                                                        true);

        if (vertices.numel() == 0 || faces.numel() == 0) {
            reconstruction.vertices = torch::empty({0, 3}, torch::kFloat32);
            reconstruction.faces = torch::empty({0, 3}, torch::kInt64);
            reconstruction.normals = torch::empty({0, 3}, torch::kFloat32);
            reconstruction.visible_primitives = torch::empty({0}, torch::kFloat32);
            reconstruction.current_frame = impl->dataloader->getLastAvailableFrame();
            return reconstruction;
        }

        // =========================================================================
        // [MESH CARVER] - NATIVE GPU PORT WITH DIAGNOSTICS
        // =========================================================================
        auto v0 = vertices.index({faces.select(1, 0)});
        auto v1 = vertices.index({faces.select(1, 1)});
        auto v2 = vertices.index({faces.select(1, 2)});
        auto face_normals = torch::cross(v1 - v0, v2 - v0, 1);
        
        auto temp_normals = torch::zeros_like(vertices);
        temp_normals.index_add_(0, faces.select(1, 0), face_normals);
        temp_normals.index_add_(0, faces.select(1, 1), face_normals);
        temp_normals.index_add_(0, faces.select(1, 2), face_normals);
        temp_normals = temp_normals / (torch::norm(temp_normals, 2, 1, true) + 1e-8f);
        
        auto keep_mask = torch::ones({vertices.size(0)}, torch::TensorOptions().dtype(torch::kBool).device(vertices.device()));
        auto origin_clip_homo_64 = torch::tensor({0.0, 0.0, 1.0, 0.0}, torch::TensorOptions().dtype(torch::kFloat64).device(vertices.device()));

        int n_valid = (int)proj_full.size(0);
        for (int ci = 0; ci < n_valid; ++ci) 
        {
            if (valid_cam_names[ci].empty() || valid_cam_names[ci][0] != 'D') continue; 
            
            int orig_cam_idx = valid_cam_indices[ci];
            auto P = proj_full[ci];
            
            auto P_64 = P.to(torch::kFloat64);
            auto P_inv_64 = torch::inverse(P_64);
            auto cam_pos_homo_64 = torch::matmul(P_inv_64, origin_clip_homo_64);
            auto cam_pos_homo = cam_pos_homo_64.to(torch::kFloat32);
            auto cam_pos = cam_pos_homo.slice(0, 0, 3) / cam_pos_homo[3];
            
            auto raw_depth = depths[orig_cam_idx].to(vertices.device(), torch::kFloat32);
            
            // --- FIX: LEAVE FLIP IN PLACE TO UN-FLIP THE DATASET IMPORTER ---
            raw_depth = raw_depth.flip({0}); 
            
            int raw_H = raw_depth.size(0);
            int raw_W = raw_depth.size(1);
            
            auto V = vertices.size(0);
            auto hom_verts = torch::cat({vertices, torch::ones({V, 1}, vertices.options())}, 1);
            auto clip_space = torch::matmul(P, hom_verts.t()).t();
            auto depth_w = clip_space.select(1, 3);
            auto ndc_x = clip_space.select(1, 0) / depth_w;
            auto ndc_y = clip_space.select(1, 1) / depth_w;
            
            auto pixel_x = ((ndc_x + 1.0f) * 0.5f * raw_W).to(torch::kInt64);
            auto pixel_y = ((1.0f - ndc_y) * 0.5f * raw_H).to(torch::kInt64); 
            
            auto valid_mask = (pixel_x >= 0) & (pixel_x < raw_W) & (pixel_y >= 0) & (pixel_y < raw_H) & (depth_w > 0.1f);
            auto px_clamped = pixel_x.clamp(0, raw_W - 1);
            auto py_clamped = pixel_y.clamp(0, raw_H - 1);
            
            auto target_depths = raw_depth.index({py_clamped, px_clamped, 0});
            auto invalid_depth_mask = (target_depths < 0.1f) | (target_depths > 3.9f);
            
            auto rays = vertices - cam_pos.unsqueeze(0);
            rays = rays / (torch::norm(rays, 2, 1, true) + 1e-8f);
            
            auto facing_dot = (temp_normals * rays).sum(1);
            auto is_facing = facing_dot < -0.4f;
            
            auto carve_mask_subset = (~invalid_depth_mask) & is_facing & ((depth_w - 0.15f) < (target_depths - 0.02f));
            auto should_remove = valid_mask & carve_mask_subset;
            
            keep_mask = keep_mask & (~should_remove);
            
            // --- NO GUESSING TELEMETRY ---
            std::cout << "\n[NO GUESSING C++ DEBUG - " << valid_cam_names[ci] << "]" << std::endl;
            if (V > 0) {
                std::cout << "  -> Camera Pos: [" << cam_pos[0].item<float>() << ", " << cam_pos[1].item<float>() << ", " << cam_pos[2].item<float>() << "]" << std::endl;
                std::cout << "  -> P Matrix Row 0: [" << P[0][0].item<float>() << ", " << P[0][1].item<float>() << ", " << P[0][2].item<float>() << ", " << P[0][3].item<float>() << "]" << std::endl;
                std::cout << "  -> Vertex [0] World Pos: [" << vertices[0][0].item<float>() << ", " << vertices[0][1].item<float>() << ", " << vertices[0][2].item<float>() << "]" << std::endl;
                std::cout << "  -> Vertex [0] Pixel (X,Y): " << pixel_x[0].item<int64_t>() << ", " << pixel_y[0].item<int64_t>() << std::endl;
                std::cout << "  -> Vertex [0] depth_w: " << depth_w[0].item<float>() << std::endl;
                std::cout << "  -> Vertex [0] target_depth: " << target_depths[0].item<float>() << std::endl;
                std::cout << "  -> Vertex [0] facing_dot: " << facing_dot[0].item<float>() << std::endl;
            }
            // -----------------------------

            int num_valid = valid_mask.sum().item().toInt();
            int num_invalid_depth = invalid_depth_mask.index({valid_mask}).sum().item().toInt();
            int num_facing = is_facing.index({valid_mask}).sum().item().toInt();
            int num_carved = should_remove.sum().item().toInt();
            
            std::cout << "  -> Target Depths Invalid (<0.1 or >3.9): " << num_invalid_depth << std::endl;
            std::cout << "  -> Normal Shield Passed (facing_dot < -0.4): " << num_facing << std::endl;
            std::cout << "  -> FINAL EXCISED: " << num_carved << std::endl;
        }

        int vertices_before = vertices.size(0);
        vertices = vertices.index({keep_mask});

        auto old_to_new = torch::full({vertices_before}, -1, torch::TensorOptions().dtype(torch::kInt64).device(vertices.device()));
        auto new_indices = torch::arange(vertices.size(0), torch::TensorOptions().dtype(torch::kInt64).device(vertices.device()));
        old_to_new.index_put_({keep_mask}, new_indices);

        faces = old_to_new.index({faces});
        auto valid_faces_mask = (faces.select(1, 0) != -1) & (faces.select(1, 1) != -1) & (faces.select(1, 2) != -1);
        faces = faces.index({valid_faces_mask});

        std::cout << "[MESH CARVER] Final Output -> Vertices: " << vertices.size(0) << " | Faces: " << faces.size(0) << std::endl;
        // =========================================================================

        torch::Tensor visible_primitives_per_camera = torch::empty({0}, torch::kFloat32);

        if(vertices.size(0) > 0)
        {
            impl->vh_graph->resizeVertices(vertices.size(0));
            impl->vh_graph->resizeFaces(faces.size(0));
            impl->vh_graph->getDevicePositions().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, vertices);
            impl->vh_graph->getDeviceFaces().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, faces);
            torch_stream.synchronize();
            impl->vh_graph->unmapAllDevicePointers();

            impl->renderPrimitiveMaps(impl->vh_graph, forced_cam_valid);
            impl->mapPrimitiveMaps(forced_cam_valid);
            
            visible_primitives_per_camera = computeVisiblePrimitivesBatched(impl->primitive_texture_handles,
                                                                            forced_cam_valid,
                                                                            impl->camera_primitive_buffers[0]->width(),
                                                                            impl->camera_primitive_buffers[0]->height(),
                                                                            impl->cameras.size(),
                                                                            faces.size(0));
            torch_stream.synchronize();
            impl->unmapPrimitiveMaps();
        }

        torch::Tensor final_normals = torch::zeros_like(vertices);
        rift::computeVertexNormals(vertices, final_normals, faces);
        
        reconstruction.vertices           = vertices;
        reconstruction.faces              = faces;
        reconstruction.normals            = final_normals;
        reconstruction.visible_primitives = visible_primitives_per_camera;
        reconstruction.current_frame      = impl->dataloader->getLastAvailableFrame();

        return reconstruction;

    } catch (const c10::OutOfMemoryError& e) {
        std::cerr << "\n[CRITICAL] CUDA Out Of Memory!" << std::endl;
        throw e;
    }
}

std::vector<atcg::ref_ptr<atcg::Framebuffer>> GeometryModule::getPrimitiveBuffers() const
{
    return impl->camera_primitive_buffers;
}
}