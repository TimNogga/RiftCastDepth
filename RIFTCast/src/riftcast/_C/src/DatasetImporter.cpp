#include <riftcast/DatasetImporter.h>
#include <DataStructure/TorchUtils.h>
#include <Network/NetworkUtils.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <torch/torch.h>

using json = nlohmann::json;

namespace rift
{
DatasetHeader IO::readDatasetHeader(const nlohmann::json& data)
{
    DatasetHeader header;
    std::string type    = data["type"];
    std::string version = data.value("version", "1.0");
    header.version      = version;
    std::string major   = version.substr(0, version.find_first_of("."));

    if(major != "2")
    {
        ATCG_ERROR("Wrong version of dataset header. Got {0}, expected {1}!", version, "2.0");
        return header;
    }

    if(type == "VCI" || type == "VCI_REAL")
    {
        header.type = DatasetHeader::DatasetType::VCI;
    }
    else
    {
        ATCG_ERROR("Could not match {0} to a dataset type. Use either ATCG, VCI, or Panoptic as dataset type");
        return header;
    }

    header.name = data.value("name", "");
    if(data.contains("dataset"))
    {
        header.frame_count = data["dataset"].value("frame_count", 1);
        header.start_frame = data["dataset"].value("start_frame", 0);
        header.path        = data["dataset"]["path"];
        header.camera_path = data["dataset"]["camera_path"];
        if(data["dataset"].contains("to_world"))
        {
            std::vector<double> to_world_data = data["dataset"]["to_world"];
            header.to_world                   = glm::transpose(glm::make_mat4(to_world_data.data()));
        }
        header.flip_images = data["dataset"].value("flip_images", true);
        header.flip_masks  = data["dataset"].value("flip_masks", true);
    }

    if(data.contains("reconstructor"))
    {
        header.partial_masks      = data["reconstructor"].value("partial_masks", false);
        header.level              = data["reconstructor"].value("level", 10);
        header.reconstruction_gpu = data["reconstructor"].value("gpu", 1);
        header.enable_smoothing   = data["reconstructor"].value("smoothing", true);
        header.kernel_size        = data["reconstructor"].value("kernel_size", 9);
        header.sigma              = data["reconstructor"].value("sigma", 2.0f);
    }

    if(data.contains("server"))
    {
        header.server_ip   = data["server"].value("ip", "127.0.0.1");
        header.server_port = data["server"].value("port", 25565);
    }

    if(data.contains("volume"))
    {
        header.volume_scale    = data["volume"].value("scale", 1.0f);
        std::vector<float> pos = data["volume"]["position"];
        header.volume_position = glm::make_vec3(pos.data());
    }

    if(data.contains("renderer"))
    {
        header.renderer_gpu = data["renderer"].value("gpu", 0);
    }

    if(data.contains("inpainting"))
    {
        header.enable_inpainting = data["inpainting"].value("enable", true);
        header.inpaint_path      = data["inpainting"].value("path", "");
    }
    if(data.contains("depth"))
    {
        header.has_depth       = data["depth"].value("has_depth", false);
        header.depth_scale     = data["depth"].value("scale", 1000.0f); 
        header.depth_extension = data["depth"].value("extension", ".png");
    }

    return header;
}

DatasetHeader IO::readDatasetHeader(const std::string& path)
{
    json data;
    try
    {
        std::fstream f(path);
        data = json::parse(f);
    }
    catch(const std::exception& e)
    {
        return DatasetHeader();
    }
    return readDatasetHeader(data);
}

std::vector<torch::Tensor> DatasetImporter::getMasks(const torch::Tensor& valid)
{
    static float frame_id = start_frame();
    frame_id += _timer.elapsedSeconds() * 30.0f;
    _timer.reset();
    int frame = ((int)frame_id) % num_frames();
    return getMasks(frame, valid);
}

VCIDatasetImporter::VCIDatasetImporter(const DatasetHeader& header) : DatasetImporter(header)
{
    _num_cameras = 0;
    importCameras(header.camera_path);
}

VCIDatasetImporter::~VCIDatasetImporter() {}

std::vector<torch::Tensor> VCIDatasetImporter::getMasks(const uint32_t frame_idx, const torch::Tensor& valid)
{
    std::vector<torch::Tensor> masks(num_cameras());
    torch::Tensor host_valid = valid.to(torch::kCPU);
    auto valid_a = host_valid.accessor<int, 1>();

    const int PROCESS_H = 1332;
    const int PROCESS_W = 1152;

    for(int i = 0; i < num_cameras(); ++i)
    {
        masks[i] = torch::ones({PROCESS_H, PROCESS_W, 1}, torch::kCUDA);

        if(valid_a[i] == 0) continue;

        std::string cam_name = _cameras[i].name;
        const bool is_depth_camera = !cam_name.empty() && cam_name[0] == 'D';
        
        if(is_depth_camera)
        {
            masks[i] = torch::ones({PROCESS_H, PROCESS_W, 1}, torch::kCUDA);
            continue; 
        }

        std::stringstream frame_ss;
        frame_ss << "frame_" << std::setfill('0') << std::setw(5) << frame_idx;
        
        std::string path = _root_path + "/" + frame_ss.str() + "/mask/mask_" + cam_name + ".bin";
        
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if(!f.is_open()) continue;

        std::streamsize size = f.tellg();
        f.seekg(0, std::ios::beg);

        torch::Tensor data = torch::empty({(int)size}, torch::kUInt8);
        f.read((char*)data.data_ptr(), size);

        torch::Tensor native_mask;
        if (size == 5328 * 4608) {
            native_mask = data.reshape({5328, 4608, 1});
        } else if (size == 1080 * 1920) {
            native_mask = data.reshape({1080, 1920, 1});
  
        } else {
            continue;
        }
       
        if (_header.flip_masks) {
            native_mask = native_mask.flip(0);
        }

        native_mask = native_mask.permute({2, 0, 1}).unsqueeze(0).to(torch::kCUDA).to(torch::kFloat32).div(255.0f);

        if(native_mask.sum().item<float>() < 1.0f)
        {
            ATCG_WARN("Camera {} mask is empty at frame {}, skipping", _cameras[i].name, frame_idx);
            continue;
        }

        auto standardized_mask = torch::nn::functional::interpolate(native_mask, 
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{PROCESS_H, PROCESS_W})
            .mode(torch::kNearest));

        masks[i] = standardized_mask.squeeze(0).permute({1, 2, 0}).contiguous();
    }
    
    _last_available_frame = frame_idx;
    return masks;
}

std::vector<std::vector<uint8_t>> VCIDatasetImporter::getImages(const uint32_t frame_idx, const torch::Tensor& valid)
{
    std::vector<std::vector<uint8_t>> file_data(num_cameras());
    torch::Tensor host_valid = valid.to(torch::kCPU);
    float total_size         = 0;
    for(uint32_t i = 0; i < num_cameras(); ++i)
    {
        if(host_valid.index({(int)i}).item<int>() == 0) continue;
        std::string cam_name = _cameras[i].name;
        std::stringstream frame;
        frame << "frame_" << std::setfill('0') << std::setw(5) << frame_idx;
        std::string path = _root_path + "/" + frame.str() + "/rgb/" + cam_name + ".jpg";
        std::ifstream input(path, std::ios::in | std::ios::binary | std::ios::ate);
        if(!input.is_open()) continue;
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        file_data[i].resize(file_size);
        input.read((char*)file_data[i].data(), file_size);
        total_size += file_size;
    }
    _rgb_logger.logSample(total_size);
    return file_data;
}

std::vector<torch::Tensor> VCIDatasetImporter::getDepths(const uint32_t frame_idx, const torch::Tensor& valid)
{
    std::vector<torch::Tensor> depth_tensors(num_cameras());
    torch::Tensor host_valid = valid.to(torch::kCPU);
    auto valid_a = host_valid.accessor<int, 1>();
    
    float total_size = 0; 

    for(int i = 0; i < num_cameras(); ++i)
    {
        if(valid_a[i] == 0) continue;

        std::string cam_name = _cameras[i].name; 
        const bool is_depth_camera = !cam_name.empty() && cam_name[0] == 'D';
        if(!is_depth_camera)
        {
            depth_tensors[i] = torch::zeros({1, 1, 1}, torch::kCUDA);
            continue;
        }

        std::stringstream frame_ss;
        frame_ss << "frame_" << std::setfill('0') << std::setw(5) << frame_idx;
        
        std::string pth_path_rgb = _root_path + "/" + frame_ss.str() + "/rgb/" + cam_name + ".pth";
        std::string pth_path_depth = _root_path + "/" + frame_ss.str() + "/depth/" + cam_name + ".pth";
        std::string pth_path = std::filesystem::exists(pth_path_rgb) ? pth_path_rgb : pth_path_depth;
        if(std::filesystem::exists(pth_path))
        {
            torch::Tensor pth_tensor;
            try
            {
                torch::load(pth_tensor, pth_path);
            }
            catch(const c10::Error& e)
            {
                try
                {
                    std::ifstream in(pth_path, std::ios::binary);
                    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                    c10::IValue iv = torch::pickle_load(bytes);
                    if(iv.isTensor())
                    {
                        pth_tensor = iv.toTensor();
                    }
                    else if(iv.isGenericDict())
                    {
                        for(const auto& kv : iv.toGenericDict())
                        {
                            if(kv.value().isTensor())
                            {
                                pth_tensor = kv.value().toTensor();
                                break;
                            }
                        }
                    }
                }
                catch(const c10::Error& e2)
                {
                    ATCG_WARN("Failed to load depth tensor from {}: {}; pickle fallback failed: {}", pth_path, e.what(), e2.what());
                }
            }

            if(pth_tensor.defined() && pth_tensor.numel() > 0)
            {
                auto t = pth_tensor.detach().to(torch::kCPU).contiguous();

                if(t.dim() == 3 && t.size(0) == 3)
                {
                    t = t.index({0, torch::indexing::Slice(), torch::indexing::Slice()});
                }

                while(t.dim() > 2)
                {
                    if(t.size(0) == 1) t = t.squeeze(0);
                    else if(t.size(t.dim() - 1) == 1) t = t.squeeze(-1);
                    else break;
                }

                if(t.dim() == 1)
                {
                    const int64_t numel = t.numel();
                    const int64_t cam_h = static_cast<int64_t>(_cameras[i].height);
                    const int64_t cam_w = static_cast<int64_t>(_cameras[i].width);

                    if(cam_h > 0 && cam_w > 0 && numel == cam_h * cam_w)
                    {
                        t = t.view({cam_h, cam_w});
                    }
                    else if(cam_h > 0 && cam_w > 0 && numel == cam_h * cam_w * 3)
                    {
                        t = t.view({3, cam_h, cam_w}).index({0, torch::indexing::Slice(), torch::indexing::Slice()});
                    }
                    else if(numel % 3 == 0)
                    {
                        const int64_t gray_numel = numel / 3;
                        const int64_t guessed_h = 1080;
                        if(gray_numel % guessed_h == 0)
                        {
                            const int64_t guessed_w = gray_numel / guessed_h;
                            t = t.view({3, guessed_h, guessed_w}).index({0, torch::indexing::Slice(), torch::indexing::Slice()});
                        }
                    }
                }

                if(t.dim() != 2)
                {
                    ATCG_WARN("Unsupported depth tensor shape in {}: {}", pth_path, t.sizes());
                    depth_tensors[i] = torch::zeros({1, 1, 1}, torch::kCUDA);
                    continue;
                }

                static bool pth_depth_logged = false;
                if(!pth_depth_logged)
                {
                    ATCG_INFO("Using .pth depth tensors (example: {}, dtype={}, shape={})",
                              pth_path,
                              t.dtype(),
                              t.sizes());
                    pth_depth_logged = true;
                }

                if(_header.flip_images)
                {
                    t = t.flip({0}).contiguous();
                }

                total_size += t.numel() * t.element_size();

                torch::Tensor depth_tensor;
                if(t.is_floating_point())
                {
                    depth_tensor = t.to(torch::kCUDA).to(torch::kFloat32).mul(4.0f).unsqueeze(-1);
                }
                else
                {
                    depth_tensor = t.to(torch::kCUDA).to(torch::kFloat32).div(_header.depth_scale).unsqueeze(-1);
                }

                depth_tensors[i] = depth_tensor;
                continue;
            }
        }

        std::string path = _root_path + "/" + frame_ss.str() + "/depth/" + cam_name + _header.depth_extension;
        cv::Mat depth_map = cv::imread(path, cv::IMREAD_UNCHANGED);
        
        if(depth_map.empty()) {
             path = _root_path + "/" + frame_ss.str() + "/rgb/" + cam_name + _header.depth_extension;
             depth_map = cv::imread(path, cv::IMREAD_UNCHANGED);
             if(depth_map.empty()) {
                 depth_tensors[i] = torch::zeros({1, 1, 1}, torch::kCUDA);
                 continue;
             }
        }

        if(_header.flip_images)
        {
            cv::flip(depth_map, depth_map, 0);
        }

        total_size += depth_map.total() * depth_map.elemSize();

        if(depth_map.channels() > 1) cv::extractChannel(depth_map, depth_map, 0);

        const bool is_8bit = depth_map.depth() == CV_8U;
        if(!is_8bit && depth_map.depth() != CV_16U) depth_map.convertTo(depth_map, CV_16U);

        torch::Tensor cpu_tensor;
        if(is_8bit)
        {
            cpu_tensor = torch::empty({depth_map.rows, depth_map.cols, 1}, torch::kUInt8);
            for(int r = 0; r < depth_map.rows; ++r)
            {
                std::memcpy(cpu_tensor.data_ptr<uint8_t>() + r * depth_map.cols,
                            depth_map.ptr<uint8_t>(r),
                            depth_map.cols * sizeof(uint8_t));
            }
        }
        else
        {
            cpu_tensor = torch::empty({depth_map.rows, depth_map.cols, 1}, torch::kInt16);
            for(int r = 0; r < depth_map.rows; ++r)
            {
                std::memcpy(cpu_tensor.data_ptr<int16_t>() + r * depth_map.cols,
                            depth_map.ptr<uint16_t>(r),
                            depth_map.cols * sizeof(uint16_t));
            }
        }

        float depth_divisor = _header.depth_scale;
        if(is_8bit)
        {
            depth_divisor = 255.0f / 3.0f;
        }

        auto depth_tensor = cpu_tensor.to(torch::kCUDA).to(torch::kFloat32).div(depth_divisor);
        if(is_8bit)
        {
            depth_tensor = depth_tensor.clamp(0.0f, 3.0f);
        }
        depth_tensors[i] = depth_tensor;
    }

    _depth_logger.logSample(total_size); 
    return depth_tensors;
}

void VCIDatasetImporter::importCameras(const std::string& camera_config)
{
    json data;
    std::string config_path = _root_path + "/calibration_dome.json";
    
    std::fstream f(config_path);
    if(!f.is_open()) return;
    try { data = json::parse(f); } catch(...) { return; }

    json camera_data = data["cameras"];
    std::vector<glm::mat4> host_view_projections;
    for(auto it = camera_data.begin(); it != camera_data.end(); ++it)
    {
        std::string cam_str_id = (*it)["camera_id"].get<std::string>();
        std::string type = (*it).value("camera_type", "");
        if(type == "mono") continue;

        std::string id_digits = "";
        for(char c : cam_str_id) if(std::isdigit(c)) id_digits += c;
        uint32_t camera_id = id_digits.empty() ? 0 : std::stoi(id_digits);

        std::vector<double> extr_vec = (*it)["extrinsics"]["view_matrix"];
        std::vector<double> intr_vec = (*it)["intrinsics"]["camera_matrix"];
        std::vector<int> res = (*it)["intrinsics"]["resolution"];

        glm::mat4 extrinsic_matrix = glm::transpose(glm::make_mat4(extr_vec.data()));
        glm::mat4 cv_to_gl(1);
        cv_to_gl[1][1] = -1.0f; cv_to_gl[2][2] = -1.0f;
        extrinsic_matrix = cv_to_gl * extrinsic_matrix * cv_to_gl * _header.to_world;

        atcg::CameraIntrinsics intrinsics = atcg::CameraUtils::convert_from_opencv((float)intr_vec[0], (float)intr_vec[4], (float)intr_vec[2], (float)intr_vec[5], 0.01f, 1000.0f, res[0], res[1]);
        atcg::CameraExtrinsics extr_obj(extrinsic_matrix);
        atcg::ref_ptr<atcg::PerspectiveCamera> cam = atcg::make_ref<atcg::PerspectiveCamera>(extr_obj, intrinsics);

        _cameras.push_back(CameraData {cam_str_id, (uint32_t)res[0], (uint32_t)res[1], camera_id, cam});
        host_view_projections.push_back(cam->getViewProjection());
    }

    _num_cameras = (uint32_t)_cameras.size();
    
    if (_num_cameras > 0) {
        this->_width = _cameras[0].width;
        this->_height = _cameras[0].height;

        _view_projection_tensor = atcg::createHostTensorFromPointer((float*)host_view_projections.data(), {(int)_num_cameras, 4, 4}).to(torch::kCUDA).transpose(1, 2);
    }
}

atcg::ref_ptr<DatasetImporter> createDatasetImporter(const DatasetHeader& header)
{
    if(header.type == DatasetHeader::DatasetType::VCI) return atcg::make_ref<VCIDatasetImporter>(header);
    return nullptr;
}
}