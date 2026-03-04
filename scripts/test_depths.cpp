#include <riftcast/DatasetImporter.h>
#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace rift;

int main(){
    std::string meta_path = "/home/timnogga/bachelorthesis/biwi_kinect_head_pose/RIFTCast/config_vci.json";
    DatasetHeader header = IO::readDatasetHeader(meta_path);
    auto importer = createDatasetImporter(header);
    int num_cameras = importer->num_cameras();
    std::cout << "Number of cameras: " << num_cameras << std::endl;
    auto cam_valid = torch::zeros({num_cameras}, torch::kInt32);
    const auto& cameras = importer->getCameras();
    bool found_depth = false;
    for(int i = 0; i < num_cameras; ++i)
    {
        if (cameras[i].name.find("D") == 0){
            cam_valid[i] = 1;
            std::cout << "Camera " << cameras[i].name << " is a depth camera." << std::endl;
            found_depth = true;
        }
    }
    uint32_t test_frame = 10;
    std::vector<torch::Tensor> depths = importer->getDepths(test_frame, cam_valid);
    for(int i = 0; i < depths.size(); ++i)
    {
        if (!depths[i].defined()) continue; // skip if this is not a depth camera

        torch::Tensor cpu_depth = depths[i].to(torch::kCPU).contiguous(); // ensure contiguous for OpenCV because else it does weird shit
        cv::Mat visual(cpu_depth.size(0), cpu_depth.size(1), CV_32FC1, cpu_depth.data_ptr<float>()); // this line allows opencv to read the data without copying the pixels into memory

        double min_val, max_val;
        cv::minMaxLoc(visual, &min_val, &max_val);
        std::cout << "Camera " << cameras[i].name << " Raw Stats: " 
                  << "Min=" << min_val << " | Max=" << max_val << std::endl;
        
       
        cv::Mat mask = (visual > 0); //ignore backround pixels with depth value of 0

        // Scale the actual depth range to fill 0.0 - 1.0 else the image was just black cause the values were so small
        cv::Mat normalized;
        cv::normalize(visual, normalized, 0, 1, cv::NORM_MINMAX, CV_32F, mask);

        // Convert to 8-bit for visualization
        cv::Mat visual_8u;
        normalized.convertTo(visual_8u, CV_8UC1, 255.0);

        //apply colormap cause colored images = pretty :D 
        cv::Mat color_mapped;
        cv::applyColorMap(visual_8u, color_mapped, cv::COLORMAP_JET);

        std::string filename = "visible_depth_" + cameras[i].name + ".png";
        cv::imwrite(filename, color_mapped);
        
        std::cout << "Saved colorized depth: " << filename << std::endl;
    
    }
    return 0;
} 