#include "main.h"
#include "ACMP.h"
#include "cnpy.h"
#include <algorithm>
#include <set>
#include <cstdlib>
#include <ctime>


void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %ld.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %d.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %d.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %d.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

bool InitCUDA()
{
    //used to count the device numbers
    int count;
    // get the cuda device count
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    // find the device >= 1.X
    int i;
    for (i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                printDeviceProp(prop);
                break;
            }
        }
    }
    // if can't find the device
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    // set cuda device
    // cudaSetDevice(i);
    // printf("CUDA initialized.\n");
    return true;
}


void GenerateSampleList(const std::string &dense_folder, std::vector<Problem> &problems)
{
    std::string cluster_list_path = dense_folder + std::string("/pair.txt");
    problems.clear();
    std::ifstream file(cluster_list_path);
    int num_images;
    file >> num_images;
    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;
        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

void ProcessProblem(const std::string &dense_folder, const Problem &problem, bool geom_consistency, bool planar_prior, bool multi_geometrty=false)
{
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    cudaSetDevice(0); // * 设置 CUDA 设备
    std::stringstream result_path;
    result_path << dense_folder << "/ACMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    ACMP acmp;
    if (geom_consistency) {
        acmp.SetGeomConsistencyParams(multi_geometrty); // * 仅仅设计几个参数
    }
    // std::cout<<"acmp.InuputInitialization 输入数据初始化\n";
    acmp.InuputInitialization(dense_folder, problem); // * 读取相机和图像信息、设置参数、如果几何一致则再读深度图

    // std::cout<<"acmp.CudaSpaceInitialization cuda空间初始化\n";
    acmp.CudaSpaceInitialization(dense_folder, problem);

    // std::cout<<"acmp.RunPatchMatch\n";
    acmp.RunPatchMatch();

    const int width = acmp.GetReferenceImageWidth();
    const int height = acmp.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            float4 plane_hypothesis = acmp.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = acmp.GetCost(center);
        }
    }

    if (planar_prior) {
        std::cout << "Run Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        acmp.SetPlanarPriorParams(); // * params.planar_prior = true

        const cv::Rect imageRC(0, 0, width, height);
        std::vector<cv::Point> support2DPoints;

        acmp.GetSupportPoints(support2DPoints);
        const auto triangles = acmp.DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat refImage = acmp.GetReferenceImage().clone();
        std::vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
        cv::Mat srcImage;
        cv::merge(mbgr, srcImage);
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            }
        }
        std::string triangulation_path = result_folder + "/triangulation.png";
        cv::imwrite(triangulation_path, srcImage);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        std::vector<float4> planeParams_tri;
        planeParams_tri.clear();

        uint32_t idx = 0;
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                float max_edge_length = std::max(L01, std::max(L02, L12));
                float step = 1.0 / max_edge_length;

                for (float p = 0; p < 1.0; p += step) {
                    for (float q = 0; q < 1.0 - p; q += step) {
                        int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                        int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                        mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                    }
                }

                // estimate plane parameter
                float4 n4 = acmp.GetPriorPlaneParams(triangle, depths);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }

        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    float d = acmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= acmp.GetMaxDepth() && d >= acmp.GetMinDepth()) {
                        priordepths(j, i) = d;
                    }
                    else {
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }
        //* 同一个三角形中且深度在给定范围的点拥有相同的mask
        // std::string depth_path = result_folder + "/depths_prior.dmb";
       //  writeDepthDmb(depth_path, priordepths);

        acmp.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        acmp.RunPatchMatch();

        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis = acmp.GetPlaneHypothesis(center);
                depths(row, col) = plane_hypothesis.w;
                normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                costs(row, col) = acmp.GetCost(center);
            }
        }
    }

    std::string suffix = "/depths.dmb"; // *depths
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

void single_depth(std::string &dense_folder, const std::vector<Problem> &problems, bool geom_consistency)
{
    size_t num_images = problems.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams_1");

    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat_<uchar>> seg_masks;
    std::map<int,int> src_ids;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();
    seg_masks.clear();
    uchar mask_value=18;
    bool use_orig=1==1;
    bool use_fit=1==0;
    size_t i=0;
    std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
    cv::Mat_<cv::Vec3b> image = cv::imread(image_path.str(), cv::IMREAD_COLOR);
    std::cout<<"Reach here:"<<257<<"\n";

    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
    Camera camera = ReadCamera(cam_path.str());
    std::stringstream seg_mask_path;
    seg_mask_path << image_folder << "/region_seg_mask/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_region_seg_mask.npy";
    std::string seg_mask_path_str=seg_mask_path.str();
    cv::Mat_<uchar> seg_mask;
    std::cout<<"Reach here:"<<269<<"\n";
    std::vector<cv::Point> mask_points;
    std::vector<cv::Point> edge_points;
    mask_points.clear();
    edge_points.clear();

    std::stringstream result_path;
    result_path << dense_folder << "/ACMP" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
    std::string result_folder = result_path.str();
    std::string suffix = "/depths.dmb"; //*depths
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    cv::Mat_<float> depth;
    cv::Mat_<cv::Vec3f> normal;
    readDepthDmb(depth_path, depth);
    readNormalDmb(normal_path, normal);

    cv::Mat_<cv::Vec3b> scaled_image;
    RescaleImageAndCamera(image, scaled_image, depth, camera);
    images.push_back(scaled_image);
    cameras.push_back(camera);
    depths.push_back(depth);
    normals.push_back(normal);
    cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    masks.push_back(mask);
    src_ids[problems[i].ref_image_id]=i;

    std::vector<PointList> PointCloud;
    PointCloud.clear();
    i=0;
    std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
    const int cols = depths[i].cols;
    const int rows = depths[i].rows;

    int num_ngb = problems[i].src_image_ids.size();
    std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
    int patchsize=25;
    for (int r =0+patchsize; r < rows-patchsize; ++r) {
        for (int c = 0+patchsize; c < cols-3*patchsize; ++c) {
            if (masks[i].at<uchar>(r, c) == 1)
                continue;
            float ref_depth = depths[i].at<float>(r, c);
            cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);
            if (ref_depth <= 0.0)
                continue;
            float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
            float3 consistent_Point = PointX;
            cv::Vec3f consistent_normal = ref_normal;
            float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
            int num_consistent = 0;
            PointList point3D;
            point3D.coord = consistent_Point;
            point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
            point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
            PointCloud.push_back(point3D);
        }
    }
    // }
    std::string ply_path;
    if(use_orig)
        ply_path = dense_folder + "/ACMP/ACMP_model_single_mask.ply";
    else if(use_fit)
        ply_path = dense_folder + "/ACMP/ACMP_model_single_mask_fit.ply";
    else
        ply_path = dense_folder + "/ACMP/ACMP_model_single_mask_comp.ply";

    std::cout<<ply_path<<"\n";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
}


void RunFusion(std::string &dense_folder, const std::vector<Problem> &problems, bool geom_consistency)
{
    size_t num_images = problems.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams_1");

    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    std::map<int,int> src_ids;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();
    
    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
        cv::Mat_<cv::Vec3b> image = cv::imread (image_path.str(), cv::IMREAD_COLOR);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());

        std::stringstream result_path;
        result_path << dense_folder << "/ACMP" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (geom_consistency) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        // std::cout <<problems[i].ref_image_id<<" "<<depths[i].rows<<" "<<depths[i].cols<<" "<<std::endl;
        masks.push_back(mask);
        src_ids[problems[i].ref_image_id]=i;
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();
    // num_images=1;
    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const int cols = depths[i].cols;
        const int rows = depths[i].rows;

        int num_ngb = problems[i].src_image_ids.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (masks[i].at<uchar>(r, c) == 1)
                    continue;
                float ref_depth = depths[i].at<float>(r, c);
                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);
                if (ref_depth <= 0.0)
                    continue;
                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
                int num_consistent = 0;
                if (consistent_Color[0] == 0.0 && consistent_Color[1] == 0.0 && consistent_Color[2] == 0.0 )
                    continue;
                for (int j = 0; j < num_ngb; ++j) {
                    int src_id = src_ids[problems[i].src_image_ids[j]];
                    const int src_cols = depths[src_id].cols;
                    const int src_rows = depths[src_id].rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_id].at<uchar>(src_r, src_c) == 1)
                            continue;
                            
                        float src_depth = depths[src_id].at<float>(src_r, src_c);
                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;
                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);
                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                            consistent_Point.x += tmp_X.x;
                            consistent_Point.y += tmp_X.y;
                            consistent_Point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_Color[0] += images[src_id].at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_Color[1] += images[src_id].at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_Color[2] += images[src_id].at<cv::Vec3b>(src_r, src_c)[2];

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;
                            num_consistent++;
                        }
                    }
                }
                if (num_consistent >= 2) {
                    consistent_Point.x /= (num_consistent + 1.0f);
                    consistent_Point.y /= (num_consistent + 1.0f);
                    consistent_Point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_Color[0] /= (num_consistent + 1.0f);
                    consistent_Color[1] /= (num_consistent + 1.0f);
                    consistent_Color[2] /= (num_consistent + 1.0f);

                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks[src_ids[problems[i].src_image_ids[j]]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }
    }

    std::string ply_path = dense_folder + "/ACMP/ACMP_model.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "USAGE: ACMP dense_folder" << std::endl;
        return -1;
    }
    InitCUDA();
    std::string dense_folder = argv[1];
    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems);

    std::string output_folder = dense_folder + std::string("/ACMP");
    mkdir(output_folder.c_str(), 0777);

    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

    bool geom_consistency = false;
    bool planar_prior = false; //* true
    bool multi_geometry = false;
    int geom_iterations = 2;
    // num_images=1;
    // for (size_t i = 0; i < num_images; ++i) {
    //     if(i!=14) continue;
    //     ProcessProblem(dense_folder, problems[i], geom_consistency, planar_prior);
    // }
    // geom_consistency = true;
    // planar_prior = false;
    // for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
    //     if (geom_iter == 0) {
    //         multi_geometry = false;
    //     }
    //     else {
    //         multi_geometry = true;
    //     }
    //     for (size_t i = 0; i < num_images; ++i) {
    //         ProcessProblem(dense_folder, problems[i], geom_consistency, planar_prior, multi_geometry);
    //     }
    // }

    // RunFusion(dense_folder, problems, geom_consistency);
    single_depth(dense_folder, problems, geom_consistency);


    return 0;
}
