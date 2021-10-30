#include<opencv2/opencv.hpp>
#include<iostream>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include<cmath>
#include<time.h>

using namespace cv;
using namespace std;


__global__ void gaussian_kernel(uchar *d_img_in, uchar *d_img_out, float *d_arr,
                                const int img_cols, const int img_rows, const int size)
{
    const auto col_id = blockDim.x*threadIdx.y + threadIdx.x;
    const auto row_id = gridDim.x*blockIdx.y + blockIdx.x;
    if (col_id < img_cols - size && row_id < img_rows - size)
    {
        float sum{};
        for (int y = 0; y < size; ++y)
        {
            for (int x = 0; x < size; ++x)
            {
                sum += d_arr[y * size + x] * d_img_in[col_id + x + (row_id + y)* img_cols];
            }
        }
        d_img_out[col_id + row_id * img_cols] = (uchar)sum;
    }


}


void gaussian_cuda(const Mat &img_in, Mat &img_out, const int &size, const float &sigma, int block_size = 16)
{
    bool ifdebug = false;

    const int img_sizeof = img_in.cols*img_in.rows * sizeof(uchar);
    const int arr_sizeof = size * size * sizeof(float);
    img_out = Mat::zeros(img_in.size(), CV_8UC1);
    float *arr = (float*)malloc(size*size * sizeof(float));
    auto getGuassionArray = [&]()
    {
        float sum = 0.0;
        auto sigma_2 = sigma * sigma;
        for (int i{}; i < size; ++i)
        {
            auto dx = i - size;
            for (int j{}; j < size; ++j)
            {
                auto dy = j - size;
                arr[i * size + j] = exp(-(dx*dx + dy * dy) / (sigma_2 * 2));
                sum += arr[i * size + j];
            }
        }
        for (size_t i{}; i < size; ++i)
        {
            for (size_t j{}; j < size; ++j)
            {
                arr[i * size + j] /= sum;
            }
        }
    };
    getGuassionArray();

    if(ifdebug)
        for (int i{}; i < size; ++i)
        {
            for (int j{}; j < size; ++j)
                cout << arr[j + i * size] << " ";
            cout << endl;
        }

    float *d_arr;		//之后做成共享内存
    uchar *d_img_in;
    uchar *d_img_out;
    cudaMalloc(&d_arr, arr_sizeof);
    cudaMalloc(&d_img_in,img_sizeof);
    cudaMalloc(&d_img_out,img_sizeof);
    cudaMemcpy(d_arr, arr, arr_sizeof, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_in, img_in.data, img_sizeof, cudaMemcpyHostToDevice);
    dim3 grid;
    grid.x=(uint)ceil((double)img_in.rows / block_size);
    grid.y=(uint)block_size;
    grid.z=1;
    dim3 block;
    block.x=(uint)32;
    block.y=(uint)ceil((double)img_in.cols / 32);
    block.z=1;

    gaussian_kernel<<<grid,block>>>(d_img_in, d_img_out, d_arr, img_in.cols, img_in.rows, size);

    cudaMemcpy(img_out.data, d_img_out, img_sizeof, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    free(arr);
}

bool gaussianBlur_gpu(cv::Mat &src,cv::Mat &dst,int kernelType = 5)
{
	if (src.data == nullptr)
	{
		return false;
	}
	cv::cuda::GpuMat src_gpu, dst_gpu5x5;
	src_gpu.upload(src);
    //注意这个细节 cv::ptr， 这个东西就是个智能指针,shred_ptr<> 这个东西，别被迷惑了.
	cv::Ptr<cv::cuda::Filter> filter5x5,;
    //这里的这几个参数，大家编译好之后，直接看源码的注释，或者我这里献丑了，给大家记录一下.
	filter5x5 = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3,cv::Size(5,5),1);
	filter5x5->apply(src_gpu, dst_gpu5x5);
	dst_gpu5x5.download(dst);
	return true;
}

int main()
{

    auto img = imread("images/00000000.jpg", IMREAD_GRAYSCALE);
    auto img2 {Mat::zeros(33,33, CV_8UC1)};
    Mat gaussian;
    clock_t start,end;
    start=clock();
    gaussian_cuda(img, gaussian, 7, 100);
    end=clock();
	cout<<"运行时间"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    Mat out;
    start=clock();
    GaussianBlur(img, out, Size(7, 7), 100);
    end=clock();
	cout<<"运行时间"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    cv::Mat gaussiand_result=cv::Mat::zeros(img.cols, img.rows,CV_8UC3);
    start=clock();
	gaussianBlur_gpu(img,gaussiand_result);
    end=clock();
	cout<<"运行时间"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    imwrite("gaussian_cuda.jpg", gaussian);
    imwrite("gaussian_out.jpg", out);
    imwrite("gaussian_result.jpg", gaussiand_result);

}

