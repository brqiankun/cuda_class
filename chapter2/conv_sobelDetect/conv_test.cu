#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#define SWITCH 0

// sobel 边缘检测/卷积
//kernel x00 x01 x02                m00 m01 m02 m03 m04            
//       x10 x11 x12                m10 m11 m12 m13 m14
//       x20 x21 x22                m20 m21 m22 m23 m24
//                                  m30 m31 m32 m33 m34
//                                  m40 m41 m42 m43 m44
//                                     卷积结果
//                                  r00 r01 r02 r03 r04            
//                                  r10 r11 r12 r13 r14
//                                  r20 r21 r22 r23 r24
//                                  r30 r31 r32 r33 r34
//                                  r40 r41 r42 r43 r44
//  卷积操作也是乘加
//  r11 = x00 * m00 + x01 * m01 + x02 * m02 + x10 * m10 + x11 * m11 + x12 * m12 + x20 * m20 + x21 * m21 + x22 * m22
// sobel 边缘检测的卷积核为
//Gx 1 0 -1   Gy  1  2  1
//   2 0 -2       0  0  0
//   1 0 -1       -1 -2 -1
// Gx_r11 = m00 + 2 * m10 + m20 - m02 - 2 * m12 - m22
// Gy_r11 = m00 + 2 * m01 + m02 - m20 - 2 * m21 - m22
// r11 = (Gx_r11 + Gy_r11) / 2
// 坐标索引和线程索引统一使用图像坐标系(向下为x，向右为y，坐标原点在图像左上角)

__global__ void SobelKernel(unsigned char* input, unsigned char* output, const int H, const int W) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int Gx = 0;
  int Gy = 0;
  unsigned char x0, x1, x2, x3, x4, x5, x6, x7, x8;
  // 每个线程计算得到卷积后图像的对应位置的像素的卷积值
  if (x > 0 && x < H - 1 && y > 0 && y < W - 1) {   // 最外圈的不进行计算(不进行padding，所以不进行计算)
    x0 = input[(x - 1) * W + y - 1];    // x0 ~ x8 分别是输入图像对应卷积核大小的对应位置的元素
    x1 = input[(x - 1) * W + y];
    x2 = input[(x - 1) * W + y + 1];
    x3 = input[x * W + y - 1];
    x4 = input[x * W + y];
    x5 = input[x * W + y + 1];
    x6 = input[(x + 1) * W + y - 1];
    x7 = input[(x + 1) * W + y];
    x8 = input[(x + 1) * W + y + 1];

    Gx = x0 + 2 * x3 + x6 - x2 - 2 * x5 - x8;
    Gy = x0 + 2 * x1 + x2 - x6 - 2 * x7 - x8;
#if SWITCH == 0
    output[x * W + y] = (abs(Gx) + abs(Gy)) / 2;
#elif SWITCH == 1
    output[x * W + y] = sqrtf((Gx * Gx) + (Gy * Gy));
#endif
  }
}

int main() {
  cv::Mat img = cv::imread("/home/br/program/cuda_class/chapter2/conv_sobelDetect/lena.jpg", 0);
  int h = img.rows;
  int w = img.cols;
  cv::Mat gaussImg;
  cv::GaussianBlur(img, gaussImg, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
  cv::Mat dst_gpu(h, w, CV_8UC1, cv::Scalar(0));
  int memsize = h * w * sizeof(unsigned char);
  unsigned char* i_d;
  unsigned char* o_d;
  cudaMalloc((void**)&i_d, memsize);
  cudaMalloc((void**)&o_d, memsize);
  dim3 block(32, 32);
  dim3 grid((h + block.x - 1) / block.x, (w + block.y - 1) / block.y);

  cudaMemcpy(i_d, gaussImg.data, memsize, cudaMemcpyHostToDevice);
  SobelKernel<<<grid, block>>>(i_d, o_d, h, w);
  cudaMemcpy(dst_gpu.data, o_d, memsize, cudaMemcpyDeviceToHost);
#if SWITCH == 0
  cv::imwrite("./res0.jpg", dst_gpu);
#elif SWITCH == 1
  cv::imwrite("./res1.jpg", dst_gpu);
#endif
  cudaFree(i_d);
  cudaFree(o_d);
  return 0;
}



