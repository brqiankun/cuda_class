TODO 完成conv kernel的编写和熟悉，完成b站的CUDA编程模型系列四的内容
```
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
```
sobel 边缘检测的原理是使用两个设计好的卷积核
Gx 1 0 -1      Gy 1  2  1
   2 0 -2         0  0  0
   1 0 -1         -1 -2 -1
两个卷积核分别计算后得到Gx_r11, Gy_r11
之后得到每个位置输出像素为r11 = (Gx_r11 + Gy_r11) / 2


// 1. 一个block中的线程数量不能超过1024; 2. 数量最好是32的倍数(warp size，一个warp的线程在同时刻内执行相同的指令)
// A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. 
// If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, 
// disabling threads that are not on that path. 
