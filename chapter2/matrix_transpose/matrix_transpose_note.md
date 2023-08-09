# cuda 合并访存(访问/存储)
连续的线程读取连续的数据，通过global memory中的读写效率会得到提高
将shared memory当作缓存

shared memory可以用于:
1. 矩阵乘法中，减少从global memory读取相同元素的次数，将读数据转换到延迟更低的shared memory中
2. 矩阵转置中用于减少从全局内存读写非连续访问导致的合并访存问题

bank conflict
合并访存
warp的32个线程在访问(SIMD)连续的32个内存数据时，能达到较高效率
在shared_memory中，连续的32个内存单元分布在32个bank中，若同一个warp的不同线程访问同一个bank中的数据时，会引起bank conflict, 导致需要多个内存事务