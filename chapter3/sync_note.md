CUDA中的显式同步
1. device synchronize 影响整个gpu和cpu
2. stream synchronize 影响单个流和CPU
3. event synchronize 影响CPU
4. cudaStreamWaitEvent 不同流之间的event进行同步

