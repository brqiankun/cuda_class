import time

t1 = time.perf_counter()   #（以小数表示的秒为单位）返回一个性能计数器的值，即用于测量较短持续时间的具有最高有效精度的时钟。
i = 0
while i < 1e7:
    i = i + 1
t2 = time.perf_counter()
print("time: {} ms".format((t2 - t1) * 1000))