from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx] = a[idx] + b[idx]
@cuda.jit
def matmul(A, B, C):
    """  矩阵乘法 C = A * B
    """
    # Numba库提供了更简易的计算方法
    # x, y = cuda.grid(2)
    # 具体计算公式如下
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y


    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

# thread per block
# 每个block有 BLOCK_SIZE x BLOCK_SIZE 个元素
BLOCK_SIZE = 16
@cuda.jit
def matmul_shared_memory(A, B, C):
    """
    使用Shared Memory的矩阵乘法 C = A * B
    """
    # 在Shared Memory中定义向量
    # 向量可被整个Block的所有Thread共享
    # 必须声明向量大小和数据类型
    sA = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    sB = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    if row >= C.shape[0] and col >= C.shape[1]:
        # 当(x, y)越界时退出
        return

    tmp = 0.
    # 以一个 BLOCK_SIZE x BLOCK_SIZE 为单位
    for m in range(math.ceil(A.shape[1] / BLOCK_SIZE)):
        sA[tx, ty] = A[row, ty + m * BLOCK_SIZE]
        sB[tx, ty] = B[tx + m * BLOCK_SIZE, col]
        # 线程同步，等待Block中所有Thread预加载结束
        # 该函数会等待所有Thread执行完之后才执行下一步
        cuda.syncthreads()
        # 此时已经将A和B的子矩阵拷贝到了sA和sB

        # 计算Shared Memory中的向量点积
        # 直接从Shard Memory中读取数据的延迟很低
        for n in range(BLOCK_SIZE):
            tmp += sA[tx, n] * sB[n, ty]

        # 线程同步，等待Block中所有Thread计算结束
        cuda.syncthreads()

    # 循环后得到每个BLOCK的点积之和
    C[row, col] = tmp


def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    start = time()
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    out_device = cuda.device_array(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)

    # 使用默认流
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, out_device, n)
    gpu_result = out_device.copy_to_host()
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))

    start = time()

    # 使用5个stream
    number_of_streams = 5
    # 每个stream处理的数据量为原来的 1/5
    # 符号//得到一个整数结果
    segment_size = n // number_of_streams

    # 创建5个cuda stream
    stream_list = list()
    for i in range (0, number_of_streams):
        stream = cuda.stream()
        stream_list.append(stream)

    threads_per_block = 1024
    # 每个stream的处理的数据变为原来的1/5
    blocks_per_grid = math.ceil(segment_size / threads_per_block)
    streams_out_device = cuda.device_array(segment_size)
    streams_gpu_result = np.empty(n)

    # 启动多个stream
    for i in range(0, number_of_streams):
        # 传入不同的参数，让函数在不同的流执行
        x_i_device = cuda.to_device(x[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])
        y_i_device = cuda.to_device(y[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])

        gpu_add[blocks_per_grid, threads_per_block, stream_list[i]](
                x_i_device, 
                y_i_device, 
                streams_out_device,
                segment_size)

        streams_gpu_result[i * segment_size : (i + 1) * segment_size] = streams_out_device.copy_to_host(stream=stream_list[i])

    cuda.synchronize()
    print("gpu streams vector add time " + str(time() - start))



    # 初始化矩阵
    M = 6000
    N = 4800
    P = 4000
    A = np.random.random((M, N)) # 随机生成的 [M x N] 矩阵
    B = np.random.random((N, P)) # 随机生成的 [N x P] 矩阵

    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((M, P)) # [M x P] 矩阵

    # 执行配置
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(math.ceil(A.shape[0] / BLOCK_SIZE))
    blocks_per_grid_y = int(math.ceil(B.shape[1] / BLOCK_SIZE))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start = time()
    matmul[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    print("matmul time :" + str(time() - start))

    start = time()
    matmul_shared_memory[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    print("matmul with shared memory time :" + str(time() - start))
    C = C_device.copy_to_host()

if __name__ == "__main__":
    main()
