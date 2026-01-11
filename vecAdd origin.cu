#include <cstdio>
#include <cstdlib>

#define CHECK(call) do {                                   \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",   \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                      \
    }                                                      \
} while (0)

__global__ void add_kernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 64;
    const int TPB = 32;
    const size_t BYTES = N * sizeof(int);

    // @@ 1. Allocate in host memory  在主机内存中分配
    int *h_a = (int*)malloc(BYTES);
    int *h_b = (int*)malloc(BYTES);
    int *h_c = (int*)malloc(BYTES);

    // @@ 3. Initialize host memory  初始化主机内存
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;          // 0,1,2,...
        h_b[i] = 100 + i;    // 100,101,...
        h_c[i] = 0;
    }

    // @@ 2. Allocate in device memory  在设备内存中分配
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK(cudaMalloc((void**)&d_a, BYTES));
    CHECK(cudaMalloc((void**)&d_b, BYTES));
    CHECK(cudaMalloc((void**)&d_c, BYTES));

    // @@ 4. Copy from host memory to device memory  从主机内存拷贝到设备内存
    CHECK(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice));

    // @@ 5. Initialize thread block and thread grid  初始化线程块和线程网格
    dim3 block(TPB);
    dim3 grid((N + TPB - 1) / TPB);

    // @@ 6. Invoke the CUDA kernel  调用 CUDA 内核
    add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // @@ 7. Copy results from GPU to CPU  将结果从 GPU 拷到 CPU
    CHECK(cudaMemcpy(h_c, d_c, BYTES, cudaMemcpyDeviceToHost));

    // @@ 8. Compare the results with the CPU reference result  与 CPU 参考结果对比
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        int ref = h_a[i] + h_b[i];
        if (h_c[i] != ref) {
            std::fprintf(stderr, "mismatch at %d: got %d, expect %d\n", i, h_c[i], ref);
            ok = false; break;
        }
    }
    std::printf("Result %s\n", ok ? "PASSED" : "FAILED");
    std::printf("Sample: c[0]=%d, c[%d]=%d\n", h_c[0], N-1, h_c[N-1]);

    // @@ 9. Free host memory  释放主机内存
    free(h_a); free(h_b); free(h_c);

    // @@ 10. Free device memory  释放设备内存
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return ok ? 0 : 1;
}
