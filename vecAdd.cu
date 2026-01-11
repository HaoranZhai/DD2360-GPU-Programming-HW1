#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

int main(int argc, char** argv) {
    int N = 64;                        // default N
    if (argc > 1) N = std::atoi(argv[1]);

    const int TPB = 32;
    const size_t BYTES = static_cast<size_t>(N) * sizeof(int);

    // 1. host memory
    int *h_a = (int*)std::malloc(BYTES);
    int *h_b = (int*)std::malloc(BYTES);
    int *h_c = (int*)std::malloc(BYTES);
    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }

    // 3. init host
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 100 + i;
        h_c[i] = 0;
    }

    // 2. device memory
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK(cudaMalloc((void**)&d_a, BYTES));
    CHECK(cudaMalloc((void**)&d_b, BYTES));
    CHECK(cudaMalloc((void**)&d_c, BYTES));

    // grid/block
    dim3 block(TPB);
    dim3 grid((N + TPB - 1) / TPB);

    // timing events
    cudaEvent_t t0, t1, t2, t3;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventCreate(&t2));
    CHECK(cudaEventCreate(&t3));

    // 4. H2D
    CHECK(cudaEventRecord(t0));
    CHECK(cudaMemcpy(d_a, h_a, BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));

    // 6. kernel
    CHECK(cudaEventRecord(t1));
    add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(t2));
    CHECK(cudaEventSynchronize(t2));

    // 7. D2H
    CHECK(cudaEventRecord(t2));
    CHECK(cudaMemcpy(h_c, d_c, BYTES, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(t3));
    CHECK(cudaEventSynchronize(t3));

    // 8. check
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        int ref = h_a[i] + h_b[i];
        if (h_c[i] != ref) {
            std::fprintf(stderr, "mismatch at %d: got %d, expect %d\n", i, h_c[i], ref);
            ok = false; break;
        }
    }
    std::printf("Result %s\n", ok ? "PASSED" : "FAILED");
    if (N > 0) std::printf("Sample: c[0]=%d, c[%d]=%d\n", h_c[0], N-1, h_c[N-1]);

    // timings (ms)
    float ms_h2d = 0.f, ms_kernel = 0.f, ms_d2h = 0.f;
    CHECK(cudaEventElapsedTime(&ms_h2d,  t0, t1));
    CHECK(cudaEventElapsedTime(&ms_kernel,t1, t2));
    CHECK(cudaEventElapsedTime(&ms_d2h,  t2, t3));
    std::printf("CSV,%d,%.6f,%.6f,%.6f,%.6f\n",
                N, ms_h2d, ms_kernel, ms_d2h, ms_h2d + ms_kernel + ms_d2h);

    // cleanup
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHECK(cudaEventDestroy(t2));
    CHECK(cudaEventDestroy(t3));

    std::free(h_a); std::free(h_b); std::free(h_c);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return ok ? 0 : 1;
}
