// matmul.cu  -- Naive 2D Dense Matrix Multiplication
// Features:
//  - CUDA events timing (CSV,...)
//  - CPU chrono timing (CPUCSV,...)
//  - Optional FAST_CHECK (sampling verification)
//  - Optional USE_DOUBLE (compile-time)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// ---------- switches ----------
// define at compile time if needed, e.g. -DUSE_DOUBLE=1 -DFAST_CHECK=1
#ifndef USE_DOUBLE
#define USE_DOUBLE 0
#endif
#ifndef FAST_CHECK
#define FAST_CHECK 0
#endif

#if USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif

#define CHECK(call) do {                                   \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",   \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                      \
    }                                                      \
} while (0)

// C[m x n] = A[m x k] * B[k x n]   (row-major)
__global__ void matmul_naive(const real_t* __restrict__ A,
                             const real_t* __restrict__ B,
                             real_t* __restrict__ C,
                             int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..m)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..n)
    if (row < m && col < n) {
        real_t sum = (real_t)0;
        #pragma unroll 1
        for (int t = 0; t < k; ++t) {
            sum += A[row * (size_t)k + t] * B[t * (size_t)n + col];
        }
        C[row * (size_t)n + col] = sum;
    }
}

// CPU reference (same arithmetic order)
static void matmul_cpu_ref(const real_t* A, const real_t* B, real_t* C,
                           int m, int k, int n)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            real_t s = (real_t)0;
            for (int t = 0; t < k; ++t) {
                s += A[i * (size_t)k + t] * B[t * (size_t)n + j];
            }
            C[i * (size_t)n + j] = s;
        }
    }
}

int main(int argc, char** argv)
{
    // A(m x k) * B(k x n) = C(m x n)
    int m = 128, k = 256, n = 32;      // defaults for Q3 case-1
    int BX = 16, BY = 16;              // default block size
    if (argc >= 4) { m = std::atoi(argv[1]); k = std::atoi(argv[2]); n = std::atoi(argv[3]); }
    if (argc >= 6) { BX = std::atoi(argv[4]); BY = std::atoi(argv[5]); if (BX<=0) BX=16; if (BY<=0) BY=16; }

    std::printf("Type: %s, m=%d, k=%d, n=%d, block=(%d,%d)\n",
#if USE_DOUBLE
        "double",
#else
        "float",
#endif
        m, k, n, BX, BY);

    const size_t bytesA = (size_t)m * k * sizeof(real_t);
    const size_t bytesB = (size_t)k * n * sizeof(real_t);
    const size_t bytesC = (size_t)m * n * sizeof(real_t);

    // host memory
    real_t *h_A = (real_t*)std::malloc(bytesA);
    real_t *h_B = (real_t*)std::malloc(bytesB);
    real_t *h_C = (real_t*)std::malloc(bytesC);
    real_t *h_Cref = (real_t*)std::malloc(bytesC);
    if (!h_A || !h_B || !h_C || !h_Cref) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }

    // deterministic init (small magnitudes to avoid overflow)
    for (int i = 0; i < m; ++i)
        for (int t = 0; t < k; ++t)
            h_A[i*(size_t)k + t] = (real_t)((i + t) % 13) / (real_t)13;

    for (int t = 0; t < k; ++t)
        for (int j = 0; j < n; ++j)
            h_B[t*(size_t)n + j] = (real_t)((t + j) % 17) / (real_t)17;

    for (int i = 0; i < m * n; ++i) h_C[i] = (real_t)0;

    // device memory
    real_t *d_A=nullptr, *d_B=nullptr, *d_C=nullptr;
    CHECK(cudaMalloc((void**)&d_A, bytesA));
    CHECK(cudaMalloc((void**)&d_B, bytesB));
    CHECK(cudaMalloc((void**)&d_C, bytesC));

    // grid/block
    dim3 block(BX, BY);
    dim3 grid( (n + block.x - 1) / block.x,
               (m + block.y - 1) / block.y );

    // CUDA events
    cudaEvent_t t0, t1, t2, t3;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventCreate(&t2));
    CHECK(cudaEventCreate(&t3));

    // ---------- CPU timers (host perspective) ----------
    using clock_t = std::chrono::high_resolution_clock;
    auto ms = [](auto dt){ return std::chrono::duration<double, std::milli>(dt).count(); };

    // H2D  (CPU + events)
    auto c0 = clock_t::now();
    CHECK(cudaEventRecord(t0));
    CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    auto c1 = clock_t::now();

    // Kernel (CPU + events)
    auto c2 = clock_t::now();
    CHECK(cudaEventRecord(t1));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(t2));
    CHECK(cudaEventSynchronize(t2));
    auto c3 = clock_t::now();

    // D2H  (CPU + events)
    auto c4 = clock_t::now();
    CHECK(cudaEventRecord(t2));
    CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(t3));
    CHECK(cudaEventSynchronize(t3));
    auto c5 = clock_t::now();

    // ---------- verification ----------
    double max_abs = 0.0, max_rel = 0.0;
#if FAST_CHECK
    // Sampling verification to avoid long CPU time on large sizes
    int samples = 64;
    unsigned seed = 1315423911u;
    auto next_u = [&](){ seed ^= seed<<5; seed ^= seed>>7; seed ^= seed<<22; return seed; };
    for (int s = 0; s < samples; ++s) {
        int i = (int)(next_u() % (unsigned)m);
        int j = (int)(next_u() % (unsigned)n);
        double ref = 0.0;
        for (int t = 0; t < k; ++t)
            ref += (double)h_A[i*(size_t)k + t] * (double)h_B[t*(size_t)n + j];
        double got = (double)h_C[i*(size_t)n + j];
        double abs_err = std::fabs(got - ref);
        double rel_err = abs_err / (std::fabs(ref) + 1e-12);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
    }
    bool ok = (max_rel < 1e-4) || (max_abs < 1e-5);
#else
    // Full CPU reference (good for small sizes)
    matmul_cpu_ref(h_A, h_B, h_Cref, m, k, n);
    for (int i = 0; i < m * n; ++i) {
        double a = (double)h_C[i];
        double b = (double)h_Cref[i];
        double abs_err = std::fabs(a - b);
        double rel_err = abs_err / (std::fabs(b) + 1e-12);
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
    }
    bool ok = (max_rel < 1e-4) || (max_abs < 1e-5);
#endif

    std::printf("Result %s  (max_abs=%.3e, max_rel=%.3e)\n",
                ok ? "PASSED" : "FAILED", max_abs, max_rel);

    // ---------- timings (events) ----------
    float ms_h2d=0.f, ms_kernel=0.f, ms_d2h=0.f;
    CHECK(cudaEventElapsedTime(&ms_h2d,  t0, t1));
    CHECK(cudaEventElapsedTime(&ms_kernel,t1, t2));
    CHECK(cudaEventElapsedTime(&ms_d2h,  t2, t3));

    const double flops = 2.0 * (double)m * (double)k * (double)n;
    const double gflops = (ms_kernel > 0.0f)
        ? (flops / (ms_kernel/1000.0) / 1e9) : 0.0;

    std::printf("H2D(ms)=%.6f  Kernel(ms)=%.6f  D2H(ms)=%.6f  Total(ms)=%.6f\n",
                ms_h2d, ms_kernel, ms_d2h, ms_h2d + ms_kernel + ms_d2h);
    std::printf("Kernel GFLOP/s (approx) = %.3f\n", gflops);
    // device-side CSV (events)
    std::printf("CSV,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.3f\n",
                m, k, n, ms_h2d, ms_kernel, ms_d2h,
                ms_h2d + ms_kernel + ms_d2h, gflops);

    // ---------- timings (CPU wall-clock) ----------
    double cpu_h2d_ms   = std::chrono::duration<double,std::milli>(c1 - c0).count();
    double cpu_kernel_ms= std::chrono::duration<double,std::milli>(c3 - c2).count();
    double cpu_d2h_ms   = std::chrono::duration<double,std::milli>(c5 - c4).count();
    double cpu_total_ms = cpu_h2d_ms + cpu_kernel_ms + cpu_d2h_ms;

    // host-side CSV (for Q5/6 stacked bar chart)
    std::printf("CPUCSV,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
#if USE_DOUBLE
                "double",
#else
                "float",
#endif
                m, k, n, cpu_h2d_ms, cpu_kernel_ms, cpu_d2h_ms, cpu_total_ms);

    // cleanup
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHECK(cudaEventDestroy(t2));
    CHECK(cudaEventDestroy(t3));
    std::free(h_A); std::free(h_B); std::free(h_C); std::free(h_Cref);
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));

    return ok ? 0 : 1;
}
