// Bias of the tensor-core GEMM cubins (throwaway; S71): mean signed error of C = A @ B relative to sum_k |a_ik b_kj|,
// against a float64 host reference, for pedantic cuBLAS FP32 and tile 0 (B row-major, beta 0) of each cubin.
// build: nvcc -O3 -std=c++17 -arch=sm_120 tc_bias.cu -lcublas -lcuda -o tc_bias
// usage: tc_bias M N K CUBIN...
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { auto e = (x); if (e != 0) { fprintf(stderr, "%s:%d error %d\n", __FILE__, __LINE__, (int)e); exit(1); } } while (0)

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: tc_bias M N K CUBIN...\n"); return 1; }
  const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
  CK(cudaFree(0));
  cublasHandle_t cb; CK(cublasCreate(&cb));
  CK(cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH));
  std::mt19937 rng(7);
  for (int mode = 0; mode < 2; ++mode) {
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    const auto sample = [&]() {
      float x = mode ? nd(rng) : ud(rng);
#ifdef DIAGNOSTIC_TF32_INPUTS
      unsigned bits;
      std::memcpy(&bits, &x, sizeof(bits));
      bits = (bits + 0x1000u) & 0xffffe000u;
      std::memcpy(&x, &bits, sizeof(x));
#endif
      return x;
    };
    for (auto& x : A) x = sample();
    for (auto& x : B) x = sample();
    std::vector<double> ref((size_t)M * N), scale((size_t)M * N);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        double s = 0, a = 0;
        for (int k = 0; k < K; ++k) {
          const double p = (double)A[(size_t)i * K + k] * (double)B[(size_t)k * N + j];
          s += p; a += std::fabs(p);
        }
        ref[(size_t)i * N + j] = s; scale[(size_t)i * N + j] = a;
      }
    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, A.size() * 4)); CK(cudaMalloc(&dB, B.size() * 4)); CK(cudaMalloc(&dC, (size_t)M * N * 4));
    CK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> out((size_t)M * N);
    auto report = [&](const char* name) {
      CK(cudaMemcpy(out.data(), dC, out.size() * 4, cudaMemcpyDeviceToHost));
      double sum = 0, sq = 0, mx = 0;
      size_t nearest = 0, below = 0, above = 0;
      for (size_t i = 0; i < out.size(); ++i) {
        const double r = ((double)out[i] - ref[i]) / scale[i];
        sum += r; sq += r * r; mx = std::fmax(mx, std::fabs(r));
        nearest += out[i] == (float)ref[i];
        below += (double)out[i] < ref[i];
        above += (double)out[i] > ref[i];
      }
      const double n = (double)out.size();
      printf("  %-44s mean %+.3e  rms %.3e  max %.3e\n", name, sum / n, std::sqrt(sq / n), mx);
      #ifdef DIAGNOSTIC_TF32_INPUTS
      printf("    TF32-exact operands: nearest-rounded %zu/%zu, below exact %zu, above exact %zu\n",
             nearest, out.size(), below, above);
      #endif
    };
    printf("%s data, M %d N %d K %d (error / sum|a b|)\n", mode ? "symmetric normal" : "positive U[0.5,1.5)", M, N, K);
    const float one = 1.0f, zero = 0.0f;
    CK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, dB, N, dA, K, &zero, dC, N));
    CK(cudaDeviceSynchronize());
    report("cuBLAS FP32 (pedantic)");
    for (int c = 4; c < argc; ++c) {
      CUmodule mod; CK(cuModuleLoad(&mod, argv[c]));
      CUdeviceptr gi; size_t gsz; CK(cuModuleGetGlobal(&gi, &gsz, mod, "loup_gemm_tc_info"));
      std::vector<int> info(gsz / 4); CK(cuMemcpyDtoH(info.data(), gi, gsz));
      CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "loup_gemm_tc_rr_0"));
      const int tm = info[4], tn = info[5], threads = info[6], smem = info[7];
      CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
      int m = M, n = N, k = K, lda = K, ldb = N, ldc = N; float beta = 0.0f;
      void* args[] = {&m, &n, &k, &dA, &lda, &dB, &ldb, &dC, &ldc, &beta};
      CK(cuLaunchKernel(fn, (M + tm - 1) / tm, (N + tn - 1) / tn, 1, threads, 1, 1, smem, 0, args, nullptr));
      CK(cudaDeviceSynchronize());
      report(argv[c]);
      CK(cuModuleUnload(mod));
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
  }
  return 0;
}
