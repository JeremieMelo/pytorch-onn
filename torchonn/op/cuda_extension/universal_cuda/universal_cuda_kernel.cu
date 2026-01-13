/*
 * @Author: Jiaqi Gu
 * @Date: 2020-06-04 14:16:01
 * @LastEditors: Jiaqi Gu (jqgu@utexas.edu)
 * @LastEditTime: 2021-11-29 03:01:34
 */

/* Adated from the CUDA samples
  https://docs.nvidia.com/cuda/cuda-samples/index.html. Changed from "natural
  order" Hadamard transform (larger strides before smaller strides) to the
  standard Hadamard transform (smaller strides before larger strides).
*/

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename T> struct ComplexType {
  T x;
  T y;
  __host__ __device__ ComplexType() {
    x = 0;
    y = 0;
  }

  __host__ __device__ ComplexType(T real, T imag) {
    x = real;
    y = imag;
  }

  __host__ __device__ ~ComplexType() {}
};

template <typename T>
inline __host__ __device__ ComplexType<T> complexMul(const ComplexType<T> &x,
                                                     const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = x.x * y.y + x.y * y.x;
  return res;
}

template <typename T>
inline __host__ __device__ T RealPartOfMul(const ComplexType<T> &x,
                                           const ComplexType<T> &y) {
  return x.x * y.x - x.y * y.y;
}

template <typename T>
inline __host__ __device__ T ImaginaryPartOfMul(const ComplexType<T> &x,
                                                const ComplexType<T> &y) {
  return x.x * y.y + x.y * y.x;
}

template <typename T>
inline __host__ __device__ ComplexType<T> complexAdd(const ComplexType<T> &x,
                                                     const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x + y.x;
  res.y = x.y + y.y;
  return res;
}

template <typename T>
inline __host__ __device__ ComplexType<T>
complexSubtract(const ComplexType<T> &x, const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x - y.x;
  res.y = x.y - y.y;
  return res;
}

template <typename T>
inline __host__ __device__ ComplexType<T> complexConj(const ComplexType<T> &x) {
  ComplexType<T> res;
  res.x = x.x;
  res.y = -x.y;
  return res;
}

template <typename T>
inline __host__ __device__ ComplexType<T>
complexMulConj(const ComplexType<T> &x, const ComplexType<T> &y) {
  ComplexType<T> res;
  res.x = x.x * y.x - x.y * y.y;
  res.y = -(x.x * y.y + x.y * y.x);
  return res;
}

///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

__global__ void uftBatch1Kernel(ComplexType<float> *d_Output,
                                ComplexType<float> *d_Input, int log2N) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  const int N = 1 << log2N;
  const int base = blockIdx.x << log2N;

  //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
  extern __shared__ ComplexType<float> s_data[];
  ComplexType<float> *d_Src = d_Input + base;
  ComplexType<float> *d_Dst = d_Output + base;

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    s_data[pos] = d_Src[pos];
  }

  int stride = 1;
  // Do single radix-2 stage for odd power of two
  if (log2N & 1) {
    cg::sync(cta);

    for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
      int i0 = pos << 1;
      int i1 = i0 + 1;

      //  float D0 = s_data[i0];
      //  float D1 = s_data[i1];
      //  s_data[i0] = D0 + D1;
      //  s_data[i1] = D0 - D1;
      ComplexType<float> &D0 = s_data[i0];
      ComplexType<float> &D1 = s_data[i1];
      float D0_real_new = D0.x - D1.y;
      float D0_imag_new = D0.y + D1.x;

      float D1_real_new = D1.x - D0.y;
      float D1_imag_new = D0.x + D1.y;
      D0.x = D0_real_new;
      D0.y = D0_imag_new;
      D1.x = D1_real_new;
      D1.y = D1_imag_new;
    }
    stride <<= 1;
  }

  // Main radix-4 stages
  const int pos = threadIdx.x;

  for (; stride <= N >> 2; stride <<= 2) {
    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    cg::sync(cta);
    //  float D0 = s_data[i0];
    //  float D1 = s_data[i1];
    //  float D2 = s_data[i2];
    //  float D3 = s_data[i3];

    //  float T;
    //  T = D0;
    //  D0         = D0 + D2;
    //  D2         = T - D2;
    //  T = D1;
    //  D1         = D1 + D3;
    //  D3         = T - D3;
    //  T = D0;
    //  s_data[i0] = D0 + D1;
    //  s_data[i1] = T - D1;
    //  T = D2;
    //  s_data[i2] = D2 + D3;
    //  s_data[i3] = T - D3;

    ComplexType<float> &D0 = s_data[i0];
    ComplexType<float> &D1 = s_data[i1];
    ComplexType<float> &D2 = s_data[i2];
    ComplexType<float> &D3 = s_data[i3];

    float D0_real_new = D0.x - D2.y;
    float D0_imag_new = D0.y + D2.x;
    float D2_real_new = D2.x - D0.y;
    float D2_imag_new = D0.x + D2.y;

    float D1_real_new = D1.x - D3.y;
    float D1_imag_new = D1.y + D3.x;
    float D3_real_new = D3.x - D1.y;
    float D3_imag_new = D1.x + D3.y;

    float D0_real_new_2 = D0_real_new - D1_imag_new;
    float D0_imag_new_2 = D0_imag_new + D1_real_new;
    float D1_real_new_2 = D1_real_new - D0_imag_new;
    float D1_imag_new_2 = D0_real_new + D1_imag_new;

    float D2_real_new_2 = D2_real_new - D3_imag_new;
    float D2_imag_new_2 = D2_imag_new + D3_real_new;
    float D3_real_new_2 = D3_real_new - D2_imag_new;
    float D3_imag_new_2 = D2_real_new + D3_imag_new;

    D0.x = D0_real_new_2;
    D0.y = D0_imag_new_2;

    D1.x = D1_real_new_2;
    D1.y = D1_imag_new_2;

    D2.x = D2_real_new_2;
    D2.y = D2_imag_new_2;

    D3.x = D3_real_new_2;
    D3.y = D3_imag_new_2;
  }

  cg::sync(cta);

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    d_Dst[pos] = s_data[pos];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ void uftBatch2Kernel(ComplexType<float> *d_Output,
                                ComplexType<float> *d_Input, int stride) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = blockDim.x * gridDim.x * 4;

  ComplexType<float> *d_Src = d_Input + blockIdx.y * N;
  ComplexType<float> *d_Dst = d_Output + blockIdx.y * N;

  int lo = pos & (stride - 1);
  int i0 = ((pos - lo) << 2) + lo;
  int i1 = i0 + stride;
  int i2 = i1 + stride;
  int i3 = i2 + stride;

  ComplexType<float> &D0 = d_Src[i0];
  ComplexType<float> &D1 = d_Src[i1];
  ComplexType<float> &D2 = d_Src[i2];
  ComplexType<float> &D3 = d_Src[i3];

  // float T;
  // T = D0;
  // D0        = D0 + D2;
  // D2        = T - D2;
  // T = D1;
  // D1        = D1 + D3;
  // D3        = T - D3;
  // T = D0;
  // d_Dst[i0] = D0 + D1;
  // d_Dst[i1] = T - D1;
  // T = D2;
  // d_Dst[i2] = D2 + D3;
  // d_Dst[i3] = T - D3;

  float D0_real_new = D0.x - D2.y;
  float D0_imag_new = D0.y + D2.x;
  float D2_real_new = D2.x - D0.y;
  float D2_imag_new = D0.x + D2.y;

  float D1_real_new = D1.x - D3.y;
  float D1_imag_new = D1.y + D3.x;
  float D3_real_new = D3.x - D1.y;
  float D3_imag_new = D1.x + D3.y;

  float D0_real_new_2 = D0_real_new - D1_imag_new;
  float D0_imag_new_2 = D0_imag_new + D1_real_new;
  float D1_real_new_2 = D1_real_new - D0_imag_new;
  float D1_imag_new_2 = D0_real_new + D1_imag_new;

  float D2_real_new_2 = D2_real_new - D3_imag_new;
  float D2_imag_new_2 = D2_imag_new + D3_real_new;
  float D3_real_new_2 = D3_real_new - D2_imag_new;
  float D3_imag_new_2 = D2_real_new + D3_imag_new;

  D0.x = D0_real_new_2;
  D0.y = D0_imag_new_2;

  D1.x = D1_real_new_2;
  D1.y = D1_imag_new_2;

  D2.x = D2_real_new_2;
  D2.y = D2_imag_new_2;

  D3.x = D3_real_new_2;
  D3.y = D3_imag_new_2;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void uftBatchGPU(float *d_Data_raw, int batchSize, int log2N) {
  ComplexType<float> *d_Data = (ComplexType<float> *)d_Data_raw;
  int nMixedRadixPasses =
      log2N > ELEMENTARY_LOG2SIZE
          ? ELEMENTARY_LOG2SIZE - (log2N - ELEMENTARY_LOG2SIZE) % 2
          : log2N;
  int N = 1 << nMixedRadixPasses;
  int curBatchSize = batchSize << (log2N - nMixedRadixPasses);

  uftBatch1Kernel<<<curBatchSize, N / 4, N * sizeof(float)>>>(
      d_Data, d_Data, nMixedRadixPasses);

  const int THREAD_N = 256;
  dim3 grid((1 << log2N) / (4 * THREAD_N), batchSize, 1);

  for (int logSize = nMixedRadixPasses + 2; logSize <= log2N; logSize += 2) {
    uftBatch2Kernel<<<grid, THREAD_N>>>(d_Data, d_Data, (1 << logSize) / 4);
  }
}

__global__ void iuftBatch1Kernel(ComplexType<float> *d_Output,
                                 ComplexType<float> *d_Input, int log2N) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  const int N = 1 << log2N;
  const int base = blockIdx.x << log2N;

  //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
  extern __shared__ ComplexType<float> s_data[];
  ComplexType<float> *d_Src = d_Input + base;
  ComplexType<float> *d_Dst = d_Output + base;

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    s_data[pos] = d_Src[pos];
  }

  int stride = 1;
  // Do single radix-2 stage for odd power of two
  if (log2N & 1) {
    cg::sync(cta);

    for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
      int i0 = pos << 1;
      int i1 = i0 + 1;

      //  float D0 = s_data[i0];
      //  float D1 = s_data[i1];
      //  s_data[i0] = D0 + D1;
      //  s_data[i1] = D0 - D1;
      ComplexType<float> &D0 = s_data[i0];
      ComplexType<float> &D1 = s_data[i1];
      float D0_real_new = D0.x + D1.y;
      float D0_imag_new = D0.y - D1.x;

      float D1_real_new = D1.x + D0.y;
      float D1_imag_new = D1.y - D0.x;
      D0.x = D0_real_new;
      D0.y = D0_imag_new;
      D1.x = D1_real_new;
      D1.y = D1_imag_new;
    }
    stride <<= 1;
  }

  // Main radix-4 stages
  const int pos = threadIdx.x;

  for (; stride <= N >> 2; stride <<= 2) {
    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    cg::sync(cta);
    //  float D0 = s_data[i0];
    //  float D1 = s_data[i1];
    //  float D2 = s_data[i2];
    //  float D3 = s_data[i3];

    //  float T;
    //  T = D0;
    //  D0         = D0 + D2;
    //  D2         = T - D2;
    //  T = D1;
    //  D1         = D1 + D3;
    //  D3         = T - D3;
    //  T = D0;
    //  s_data[i0] = D0 + D1;
    //  s_data[i1] = T - D1;
    //  T = D2;
    //  s_data[i2] = D2 + D3;
    //  s_data[i3] = T - D3;

    ComplexType<float> &D0 = s_data[i0];
    ComplexType<float> &D1 = s_data[i1];
    ComplexType<float> &D2 = s_data[i2];
    ComplexType<float> &D3 = s_data[i3];

    float D0_real_new = D0.x + D2.y;
    float D0_imag_new = D0.y - D2.x;
    float D2_real_new = D2.x + D0.y;
    float D2_imag_new = D2.y - D0.x;

    float D1_real_new = D1.x + D3.y;
    float D1_imag_new = D1.y - D3.x;
    float D3_real_new = D3.x + D1.y;
    float D3_imag_new = D3.y - D1.x;

    float D0_real_new_2 = D0_real_new + D1_imag_new;
    float D0_imag_new_2 = D0_imag_new - D1_real_new;
    float D1_real_new_2 = D1_real_new + D0_imag_new;
    float D1_imag_new_2 = D1_imag_new - D0_real_new;

    float D2_real_new_2 = D2_real_new + D3_imag_new;
    float D2_imag_new_2 = D2_imag_new - D3_real_new;
    float D3_real_new_2 = D3_real_new + D2_imag_new;
    float D3_imag_new_2 = D3_imag_new - D2_real_new;

    D0.x = D0_real_new_2;
    D0.y = D0_imag_new_2;

    D1.x = D1_real_new_2;
    D1.y = D1_imag_new_2;

    D2.x = D2_real_new_2;
    D2.y = D2_imag_new_2;

    D3.x = D3_real_new_2;
    D3.y = D3_imag_new_2;
  }

  cg::sync(cta);

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    d_Dst[pos] = s_data[pos];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ void iuftBatch2Kernel(ComplexType<float> *d_Output,
                                 ComplexType<float> *d_Input, int stride) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = blockDim.x * gridDim.x * 4;

  ComplexType<float> *d_Src = d_Input + blockIdx.y * N;
  ComplexType<float> *d_Dst = d_Output + blockIdx.y * N;

  int lo = pos & (stride - 1);
  int i0 = ((pos - lo) << 2) + lo;
  int i1 = i0 + stride;
  int i2 = i1 + stride;
  int i3 = i2 + stride;

  ComplexType<float> &D0 = d_Src[i0];
  ComplexType<float> &D1 = d_Src[i1];
  ComplexType<float> &D2 = d_Src[i2];
  ComplexType<float> &D3 = d_Src[i3];

  // float T;
  // T = D0;
  // D0        = D0 + D2;
  // D2        = T - D2;
  // T = D1;
  // D1        = D1 + D3;
  // D3        = T - D3;
  // T = D0;
  // d_Dst[i0] = D0 + D1;
  // d_Dst[i1] = T - D1;
  // T = D2;
  // d_Dst[i2] = D2 + D3;
  // d_Dst[i3] = T - D3;

  float D0_real_new = D0.x - D2.y;
  float D0_imag_new = D0.y + D2.x;
  float D2_real_new = D2.x - D0.y;
  float D2_imag_new = D0.x + D2.y;

  float D1_real_new = D1.x - D3.y;
  float D1_imag_new = D1.y + D3.x;
  float D3_real_new = D3.x - D1.y;
  float D3_imag_new = D1.x + D3.y;

  float D0_real_new_2 = D0_real_new - D1_imag_new;
  float D0_imag_new_2 = D0_imag_new + D1_real_new;
  float D1_real_new_2 = D1_real_new - D0_imag_new;
  float D1_imag_new_2 = D0_real_new + D1_imag_new;

  float D2_real_new_2 = D2_real_new - D3_imag_new;
  float D2_imag_new_2 = D2_imag_new + D3_real_new;
  float D3_real_new_2 = D3_real_new - D2_imag_new;
  float D3_imag_new_2 = D2_real_new + D3_imag_new;

  D0.x = D0_real_new_2;
  D0.y = D0_imag_new_2;

  D1.x = D1_real_new_2;
  D1.y = D1_imag_new_2;

  D2.x = D2_real_new_2;
  D2.y = D2_imag_new_2;

  D3.x = D3_real_new_2;
  D3.y = D3_imag_new_2;
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void iuftBatchGPU(float *d_Data_raw, int batchSize, int log2N) {
  ComplexType<float> *d_Data = (ComplexType<float> *)d_Data_raw;
  int nMixedRadixPasses =
      log2N > ELEMENTARY_LOG2SIZE
          ? ELEMENTARY_LOG2SIZE - (log2N - ELEMENTARY_LOG2SIZE) % 2
          : log2N;
  int N = 1 << nMixedRadixPasses;
  int curBatchSize = batchSize << (log2N - nMixedRadixPasses);

  iuftBatch1Kernel<<<curBatchSize, N / 4, N * sizeof(ComplexType<float>)>>>(
      d_Data, d_Data, nMixedRadixPasses);

  const int THREAD_N = 256;
  dim3 grid((1 << log2N) / (4 * THREAD_N), batchSize, 1);

  for (int logSize = nMixedRadixPasses + 2; logSize <= log2N; logSize += 2) {
    iuftBatch2Kernel<<<grid, THREAD_N>>>(d_Data, d_Data, (1 << logSize) / 4);
  }
}
