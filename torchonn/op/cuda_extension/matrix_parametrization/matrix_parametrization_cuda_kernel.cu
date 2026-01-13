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

#include "cuda_runtime.h"

#define PI (3.141592653589793)
#define abs(x) (x >= 0 ? x : -x)
#define ERROR (1e-6)

template <typename T>
__global__ void unitaryRotation(const int N, T *row_p, T *row_q,
                                const T *cos_phi, const T *sin_phi) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  T new_row = 0;
  if (tid < 2 * N) {
    int i = tid / 2;
    if ((tid & 1) == 0)
      // p
      new_row = row_p[i] * (*cos_phi) - row_q[i] * (*sin_phi);
    else
      // q
      new_row = row_p[i] * (*sin_phi) + row_q[i] * (*cos_phi);
    __syncthreads();
    if ((tid & 1) == 0)
      // p
      row_p[i] = new_row;
    else
      // q
      row_q[i] = new_row;
  }
}

template <typename T>
void unitaryRotationCUDALauncher(const int N, const int num_block,
                                 const int num_thread, T *row_p, T *row_q,
                                 const T *cos_phi, const T *sin_phi) {
  unitaryRotation<T>
      <<<num_block, num_thread>>>(N, row_p, row_q, cos_phi, sin_phi);
  cudaDeviceSynchronize();
}

template <typename T>
__global__ void
unitaryReconstructFrancis(const int BS, const int N, T *U_raw,
                          const T *__restrict__ cos_phi_mat_raw,
                          const T *__restrict__ sin_phi_mat_raw) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y;

  for (int b = bid; b < BS; b += gridDim.y) {
    T *U = U_raw + b * N * N;
    const T *cos_phi_mat = cos_phi_mat_raw + b * N * N;
    const T *sin_phi_mat = sin_phi_mat_raw + b * N * N;
    if (tid < N) {
      T *row_p = U;
      for (int i = 0; i < N; ++i) {
        T *row_q = U + (N - 1) * N;
        for (int j = 0; j < N - i - 1; ++j) {
          T c = cos_phi_mat[i * N + j];
          T s = sin_phi_mat[i * N + j];

          // int q = N - j - 1;
          // T *row_q = U + q * N;

          T new_row_p = row_p[tid] * c - row_q[tid] * s;
          T new_row_q = row_p[tid] * s + row_q[tid] * c;
          row_p[tid] = new_row_p;
          row_q[tid] = new_row_q;
          row_q -= N;
        }
        row_p += N;
      }
    }
  }
}

template <typename T>
__global__ void negateRow(const int BS, const int N, T *U_raw,
                          const T *__restrict__ delta_list, const int row_id) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y;
  T *row = U_raw + bid * N * N + row_id * N;
  for (int i = tid; i < N; i += blockDim.x) {
    row[i] *= delta_list[bid];
  }
}

template <typename T>
__global__ void
unitaryReconstructFrancisShared(const int BS, const int N, T *U_raw,
                                const T *__restrict__ cos_phi_mat_raw,
                                const T *__restrict__ sin_phi_mat_raw) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y;

  extern __shared__ unsigned char *phi_row_data[];
  T *cos_phi_row = (T *)phi_row_data;
  T *sin_phi_row = ((T *)phi_row_data) + N;

  for (int b = bid; b < BS; b += gridDim.y) {
    T *U = U_raw + b * N * N;
    const T *cos_phi_mat = cos_phi_mat_raw + b * N * N;
    const T *sin_phi_mat = sin_phi_mat_raw + b * N * N;
    T *row_p = U;
    for (int i = 0; i < N; ++i) {
      T *row_q = U + (N - 1) * N;
      // preload
      if (tid < N - i - 1) {
        cos_phi_row[tid] = cos_phi_mat[i * N + tid];
        sin_phi_row[tid] = sin_phi_mat[i * N + tid];
      }
      __syncthreads();
      if (tid < N) {
        for (int j = 0; j < N - i - 1; ++j) {
          T c = cos_phi_row[j];
          T s = sin_phi_row[j];
          // int q = N - j - 1;
          // T *row_q = U + q * N;
          T new_row_p = row_p[tid] * c - row_q[tid] * s;
          T new_row_q = row_p[tid] * s + row_q[tid] * c;
          row_p[tid] = new_row_p;
          row_q[tid] = new_row_q;
          row_q -= N;
        }
        row_p += N;
      }
    }
  }
}

template <typename T>
__global__ void unitaryReconstructClementsShared(
    const int BS, const int N, T *U_raw, const T *__restrict__ cos_phi_mat_raw,
    const T *__restrict__ sin_phi_mat_raw, const int col) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int rid = blockIdx.y;
  const int bid = blockIdx.z;

  __shared__ T phi_data[2];
  T *cos_phi = phi_data;
  T *sin_phi = phi_data + 1;

  for (int b = bid; b < BS; b += gridDim.z) {
    T *U = U_raw + b * N * N;
    const T *cos_phi_mat = cos_phi_mat_raw + b * N * N;
    const T *sin_phi_mat = sin_phi_mat_raw + b * N * N;
    for (int r = 2 * rid + (col & 1); r < N - 1; r += 2 * gridDim.y) {
      U += N * r;
      // preload cos and sin
      if (tid == 0) {
        *cos_phi = cos_phi_mat[r * N + col];
        *sin_phi = sin_phi_mat[r * N + col];
      }
      __syncthreads();
      if (tid < N) {
        T c = *cos_phi;
        T s = *sin_phi;
        T *row_p = U;
        T *row_q = U + N;
        T new_row_p = row_p[tid] * c - row_q[tid] * s;
        T new_row_q = row_p[tid] * s + row_q[tid] * c;
        row_p[tid] = new_row_p;
        row_q[tid] = new_row_q;
      }
    }
  }
}

template <typename T>
__global__ void unitaryReconstructClementsSmallShared(
    const int BS, const int N, T *U_raw, const T *__restrict__ cos_phi_mat_raw,
    const T *__restrict__ sin_phi_mat_raw, const T *delta_list_0,
    const T *delta_list_N) {
  // a 2D CTA cover the entire matrix rotation, can force sync after each
  // column.
  const int tid = threadIdx.x;
  const int rid = threadIdx.y;
  const int bid = blockIdx.x;

  extern __shared__ unsigned char *phi_col_data[];
  T *cos_phi_col = (T *)phi_col_data;
  T *sin_phi_col = (T *)(phi_col_data) + (N >> 1); // compact phase vector

  // Jiaqi: GridDim.x cannot be too large, otherwise shared memory is not
  // enough. Thus this strided loop is a must.
  for (int b = bid; b < BS; b += gridDim.x) {
    T *U0 = U_raw + b * N * N;
    const T *cos_phi_mat = cos_phi_mat_raw + b * N * N;
    const T *sin_phi_mat = sin_phi_mat_raw + b * N * N;
    int r_even = 2 * rid;
    const int offset = 1 & (~N);
    // handle the delta_list[N-1] before the first column when N odd
    if (rid == 0 && tid < N && (N & 1 == 1)) {
      U0[tid + N * (N - 1)] *= delta_list_N[b];
    }
    __syncthreads();
    for (int col = 0; col < N - 1;
         ++col) { // handle layer loop inside the kernel since CTA can guarantee
                  // sync

      // preload a column of cos and sin using the x dimension
      if (tid < (N >> 1) - (col & offset)) {
        const int row = (tid << 1) + (col & 1);
        cos_phi_col[tid] = cos_phi_mat[row * N + col];
        sin_phi_col[tid] = sin_phi_mat[row * N + col];
      }
      __syncthreads();
      int r = r_even + (col & 1);
      if (r < N - 1) {
        T *U = U0 + r * N;
        if (tid < N) {
          T c = cos_phi_col[rid];
          T s = sin_phi_col[rid];
          T *row_p = U;
          T *row_q = U + N;
          T new_row_p = row_p[tid] * c - row_q[tid] * s;
          T new_row_q = row_p[tid] * s + row_q[tid] * c;
          row_p[tid] = new_row_p;
          row_q[tid] = new_row_q;
        }
      }
      __syncthreads();
      // handle the delta_list[N-1] after the first column when N even
      if (rid == 0 && tid < N && (N & 1) == 0) {
        U0[tid + N * (N - 1)] *= delta_list_N[b];
      }
      __syncthreads();
    }
    // handle the diagonal before the last column when N odd
    if (rid == 0 && tid < N && (N & 1) == 1) {
      U0[tid] *= delta_list_0[b];
    }
    __syncthreads();
    /// finish the last column
    // preload a column of cos and sin using the x dimension
    int col = N - 1;
    if (tid < (N >> 1) - (col & offset)) {
      const int row = (tid << 1) + (col & 1);
      cos_phi_col[tid] = cos_phi_mat[row * N + col];
      sin_phi_col[tid] = sin_phi_mat[row * N + col];
    }
    __syncthreads();
    int r = r_even + (col & 1);
    if (r < N - 1) {
      T *U = U0 + N * r;
      if (tid < N) {
        T c = cos_phi_col[rid];
        T s = sin_phi_col[rid];
        T *row_p = U;
        T *row_q = U + N;
        T new_row_p = row_p[tid] * c - row_q[tid] * s;
        T new_row_q = row_p[tid] * s + row_q[tid] * c;
        row_p[tid] = new_row_p;
        row_q[tid] = new_row_q;
      }
    }
    // handle the diagonal after the last column when N even
    if (rid == 0 && tid < N && (N & 1) == 0) {
      U0[tid] *= delta_list_0[b];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void unitaryReconstructClements(
    const int BS, const int N, T *U_raw, const T *__restrict__ cos_phi_mat_raw,
    const T *__restrict__ sin_phi_mat_raw, const int col) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int rid = blockIdx.y;
  const int bid = blockIdx.z;
  const int max_len = 2 * (col + 1);

  for (int b = bid; b < BS; b += gridDim.z) {
    T *U = U_raw + b * N * N;
    const T *cos_phi_mat = cos_phi_mat_raw + b * N * N;
    const T *sin_phi_mat = sin_phi_mat_raw + b * N * N;
    for (int r = 2 * rid + (col & 1); r < N - 1; r += 2 * gridDim.y) {
      U += N * r;

      // if(tid < N)

      int lower;
      int upper;
      lower = r - col;
      upper = lower + max_len;
      lower = max(0, lower);
      upper = min(upper, N);
      if (lower <= tid && tid < upper) {
        T *row_p = U;
        T *row_q = U + N;
        T c = cos_phi_mat[r * N + col];
        T s = sin_phi_mat[r * N + col];
        T new_row_p = row_p[tid] * c - row_q[tid] * s;
        T new_row_q = row_p[tid] * s + row_q[tid] * c;
        row_p[tid] = new_row_p;
        row_q[tid] = new_row_q;
      }
    }
  }
}

template <typename T>
__global__ void
unitaryReconstructReckShared(const int BS, const int N, T *U_raw,
                             const T *__restrict__ cos_phi_mat_raw,
                             const T *__restrict__ sin_phi_mat_raw) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y;

  extern __shared__ unsigned char *phi_row_data[];
  T *cos_phi_row = (T *)phi_row_data;
  T *sin_phi_row = ((T *)phi_row_data) + N;

  for (int b = bid; b < BS; b += gridDim.y) {
    T *U = U_raw + b * N * N;
    const T *cos_phi_mat = cos_phi_mat_raw + b * N * N;
    const T *sin_phi_mat = sin_phi_mat_raw + b * N * N;
    for (int i = 0; i < N - 1; ++i) {
      // preload phase row
      if (tid < i + 1) {
        int p1 = N - 2 - i;
        cos_phi_row[tid] = cos_phi_mat[p1 * N + tid];
        sin_phi_row[tid] = sin_phi_mat[p1 * N + tid];
      }
      __syncthreads();
      if (tid < N) {
        for (int j = 0; j < i + 1; ++j) {
          T c = cos_phi_row[j];
          T s = sin_phi_row[j];
          T *row_p = U + (N - 2 - i + j) * N;
          T *row_q = row_p + N;

          T new_row_p = row_p[tid] * c - row_q[tid] * s;
          T new_row_q = row_p[tid] * s + row_q[tid] * c;
          row_p[tid] = new_row_p;
          row_q[tid] = new_row_q;
        }
      }
    }
  }
}

template <typename T>
__global__ void unitaryReconstructReck(const int BS, const int N, T *U_raw,
                                       const T *__restrict__ cos_phi_mat_raw,
                                       const T *__restrict__ sin_phi_mat_raw) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bid = blockIdx.y;

  for (int b = bid; b < BS; b += gridDim.y) {
    T *U = U_raw + b * N * N;
    const T *cos_phi_mat = cos_phi_mat_raw + b * N * N;
    const T *sin_phi_mat = sin_phi_mat_raw + b * N * N;

    if (tid < N) {
      for (int i = 0; i < N - 1; ++i) {
        int lower = N - 2 - i;
        if (tid >= lower) {
          T *row_p = U + lower * N;
          T *row_q = row_p + N;
          for (int j = 0; j < i + 1; ++j) {
            T c = cos_phi_mat[lower * N + j];
            T s = sin_phi_mat[lower * N + j];

            T new_row_p = row_p[tid] * c - row_q[tid] * s;
            T new_row_q = row_p[tid] * s + row_q[tid] * c;
            row_p[tid] = new_row_p;
            row_q[tid] = new_row_q;
            row_p += N;
            row_q += N;
          }
        }
      }
    }
  }
}

template <typename T>
void unitaryReconstructFrancisCUDALauncher(const int BS, const int N, T *U,
                                           const T *cos_phi_mat,
                                           const T *sin_phi_mat) {
  int num_thread = min(((N + 32 - 1) / 32) * 32, 256); // 512;
  dim3 gridSize((N + num_thread - 1) / num_thread, BS, 1);
  if (sizeof(T) == 8)
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  if (N <= 80 / BS)
    unitaryReconstructFrancisShared<T>
        <<<gridSize, num_thread, 2 * N * sizeof(T)>>>(BS, N, U, cos_phi_mat,
                                                      sin_phi_mat);
  else
    unitaryReconstructFrancis<T>
        <<<gridSize, num_thread>>>(BS, N, U, cos_phi_mat, sin_phi_mat);
}

template <typename T>
void unitaryReconstructClementsCUDALauncher(const int BS, const int N, T *U,
                                            const T *cos_phi_mat,
                                            const T *sin_phi_mat,
                                            const T *delta_list_0,
                                            const T *delta_list_N) {
  // avoid shared memory bank conflict when reading doubles
  if (sizeof(T) == 8)
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  if (N <= 32) // a small matrix can be coverd by a 2D CTA
  {
    dim3 blockSize(32, N / 2, 1);
    dim3 gridSize(BS, 1, 1);
    // dim3 gridSize2((N + num_thread - 1) / num_thread, BS, 1);
    // negateRow<T><<<gridSize2, num_thread>>>(BS, N, U, delta_list_N, N-1);
    // cudaDeviceSynchronize();
    // unitaryReconstructClementsShared<T><<<gridSize, num_thread>>>(BS, N, U,
    // cos_phi_mat, sin_phi_mat, i);
    unitaryReconstructClementsSmallShared<T>
        <<<gridSize, blockSize, 2 * (N >> 1) * sizeof(T)>>>(
            BS, N, U, cos_phi_mat, sin_phi_mat, delta_list_0, delta_list_N);
  } else {
    /// each layer launch one kernel to force device-level synchronization
    int num_thread = min(int((N + 32 - 1) / 32) * 32, 256);
    dim3 gridSize((N + num_thread - 1) / num_thread, int(N / 2), BS);
    // consider diagonal[N-1] before the first layer when N odd
    dim3 gridSize2((N + num_thread - 1) / num_thread, BS, 1);
    if ((N & 1) == 1) {
      negateRow<T><<<gridSize2, num_thread>>>(BS, N, U, delta_list_N, N - 1);
      cudaDeviceSynchronize();
    }

    for (int i = 0; i < N - 1; ++i) {
      unitaryReconstructClements<T>
          <<<gridSize, num_thread>>>(BS, N, U, cos_phi_mat, sin_phi_mat, i);
      cudaDeviceSynchronize();
      // consider diagonal[N-1] after the first layer when N even
      if ((N & 1) == 0) {
        negateRow<T><<<gridSize2, num_thread>>>(BS, N, U, delta_list_N, N - 1);
        cudaDeviceSynchronize();
      }
    }
    // consider diagonal[0] before the last layer when N odd
    if ((N & 1) == 1) {
      negateRow<T><<<gridSize2, num_thread>>>(BS, N, U, delta_list_0, 0);
      cudaDeviceSynchronize();
    }

    unitaryReconstructClements<T>
        <<<gridSize, num_thread>>>(BS, N, U, cos_phi_mat, sin_phi_mat, N - 1);
    cudaDeviceSynchronize();
    // consider diagonal[0] after the last layer when N even
    if ((N & 1) == 0) {
      negateRow<T><<<gridSize2, num_thread>>>(BS, N, U, delta_list_0, 0);
      cudaDeviceSynchronize();
    }
  }
}

template <typename T>
void unitaryReconstructReckCUDALauncher(const int BS, const int N, T *U,
                                        const T *cos_phi_mat,
                                        const T *sin_phi_mat) {
  int num_thread = min(((N + 32 - 1) / 32) * 32, 256); // 512;
  dim3 gridSize((N + num_thread - 1) / num_thread, BS, 1);
  if (sizeof(T) == 8)
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  if (N <= 80 / BS)
    unitaryReconstructReckShared<T>
        <<<gridSize, num_thread, 2 * N * sizeof(T)>>>(BS, N, U, cos_phi_mat,
                                                      sin_phi_mat);
  else
    unitaryReconstructReck<T>
        <<<gridSize, num_thread>>>(BS, N, U, cos_phi_mat, sin_phi_mat);
}

template <typename T>
__forceinline__ __device__ T calPhi(const T u1, const T u2) {
  T u1_abs = abs(u1);
  T u2_abs = abs(u2);
  T phi = 0;
  int cond = ((u1_abs >= ERROR) << 1) | (u2_abs >= ERROR);
  // switch(cond)
  // {
  //     case 0: phi = 0; break;
  //     case 1: phi = u2 > ERROR ? -0.5 * PI : 0.5 * PI; break;
  //     case 2: phi = u1 > ERROR ? 0 : -PI; break;
  //     case 3: phi = std::atan2(-u2, u1); break;
  //     default: break;
  // }
  switch (cond) {
  case 0:
    return 0;
    break;
  case 1:
    return u2 > ERROR ? -0.5 * PI : 0.5 * PI;
    break;
  case 2:
    return u1 > ERROR ? 0 : -PI;
    break;
  case 3:
    return std::atan2(-u2, u1);
    break;
  default:
    break;
  }
  return phi;
}

template <typename T>
__global__ void unitaryDecomposeFrancis(const int BS, const int N, T *U_raw,
                                        T *delta_list_raw, T *phi_mat_raw) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // use block dim x to handle batch
  __shared__ T cos_sin_phi[2];
  for (int b = bid; b < BS; b += gridDim.x) {
    T *U = U_raw + b * N * N;
    T *delta_list = delta_list_raw + b * N;
    T *phi_mat = phi_mat_raw + b * N * N;
    for (int i = 0; i < N - 1; ++i) {
      // decomposeKernel
      for (int j = i; j < N - 1; ++j) {
        int p = i;
        int q = N - 1 - j + i;
        T c = 0;
        T s = 0;
        T phi = 0;
        if (tid == 0) {
          T u1 = U[p * N + p];
          T u2 = U[p * N + q];
          phi = calPhi(u1, u2);
          c = cos(phi);
          cos_sin_phi[0] = c;
          cos_sin_phi[1] = phi <= 0 ? -sqrt(1 - c * c) : sqrt(1 - c * c);
        }
        __syncthreads();
        if (tid == 0) {
          phi_mat[i * N + j - i] = phi;
        }
        for (int k = tid + i; k < N; k += blockDim.x) {
          T col_p = U[k * N + p];
          T col_q = U[k * N + q];
          c = cos_sin_phi[0];
          s = cos_sin_phi[1];
          T col_p_cos = col_p * c;
          T col_p_sin = col_p * s;
          T col_q_cos = col_q * c;
          T col_q_sin = col_q * s;
          U[k * N + p] = col_p_cos - col_q_sin;
          U[k * N + q] = col_p_sin + col_q_cos;
        }
        __syncthreads();
        // if(tid >= i && tid < N)
        // {
        //     T col_p = U[tid * N + p];
        //     T col_q = U[tid * N + q];
        //     c = cos_sin_phi[0];
        //     s = cos_sin_phi[1];
        //     T col_p_cos = col_p * c;
        //     T col_p_sin = col_p * s;
        //     T col_q_cos = col_q * c;
        //     T col_q_sin = col_q * s;
        //     U[tid * N + p] = col_p_cos - col_q_sin;;
        //     U[tid * N + q] = col_p_sin + col_q_cos;
        // }
      }
    }
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) {
      delta_list[i] = U[i * N + i];
    }
    __syncthreads();
  }
}

template <typename T>
void unitaryDecomposeFrancisCUDALauncher(const int BS, const int N, T *U,
                                         T *delta_list, T *phi_mat) {
  int num_thread = 512;
  int num_block = BS;
  // need synchronization per loop. only one block is used for each unitary
  unitaryDecomposeFrancis<T>
      <<<num_block, num_thread>>>(BS, N, U, delta_list, phi_mat);
}

template <typename T>
__global__ void unitaryDecomposeClements(const int BS, const int N, T *U_raw,
                                         T *delta_list_raw, T *phi_mat_raw) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // use block dim x to handle batch
  __shared__ T cos_sin_phi[2];
  for (int b = bid; b < BS; b += gridDim.x) {
    T *U = U_raw + b * N * N;
    T *delta_list = delta_list_raw + b * N;
    T *phi_mat = phi_mat_raw + b * N * N;

    for (int i = 0; i < N - 1; ++i) {
      /// each outer loop deals with one off-diagonal
      // even loop for column rotation
      if ((i & 1) == 0) {
        for (int j = 0; j < i + 1; ++j) {
          int p = N - 1 - j;
          int q = i - j;
          T c = 0;
          T s = 0;
          T phi = 0;
          if (tid == 0) {
            T u1 = U[p * N + q + 1];
            T u2 = U[p * N + q];
            phi = -calPhi(u1, u2);
            c = cos(phi);
            cos_sin_phi[0] = c;
            cos_sin_phi[1] = phi <= 0 ? -sqrt(1 - c * c) : sqrt(1 - c * c);
          }
          __syncthreads();
          if (tid == 0) {
            phi_mat[(i - j) * N + j] = phi;
          }

          for (int k = tid; k < p + 1; k += blockDim.x) {
            T col_q_p1 = U[k * N + q + 1];
            T col_q = U[k * N + q];
            c = cos_sin_phi[0];
            s = cos_sin_phi[1];
            T col_q_p1_cos = col_q_p1 * c;
            T col_q_p1_sin = col_q_p1 * s;
            T col_q_cos = col_q * c;
            T col_q_sin = col_q * s;
            U[k * N + q + 1] = col_q_p1_cos + col_q_sin;
            U[k * N + q] = col_q_cos - col_q_p1_sin;
          }
          __syncthreads();
        }
      } else {
        // odd loop for row rotation
        for (int j = 0; j < i + 1; ++j) {
          int p = N - 1 - i + j;
          int q = j;
          T c = 0;
          T s = 0;
          T phi = 0;
          if (tid == 0) {
            T u1 = U[(p - 1) * N + q];
            T u2 = U[p * N + q];
            phi = calPhi(u1, u2);
            c = cos(phi);
            cos_sin_phi[0] = c;
            cos_sin_phi[1] = phi <= 0 ? -sqrt(1 - c * c) : sqrt(1 - c * c);
          }
          __syncthreads();
          if (tid == 0) {
            phi_mat[(N + j - i - 1) * N - 1 - j] = -phi;
          }
          for (int k = tid + j; k < N; k += blockDim.x) {
            T row_p_1 = U[p * N + k - N];
            T row_p = U[p * N + k];
            c = cos_sin_phi[0];
            s = cos_sin_phi[1];
            T row_p_1_cos = row_p_1 * c;
            T row_p_1_sin = row_p_1 * s;
            T row_p_cos = row_p * c;
            T row_p_sin = row_p * s;
            U[p * N + k - N] = row_p_1_cos - row_p_sin;
            U[p * N + k] = row_p_cos + row_p_1_sin;
          }
          __syncthreads();
        }
      }
    }
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) {
      delta_list[i] = U[i * N + i];
    }
    __syncthreads();
  }
}

template <typename T>
void unitaryDecomposeClementsCUDALauncher(const int BS, const int N, T *U,
                                          T *delta_list, T *phi_mat) {
  int num_thread = 512;
  int num_block = BS;
  // need synchronization per loop. only one block is used for each unitary
  unitaryDecomposeClements<T>
      <<<num_block, num_thread>>>(BS, N, U, delta_list, phi_mat);
}

template <typename T>
__global__ void unitaryDecomposeReck(const int BS, const int N, T *U_raw,
                                     T *delta_list_raw, T *phi_mat_raw) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // use block dim x to handle batch
  __shared__ T cos_sin_phi[2];
  for (int b = bid; b < BS; b += gridDim.x) {
    T *U = U_raw + b * N * N;
    T *delta_list = delta_list_raw + b * N;
    T *phi_mat = phi_mat_raw + b * N * N;

    for (int i = 0; i < N - 1; ++i) {
      /// each outer loop deals with one off-diagonal
      // even loop for column rotation

      for (int j = 0; j < i + 1; ++j) {
        int p = j;
        int q = N - 1 - i + j;
        T c = 0;
        T s = 0;
        T phi = 0;
        if (tid == 0) {
          T u1 = U[p * N + q - 1];
          T u2 = U[p * N + q];
          phi = calPhi(u1, u2);
          c = cos(phi);
          cos_sin_phi[0] = c;
          cos_sin_phi[1] = phi <= 0 ? -sqrt(1 - c * c) : sqrt(1 - c * c);
        }
        __syncthreads();
        if (tid == 0) {
          phi_mat[(N - i - 2) * N + j] = phi;
        }

        for (int k = tid; k < N - p; k += blockDim.x) {
          T col_q_m1 = U[(p + k) * N + q - 1];
          T col_q = U[(p + k) * N + q];
          c = cos_sin_phi[0];
          s = cos_sin_phi[1];
          T col_q_m1_cos = col_q_m1 * c;
          T col_q_m1_sin = col_q_m1 * s;
          T col_q_cos = col_q * c;
          T col_q_sin = col_q * s;
          U[(p + k) * N + q - 1] = col_q_m1_cos - col_q_sin;
          U[(p + k) * N + q] = col_q_cos + col_q_m1_sin;
        }
        __syncthreads();
      }
    }
    __syncthreads();
    for (int i = tid; i < N; i += blockDim.x) {
      delta_list[i] = U[i * N + i];
    }
    __syncthreads();
  }
}

template <typename T>
void unitaryDecomposeReckCUDALauncher(const int BS, const int N, T *U,
                                      T *delta_list, T *phi_mat) {
  int num_thread = 512;
  int num_block = BS;
  // need synchronization per loop. only one block is used for each unitary
  unitaryDecomposeReck<T>
      <<<num_block, num_thread>>>(BS, N, U, delta_list, phi_mat);
}

template void unitaryReconstructFrancisCUDALauncher(const int BS, const int N,
                                                    float *U,
                                                    const float *cos_phi_mat,
                                                    const float *sin_phi_mat);
template void unitaryReconstructFrancisCUDALauncher(const int BS, const int N,
                                                    double *U,
                                                    const double *cos_phi_mat,
                                                    const double *sin_phi_mat);

template void unitaryReconstructClementsCUDALauncher(const int BS, const int N,
                                                     float *U,
                                                     const float *cos_phi_mat,
                                                     const float *sin_phi_mat,
                                                     const float *delta_list_0,
                                                     const float *delta_list_N);
template void unitaryReconstructClementsCUDALauncher(
    const int BS, const int N, double *U, const double *cos_phi_mat,
    const double *sin_phi_mat, const double *delta_list_0,
    const double *delta_list_N);

template void unitaryReconstructReckCUDALauncher(const int BS, const int N,
                                                 float *U,
                                                 const float *cos_phi_mat,
                                                 const float *sin_phi_mat);
template void unitaryReconstructReckCUDALauncher(const int BS, const int N,
                                                 double *U,
                                                 const double *cos_phi_mat,
                                                 const double *sin_phi_mat);

template void unitaryDecomposeFrancisCUDALauncher(const int BS, const int N,
                                                  float *U, float *delta_list,
                                                  float *phi_mat);
template void unitaryDecomposeFrancisCUDALauncher(const int BS, const int N,
                                                  double *U, double *delta_list,
                                                  double *phi_mat);

template void unitaryDecomposeClementsCUDALauncher(const int BS, const int N,
                                                   float *U, float *delta_list,
                                                   float *phi_mat);
template void unitaryDecomposeClementsCUDALauncher(const int BS, const int N,
                                                   double *U,
                                                   double *delta_list,
                                                   double *phi_mat);

template void unitaryDecomposeReckCUDALauncher(const int BS, const int N,
                                               float *U, float *delta_list,
                                               float *phi_mat);
template void unitaryDecomposeReckCUDALauncher(const int BS, const int N,
                                               double *U, double *delta_list,
                                               double *phi_mat);

#if 0
#define REGISTER_KERNEL_LAUNCHER(T)                                            \
  void instantiateunitaryRotationCUDALauncher(                                 \
      const int N, const int num_block, const int num_thread, T *row_p,        \
      T *row_q, const T *cos_phi, const T *sin_phi) {                          \
    unitaryRotationCUDALauncher(N, num_block, num_thread, row_p, row_q,        \
                                cos_phi, sin_phi);                             \
  }                                                                            \
                                                                               \
  void instantiateunitaryReconstructFrancisCUDALauncher(                       \
      const int N, T *U, const T *cos_phi_mat, const T *sin_phi_mat) {         \
    unitaryReconstructFrancisCUDALauncher(N, U, cos_phi_mat, sin_phi_mat);     \
  }



REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);
#endif
