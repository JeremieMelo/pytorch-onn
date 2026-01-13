/*
 * @Author: Jiaqi Gu
 * @Date: 2020-06-04 14:16:01
 * @LastEditors: Jiaqi Gu (jqgu@utexas.edu)
 * @LastEditTime: 2021-11-29 03:01:43
 */
#include <torch/torch.h>

void fwtBatchGPU(float *x, int batchSize, int log2N);

at::Tensor hadamard_transform(at::Tensor x) {
  TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor");
  auto n = x.sizes().back();
  auto log2N = long(log2(n));
  TORCH_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = x.clone(); // Cloning makes it contiguous.
  auto batchSize = x.numel() / (1 << log2N);
  fwtBatchGPU(output.data<float>(), batchSize, log2N);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hadamard_transform", &hadamard_transform, "Fast Hadamard transform");
}
