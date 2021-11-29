/*
 * @Author: Jiaqi Gu
 * @Date: 2020-06-04 14:17:00
 * @LastEditors: Jiaqi Gu (jqgu@utexas.edu)
 * @LastEditTime: 2021-11-29 03:02:00
 */
#include <torch/torch.h>

void uftBatchGPU(float* x, int batchSize, int log2N);
void iuftBatchGPU(float* x, int batchSize, int log2N);

at::Tensor universal_transform(at::Tensor x) {
  TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.size(-1) == 2, "x must be a complex-valued tensor");
  auto n = x.size(-2);
  auto log2N = long(log2(n));
  TORCH_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = x.clone();  // Cloning makes it contiguous.
  auto batchSize = x.numel() / n / 2.;
  uftBatchGPU(output.data<float>(), batchSize, log2N);
  return output;
}

at::Tensor inverse_universal_transform(at::Tensor x) {
  TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.size(-1) == 2, "x must be a complex-valued tensor");
  auto n = x.size(-2);
  auto log2N = long(log2(n));
  TORCH_CHECK(n == 1 << log2N, "n must be a power of 2");
  auto output = x.clone();  // Cloning makes it contiguous.
  auto batchSize = x.numel() / n / 2.;
  iuftBatchGPU(output.data<float>(), batchSize, log2N);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("universal_transform", &universal_transform, "Universal frequency transform");
  m.def("inverse_universal_transform", &inverse_universal_transform, "Inverse universal frequency transform");
}
