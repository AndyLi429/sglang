#include <torch/extension.h>

// The ACLNN helpers use the Torch C++ API declarations above.
#include "aclnn_common.h"

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
coefficients(const at::Tensor &x, const at::Tensor &phi,
             const at::Tensor &alpha, const at::Tensor &bias, int64_t hcMult,
             int64_t numIters, double hcEps, double normEps) {
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 4 && x.size(2) == 5120 &&
                  x.scalar_type() == at::kBFloat16 && hcMult == 4 &&
                  numIters == 20,
              "MhcPreSinkhornCoeff expects BF16 [M, 4, 5120], hc_mult=4 and 20 "
              "iterations");
  auto options = x.options().dtype(at::kFloat);
  const auto rows = x.size(0);
  auto pre = at::empty({rows, 1, 4}, options);
  auto post = at::empty({rows, 1, 4}, options);
  auto residual = at::empty({rows, 1, 16}, options);
  auto input = x.unsqueeze(1);
  ACLNN_CMD(aclnnMhcPreSinkhornCoeff, input, phi, alpha, bias, hcMult, numIters,
            hcEps, normEps, pre, post, residual);
  return {pre.squeeze(1), post.squeeze(1), residual.squeeze(1)};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("coefficients", &coefficients);
}
