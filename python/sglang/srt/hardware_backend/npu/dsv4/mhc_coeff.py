"""Load the private CANN 9.2 A3 coefficient-only OPP."""

import os
from functools import cache
from pathlib import Path


@cache
def load_operator():
    import sgl_kernel_npu
    import torch_npu
    from sgl_kernel_npu.operators.utils.cann import detect_cann_version
    from torch.utils.cpp_extension import load

    if (
        torch_npu._C._npu_get_soc_version() // 10 * 10 != 250
        or not detect_cann_version().startswith("9.2.")
    ):
        raise RuntimeError("MhcPreSinkhornCoeff requires Ascend A3 and CANN 9.2")
    toolkit = Path(os.environ["ASCEND_HOME_PATH"])
    npu = Path(torch_npu.__file__).parent
    common = (
        Path(sgl_kernel_npu.__file__).parent
        / "_third_party/sgl_ops_transformer_dsv4/common"
    )
    return load(
        name="sglang_mhc_coeff_a3",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        extra_include_paths=[
            str(common),
            str(toolkit / "include"),
            str(npu / "include"),
            str(npu / "include/third_party/acl/inc"),
            str(npu / "include/third_party/hccl/inc"),
            str(npu / "include/third_party/op-plugin"),
        ],
        extra_cflags=["-O2", "-std=c++20"],
        build_directory=os.environ.get("SGLANG_NPU_MHC_COEFF_BUILD_DIR"),
        extra_ldflags=[
            f"-L{toolkit / 'lib64'}",
            "-lascendcl",
            f"-L{npu / 'lib'}",
            "-ltorch_npu",
        ],
    ).coefficients


def coefficients(x, phi, alpha, bias, streams, iters, hc_eps, norm_eps):
    return load_operator()(x, phi, alpha, bias, streams, iters, hc_eps, norm_eps)
