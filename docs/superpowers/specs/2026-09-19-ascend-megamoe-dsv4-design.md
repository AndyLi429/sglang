# DeepSeek V4 NPU MegaMOE 接入设计

## 目标与范围

在 SGLang 的 Ascend NPU MoE 链路中接入 `cann_ops_transformer.ops.mega_moe`，使 DeepSeek V4 系列的 routed-expert 前向可由 MegaMOE 一次完成 EP dispatch、两次 expert GEMM 和 combine。

本变更从 `origin/main` 的 `83e29d6c5ae` 开始，新增独立的 `ascend_megamoe` A2A 后端。它与现有 `ascend_fuseep`、普通 DeepEP 路径并存，不修改 `sgl-kernel-npu` 或 DeepEP 的公共 API。

第一阶段的必须支持目标是 DeepSeek V4 W8A8，在满足算子约束且显式启用时使用 MegaMOE；不支持的硬件、量化方式、模型形状、并行配置和运行时容量必须安全回退到现有 MoE 路径。W4A8/MXFP4 只在已安装的 `cann_ops_transformer` API 与本地 NPU 权重格式通过专门测试后开启，不以猜测的 layout 启用。

不在本次范围内：将 MegaMOE 合并进 DeepEP、改造 DeepEP 内核、为 LoRA/MTP/共享专家 overlap 增加 MegaMOE 支持，或承诺未实测 SoC/CANN 组合的性能数值。

## 用户接口

用户通过现有 A2A 后端选择显式请求：

```bash
--moe-a2a-backend ascend_megamoe
```

新增环境变量：

- `SGLANG_NPU_ENABLE_MEGAMOE=1`：允许该后端尝试加载和执行 MegaMOE；默认 `0`。
- `SGLANG_NPU_MEGAMOE_MAX_RECV_TOKENS=<int>`：Prefill/PD 的每 rank 接收 token 容量；默认由全局、rank-invariant 调度容量推导。
- `SGLANG_NPU_MEGAMOE_STRICT=1`：不满足能力条件时抛出清晰异常；默认 `0`，记录一次 warning 后回退。

后端选择和环境变量都必须在模型加载、权重后处理之前解析。这样不会在已转换权重后临时切换 MegaMOE，避免 layout 不匹配。

## 架构与数据流

```text
server args/env
  -> MoeA2ABackend.ASCEND_MEGAMOE
  -> FusedMoE 初始化：noop dispatcher + MegaMOE 专用权重后处理
  -> FusedMoE.forward：能力检查
  -> hardware_backend/npu/moe/megamoe.py
  -> get_symm_buffer_for_mega_moe(EP group, rank-invariant capacity)
  -> cann_ops_transformer.ops.mega_moe(...)
  -> 输出 routed-expert 结果，保留模型既有 shared-expert/add 流程
```

`FusedMoE.forward` 采用与 `ascend_fuseep` 相同的直通方式，而不是使用普通 token dispatcher。MegaMOE 已经拥有 dispatch 和 combine；在其外层再调用 DeepEP 会重复通信，且会产生两个互不协调的通信 buffer。

## 组件设计

### 后端与能力检查

在 `MoeA2ABackend`、CLI 参数允许列表、NPU 并行校验中新增 `ascend_megamoe`。实现 `is_ascend_megamoe()`，并在 DeepSeek V2/V4 的 EP/TP 配置判定中视为 EP 后端。

`is_ascend_megamoe_available(layer)` 必须检查：NPU 设备、环境开关、`cann_ops_transformer.ops` 可导入、EP size 大于 1、总 expert 数能被 EP size 整除、无 LoRA、当前 token 容量合法、模型 activation 为 MegaMOE 支持的 SwiGLU 语义，以及当前量化方法已提供兼容权重。失败时严格模式报错；非严格模式只回退到原 dispatcher，不能半初始化 MegaMOE buffer。

### 对称 buffer

`megamoe.py` 缓存每个进程的 buffer，key 包含 EP group、hidden、intermediate、expert 数、top-k、量化模式和 rank-invariant capacity。

buffer 初始化只能使用调度器配置、图 capture size 或所有 rank 一致的上限，绝不使用某个 rank 当前 forward 的 token 数。`max_recv_token_num` 必须不超过算子的安全上限；运行时 token 数超过 buffer capacity 时，严格模式失败，非严格模式走普通 MoE。

### 权重与量化

MegaMOE 专用权重转换放在 NPU MoE quant method 的 `process_weights_after_loading` 分支中，且只在 `ascend_megamoe` 被选中时执行。W8A8 保存逐 expert、未被普通 grouped-GMM 转置破坏的 `w13/w2` 和 scale 列表；调用时传入 `int32` top-k ids、`float32` top-k weights 与 `activation_clamp=layer.moe_runner_config.swiglu_limit`。

W4A8/MXFP4 的分支必须只在对应的 CANN MegaMOE weight dtype、packing 和 NPU format 已被单元/真机测试确认后添加。未确认时能力检查返回不支持，禁止复用 FuseEP 或普通 GMM 的转换结果。

### 回退与兼容性

非严格回退发生在 MoE forward 入口，且权重仍保留普通 runner 可消费的表示。为此，MegaMOE 权重缓存是附加缓存，不能删除或就地替换普通权重。

首次实现不支持 LoRA、MTP、`shared_expert` multistream overlap 和不兼容 ACLGraph 捕获；这些条件统一进入能力检查并回退。DeepSeek V4 shared experts 仍按现有模型逻辑独立计算，MegaMOE 只替换 routed experts。

## 修改范围

- `python/sglang/srt/environ.py`：MegaMOE 环境变量。
- `python/sglang/srt/layers/moe/utils.py`、参数定义/校验：后端枚举与 CLI 可选值。
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`：MegaMOE 直通和 dispatcher 选择。
- `python/sglang/srt/hardware_backend/npu/moe/megamoe.py`：算子加载、能力检查、buffer 管理与 forward。
- `python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py`：W8A8 专用权重缓存；后续按验证结果扩展其他 quant method。
- 受影响模型的 EP backend 判断：仅添加 `ascend_megamoe` 等价分支。
- `test/registered/unit/npu/`：mock 算子测试能力门控、buffer 参数、dtype/layout payload、回退与 DeepSeek V4 clamp 透传。

## 验收标准

1. 未安装 `cann_ops_transformer` 或环境变量关闭时，默认路径行为不变。
2. 显式选择且环境变量开启时，DeepSeek V4 W8A8 的 routed experts 调用 `mega_moe`，输入 dtype/shape 与 API 合约一致，并传递 `swiglu_limit`。
3. 所有 EP rank 使用相同的对称 buffer 形状参数；运行时容量溢出不会产生不一致 collective。
4. 不支持条件非严格回退、严格报错均有单元测试。
5. 通过相关 Python 单元测试；在匹配的 NPU/CANN/ops-transformer 环境上完成至少一组 W8A8 EP 真机精度和性能对照。CPU 测试不作为算子正确性证明。

## 验证顺序

1. Host-safe mock 单测：注册、环境门控、buffer 参数、payload、fallback。
2. NPU import/API probe：确认已安装 `cann_ops_transformer` 的真实函数签名与支持能力。
3. Dummy 模型启动：确认后端解析与模型加载。
4. DeepSeek V4 W8A8 EP 真机：与普通 MoE 输出比较精度。
5. 固定拓扑、固定 batch 和相同 CANN 版本的端到端性能基准。
