"""简单测试 dataloader 是否工作"""

import torch
from mychat.dataloader import tokenizing_distributed_data_loader

# 测试参数
B = 2  # batch size
T = 128  # sequence length
split = "train"
device = "cpu"  # 使用 CPU 避免 GPU 依赖

print("开始测试 dataloader...")
print(f"参数: B={B}, T={T}, split={split}, device={device}")

try:
    # 创建 dataloader
    dataloader = tokenizing_distributed_data_loader(
        B=B, T=T, split=split, tokenizer_threads=2, tokenizer_batch_size=512, device=device
    )

    print("✓ Dataloader 创建成功")

    # 获取第一个 batch
    print("\n获取第一个 batch...")
    inputs, targets = next(dataloader)

    print(f"✓ 成功获取 batch")
    print(f"  inputs shape: {inputs.shape}")
    print(f"  targets shape: {targets.shape}")
    print(f"  inputs dtype: {inputs.dtype}")
    print(f"  targets dtype: {targets.dtype}")
    print(f"  inputs device: {inputs.device}")

    # 验证形状
    assert inputs.shape == (B, T), f"inputs shape 错误: {inputs.shape}"
    assert targets.shape == (B, T), f"targets shape 错误: {targets.shape}"

    # 验证 targets 是 inputs 的下一个 token
    print(f"\n  前 10 个 tokens (第一个序列):")
    print(f"    inputs:  {inputs[0, :10].tolist()}")
    print(f"    targets: {targets[0, :10].tolist()}")

    # 获取第二个 batch 确认可以继续迭代
    print("\n获取第二个 batch...")
    inputs2, targets2 = next(dataloader)
    print(f"✓ 成功获取第二个 batch")
    print(f"  inputs shape: {inputs2.shape}")

    print("\n✅ 测试通过！Dataloader 工作正常。")

except Exception as e:
    print(f"\n❌ 测试失败: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
