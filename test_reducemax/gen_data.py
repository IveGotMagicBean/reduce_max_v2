#!/usr/bin/python3
import numpy as np
import os

np.random.seed(123)

# 生成简单的测试数据
input_x = np.random.uniform(-10, 10, [32]).astype(np.float16)
reduce_dim = 0  # 对第0维做reduce

# 计算golden（使用numpy）
golden_max = np.max(input_x, axis=reduce_dim).astype(np.float16)
golden_idx = np.argmax(input_x, axis=reduce_dim).astype(np.int32)

os.makedirs("./input", exist_ok=True)
os.makedirs("./output", exist_ok=True)

input_x.tofile("./input/input_x.bin")
golden_max.tofile("./output/golden_y.bin")
golden_idx.tofile("./output/golden_idx.bin")

print(f"Generated test data:")
print(f"  input shape: {input_x.shape}")
print(f"  max value: {golden_max}")
print(f"  max index: {golden_idx}")
