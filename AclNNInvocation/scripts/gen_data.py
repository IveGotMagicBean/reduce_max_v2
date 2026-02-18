#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import os

np.random.seed(123)

def gen_golden_data_simple():
    # 生成测试数据：shape=[32], dtype=float16
    input_x = np.random.uniform(-10, 10, [32]).astype(np.float16)
    input_axis = np.array([0], dtype=np.int32)  # 对axis=0做reduce
    
    # 用numpy计算golden（不需要tensorflow）
    golden = np.sum(input_x, axis=tuple(input_axis), keepdims=False).astype(np.float16)
    
    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    
    input_x.tofile("./input/input_x.bin")
    input_axis.tofile("./input/input_axis.bin")
    golden.tofile("./output/golden.bin")
    
    print(f"Generated test data:")
    print(f"  input_x shape: {input_x.shape}, dtype: {input_x.dtype}")
    print(f"  input_axis: {input_axis}")
    print(f"  golden: {golden} (shape: {golden.shape})")

if __name__ == "__main__":
    gen_golden_data_simple()
