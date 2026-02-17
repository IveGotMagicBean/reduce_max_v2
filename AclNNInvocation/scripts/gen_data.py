#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import tensorflow as tf
np.random.seed(123)

def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [32]).astype(np.float16)
    dim_num = len(input_x.shape)
    input_axis = np.random.uniform(-dim_num, dim_num, [1]).astype(np.int32)
    keepdims = False
    golden = tf.reduce_sum(input_x, axis=input_axis, keepdims=keepdims).numpy()
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_axis.tofile("./input/input_axis.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
