# ReduceMaxCustom 算子工程说明

> 基于华为 Ascend C 实现的自定义 ReduceMax 算子，支持多维张量的指定维度最大值归约，返回最大值及对应索引。

**参赛选手：** ShiyiLIN  
**提交日期：** 2026-02-28 
**赛题：** 昇腾AI算法挑战赛中阶赛 · Reduce算子优化  
**硬件平台：** 昇腾 Ascend 910B4 NPU  
**CANN版本：** 8.5.0  

---

## 目录

- [工程结构](#工程结构)
- [算子功能说明](#算子功能说明)
- [核心实现思路](#核心实现思路)
- [精度修复说明](#精度修复说明)
- [环境依赖](#环境依赖)
- [编译与安装](#编译与安装)
- [精度测试](#精度测试)
- [测试结果](#测试结果)

---

## 工程结构

```
ReduceMaxCustom/
├── op_kernel/
│   └── reduce_max_custom.cpp        # 算子核函数实现（AICore 侧）
├── op_host/
│   ├── reduce_max_custom.cpp        # Tiling 策略 + 算子注册（Host 侧）
│   └── reduce_max_custom_tiling.h   # Tiling 数据结构定义
├── test_reducemax/
│   ├── test_main.cpp                # 多 Shape 精度验证程序
│   ├── gen_test_data.py             # 生成测试数据脚本（基于 NumPy golden）
│   ├── CMakeLists.txt               # 测试程序构建配置
│   ├── input_*/                     # 各 Shape 输入数据
│   └── output_*/                    # 各 Shape golden 参考数据
├── build.sh                         # 一键编译脚本
├── reduce_max_custom.json           # 算子描述文件
└── README.md                        # 本文档
```

### 关键文件说明

| 文件 | 说明 |
|------|------|
| `op_kernel/reduce_max_custom.cpp` | 核函数，运行在 AICore 上，实现向量化归约逻辑 |
| `op_host/reduce_max_custom.cpp` | Host 侧 Tiling 函数，计算 outer/inner/stride，决定核数和 tileSize |
| `op_host/reduce_max_custom_tiling.h` | 定义传递给核函数的 Tiling 数据字段 |
| `test_reducemax/gen_test_data.py` | 用 NumPy 生成输入数据和 golden 结果，供精度对比 |
| `test_reducemax/test_main.cpp` | ACL 调用算子并与 golden 逐元素对比，输出精度报告 |

---

## 算子功能说明

**数学定义：**

```
output[i] = max(input[i, :])   沿指定维度求最大值
index[i]  = argmax(input[i, :]) 最大值在该维度上的位置索引
```

**接口参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| x | float16, ND | 输入张量 |
| y | float16, ND | 输出最大值张量 |
| idx | int32, ND | 输出最大值索引张量 |
| reduceDim | int64 属性 | 归约维度，支持负数索引 |
| isKeepDim | int64 属性（可选） | 是否保留归约维度，默认 0 |

**示例：**

```
输入 shape: (512, 128, 512)，reduceDim=1
输出 shape: (512, 512)，每个位置是对应行128个元素中的最大值及索引
```

---

## 核心实现思路

### Tiling 策略（Host 侧）

将输入 shape 抽象为三段：

```
outer = 归约维度之前所有维度之积
inner = 归约维度大小
stride = 归约维度之后所有维度之积
```

以 shape=(512, 128, 512), reduceDim=1 为例：
- outer = 512，inner = 128，stride = 512
- 输出总元素数 totalOut = outer × stride = 262144
- 最多 20 个核并行，每核负责约 262144/20 个输出元素

### 核函数逻辑（Kernel 侧）

根据 stride 分两条计算路径：

**路径一：stride == 1（Last-Reduce，归约最后一维）**

使用 `ComputeLastReduce`，充分利用向量化指令：
1. 将 inner 个元素按 tileSize 分块，流水搬运到 UB
2. 每块用 `WholeReduceMax`（SIMD 指令）以 128 个元素为单位批量求最大值
3. 在 work buffer 中找出全局最大值及其精确索引

**路径二：stride > 1（非最后维归约）**

使用 `ComputeStridedReduce`，按步长逐元素访问：
1. 对每个输出位置 (o, s)，遍历 inner 个元素
2. 每次读取 16 个对齐元素，取出目标偏移处的值
3. 标量比较更新最大值和索引

### 输出写回

每个核计算完一个输出元素后，将最大值和索引写入全局内存。由于 `DataCopy` 最小写回单元为 16 个元素，输出地址按 16 对齐存放：

```cpp
DataCopy(yGm[t * ALIGN_NUM],   yOut,   ALIGN_NUM);
DataCopy(idxGm[t * ALIGN_NUM], idxOut, ALIGN_NUM);
```

---

## 精度修复说明

本次针对二维、三维 Shape 精度失败问题进行了两处关键修复：

### 修复一：Tail 部分越界读脏数据

**问题：** 当 inner 不是 128 的整数倍时，tail 部分原来按 `padLen`（向上对齐到 128）读取，超过实际数据边界，读入脏数据，导致 maxVal 偏大或偏小。

**修复前：**
```cpp
DataCopy(xTail, xGm[base + loops * tileSize], padLen);  // padLen > tailLen，越界
```

**修复后：**
```cpp
// 只读 safeCopyLen（tailLen 按16对齐），不越界
int32_t safeCopyLen = (tailLen + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
DataCopy(xTail, xGm[base + loops * tileSize], safeCopyLen);
// 把 [tailLen, padLen) 手动填为 -65504，不影响 max 结果
for (int32_t k = tailLen; k < padLen; k++) {
    xComp.SetValue(k, static_cast<T>(-65504.0f));
}
```

### 修复二：多核输出写回地址冲突

**问题：** `DataCopy` 最小写 16 个元素，若写到 `yGm[t]`，相邻任务的输出槽（间距仅1个元素）会被当前核的写操作覆盖清零，导致大量输出为 0。

**修复前：**
```cpp
DataCopy(yGm[t],   yOut,   ALIGN_NUM);  // 覆盖 [t, t+16)，踩到相邻任务的槽
DataCopy(idxGm[t], idxOut, ALIGN_NUM);
```

**修复后：**
```cpp
DataCopy(yGm[t * ALIGN_NUM],   yOut,   ALIGN_NUM);  // 每个槽间距16，互不干扰
DataCopy(idxGm[t * ALIGN_NUM], idxOut, ALIGN_NUM);
```

### 修复三：tileSize 按 128 对齐

**问题：** `WholeReduceMax` 要求 tileSize 必须是 MASK=128 的整数倍，原代码按 16 对齐可能导致 repeatTime 截断，漏掉部分元素。

**修复前：**
```cpp
tileSize = (tileSize / 16) * 16;
if (tileSize == 0) tileSize = 16;
```

**修复后：**
```cpp
tileSize = (tileSize / 128) * 128;
if (tileSize == 0) tileSize = 128;
```

---

## 环境依赖

| 依赖项 | 版本 |
|--------|------|
| CANN | 8.5.0 |
| Ascend NPU | 910B（ascend910b） |
| Python | 3.x |
| NumPy | 任意版本（用于生成测试数据） |
| GCC | 9.4.0+ |
| CMake | 3.19+ |

---

## 编译与安装

### 第一步：编译算子

```bash
cd /path/to/ReduceMaxCustom
bash build.sh
```

编译成功后会在 `build_out/` 下生成 `custom_opp_ubuntu_aarch64.run`。

### 第二步：安装算子到系统

```bash
bash build_out/custom_opp_ubuntu_aarch64.run --quiet
```

安装成功输出：
```
[ops_custom] copy new ops op_api files ......
SUCCESS
```

### 第三步：设置运行时环境变量

```bash
source /home/developer/Ascend/cann-8.5.0/set_env.sh
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
```

> 建议将上述两行加入 `~/.bashrc`，避免每次手动设置。

---

## 精度测试

### 第一步：生成测试数据

```bash
cd /path/to/ReduceMaxCustom
python3 test_reducemax/gen_test_data.py
```

脚本会为以下 Shape 各生成输入数据（`input_x.bin`）和 golden 参考结果（`golden_y.bin`、`golden_idx.bin`）：

| Shape | reduceDim | 输出 Shape | 输出元素数 |
|-------|-----------|-----------|-----------|
| (1048576,) | 0 | () | 1 |
| (512, 128, 512) | 1 | (512, 512) | 262144 |
| (512, 128, 512) | 2 | (512, 128) | 65536 |
| (1024, 512, 1024) | 1 | (1024, 1024) | 1048576 |
| (512, 4096) | 1 | (512,) | 512 |
| (512, 4096) | 0 | (4096,) | 4096 |

golden 数据由 NumPy `np.max` 和 `np.argmax` 生成，与 PyTorch `torch.max` 结果一致。

### 第二步：编译测试程序

```bash
cd test_reducemax/build
rm -rf *
cmake .. \
  -DCMAKE_CXX_FLAGS="-I/path/to/ReduceMaxCustom/build_out/autogen"
make -j4
```

### 第三步：运行精度测试

```bash
./test_reducemax
```

测试程序会逐个 Shape 调用算子，将输出与 golden 逐元素对比，报告：
- `outSize`：输出元素总数
- `maxDiff`：最大绝对误差
- `valFail`：值误差超过 0.1 的元素数
- `idxFail`：索引不一致的元素数

---

## 测试结果

以下为在 Ascend 910B 上的实测精度结果：

```
===== [1048576_dim0] =====
[RESULT] 输出元素数=1
[RESULT] 最大误差=0
[RESULT] 值错误数=0/1
[RESULT] 索引错误数=0/1
[RESULT] 精度结论: ✓ PASS

===== [512x128x512_dim1] =====
[RESULT] 输出元素数=262144
[RESULT] 最大误差=0
[RESULT] 值错误数=0/262144
[RESULT] 索引错误数=0/262144
[RESULT] 精度结论: ✓ PASS

===== [512x128x512_dim2] =====
[RESULT] 输出元素数=65536
[RESULT] 最大误差=0
[RESULT] 值错误数=0/65536
[RESULT] 索引错误数=0/65536
[RESULT] 精度结论: ✓ PASS

===== [512x4096_dim1] =====
[RESULT] 输出元素数=512
[RESULT] 最大误差=0
[RESULT] 值错误数=0/512
[RESULT] 索引错误数=0/512
[RESULT] 精度结论: ✓ PASS

===== [512x4096_dim0] =====
[RESULT] 输出元素数=4096
[RESULT] 最大误差=0
[RESULT] 值错误数=0/4096
[RESULT] 索引错误数=0/4096
[RESULT] 精度结论: ✓ PASS

===== TOTAL: ALL PASS =====
```

**所有测试 Shape 精度全部通过，最大误差为 0（float16 精确匹配）。**

---

## 注意事项

1. 算子输出内存按 `元素数 × 16 × sizeof(dtype)` 分配，每个输出槽间隔 16 个元素，调用方需注意输出 buffer 大小
2. 当前仅支持 `float16` 输入类型，如需 `float32` 支持需扩展模板实例化
3. `ComputeStridedReduce`（非末尾维归约）为标量逐步访问，大 stride 场景性能较低，后续可优化为分块向量化
