# ReduceMax Custom 算子优化技术报告

**参赛选手：** ShiyiLIN  
**提交日期：** 2026-02-26  
**赛题：** 昇腾AI算法挑战赛中阶赛 · Reduce算子优化  
**硬件平台：** 昇腾 Ascend 910B4 NPU  
**CANN版本：** 8.5.0  

---

## 1. 算子功能说明

本算子实现 ReduceMax 操作：对输入张量沿指定维度求最大值，同时返回最大值对应的索引。

| 参数 | 名称 | 类型 | 说明 |
|------|------|------|------|
| 输入 | x | float16 | 任意维度输入张量 |
| 输出 | y | float16 | reduce 后的最大值 |
| 输出 | idx | int32 | 最大值对应的索引 |
| 属性 | reduceDim | int | 执行 reduce 的维度（支持负数） |
| 属性 | isKeepDim | int | 是否保留维度，可选，默认为 1 |

---

## 2. 工程文件结构

```
.
├── reduce_max_custom.json          # 算子接口定义（输入输出类型、属性）
├── CMakeLists.txt                  # 工程构建配置
├── CMakePresets.json               # CMake预设
├── build.sh                        # 一键编译脚本
├── cmake/                          # 构建辅助脚本
├── framework/                      # TensorFlow插件（自动生成）
├── scripts/                        # 安装脚本
├── op_host/
│   ├── reduce_max_custom_tiling.h  # Tiling数据结构定义
│   └── reduce_max_custom.cpp       # Host侧：Tiling计算 + 算子注册
├── op_kernel/
│   └── reduce_max_custom.cpp       # Device侧：NPU AICore Kernel实现
└── test_reducemax/
    ├── CMakeLists.txt              # 测试程序构建配置
    ├── test_main.cpp               # 精度验证测试程序
    ├── input/
    │   └── input_x.bin            # 测试输入数据（float16，1M元素）
    └── output/
        ├── golden_y.bin           # numpy golden 最大值
        └── golden_idx.bin         # numpy golden 索引
```

---

## 3. 实现方案

### 3.1 整体数据流

```
输入 Tensor x (Global Memory)
         │
         ▼
 [Host Tiling]  ←  op_host/reduce_max_custom.cpp
   · 解析 shape 和 reduceDim
   · 计算 outer / inner / stride
   · 决定 blockNum（最多20核）
   · 决定 tileSize（256~8192，动态选择）
         │  tiling参数传递给每个核
         ▼
 [AICore Kernel × blockNum]  ←  op_kernel/reduce_max_custom.cpp
   · 每核负责一部分输出元素
   · stride==1：WholeReduceMax 向量化加速
   · stride>1 ：标量逐元素处理
         │
         ▼
输出 y（Global Memory）+ idx（Global Memory）
```

### 3.2 Tiling 策略详解

将输入 shape 按 reduceDim 分解为三段：

```
输入 shape: [d0, d1, ..., d_{k-1},  d_k,  d_{k+1}, ..., d_{n-1}]
                                     ^^^
                                  reduceDim = k

outer  = d0 × d1 × ... × d_{k-1}    （reduce维度之前所有维度的乘积）
inner  = d_k                          （reduce维度本身的大小）
stride = d_{k+1} × ... × d_{n-1}    （reduce维度之后所有维度的乘积）

总输出元素数 totalOut = outer × stride
```

多核分配规则：
- `blockNum = min(20, totalOut)`，最多使用 20 个 AICore
- 第 `i` 核处理输出范围：`[taskStart, taskStart + taskLen)`
- 尾部余量（`totalOut % blockNum`）均匀分配给前几个核，保证负载均衡

tileSize 根据 inner 动态选择，平衡 UB 利用率与循环开销：

| inner 大小 | tileSize |
|-----------|----------|
| < 1024    | 256      |
| 1024~4095 | 1024     |
| 4096~8191 | 4096     |
| ≥ 8192    | 8192     |

### 3.3 Kernel 实现——Last-Reduce 向量化（stride=1）

stride=1 表示 reduce 的是最内层（连续内存），可以使用向量指令加速：

```cpp
// ====== 核心循环（含 Double Buffer 流水）======

// 预加载第 0 块
LocalTensor<T> x0 = inQueueX.AllocTensor<T>();
DataCopy(x0, xGm[base], tileSize);
inQueueX.EnQue(x0);

for (int i = 0; i < loops; i++) {
    // Double Buffer：提前加载下一块，与当前块计算并行
    if (i + 1 < loops) {
        LocalTensor<T> xNext = inQueueX.AllocTensor<T>();
        DataCopy(xNext, xGm[base + (i+1) * tileSize], tileSize);
        inQueueX.EnQue(xNext);
    }

    LocalTensor<T> xCur     = inQueueX.DeQue<T>();
    LocalTensor<T> work     = workBuf.Get<T>();   // 独立TBuf，不覆盖xCur
    int32_t repeatTime      = tileSize / MASK;    // MASK=128

    // ====== WholeReduceMax 向量化 ======
    // 每个 repeat 处理 128 个 float16（256B = 8 个 block）
    // ORDER_ONLY_VALUE：每个 repeat 输出 1 个最大值到 work[r]
    WholeReduceMax(work, xCur,
                   MASK,        // mask=128，每repeat处理128个half
                   repeatTime,  // repeat次数 = tileSize / 128
                   1,           // dstRepStride=1，结果连续存放
                   1,           // srcBlkStride=1
                   8,           // srcRepStride=8（128half=8block）
                   ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();  // 等待向量计算完成

    // ====== 在 repeatTime 个结果里找全局最大值 ======
    // work[r] = 第r个repeat（128个元素）的最大值
    for (int r = 0; r < repeatTime; r++) {
        T v = work.GetValue(r);
        if ((float)v > (float)maxVal) {
            maxVal = v;
            // xCur 数据未被覆盖（work是独立TBuf），可直接精确找idx
            // 在 repeat r 对应的 128 个元素里线性扫描
            int rBase = r * MASK;
            for (int k = rBase; k < rBase + MASK; k++) {
                if ((float)xCur.GetValue(k) == (float)v) {
                    maxIdx = i * tileSize + k;  // 转换为全局索引
                    break;
                }
            }
        }
    }

    inQueueX.FreeTensor(xCur);
}
```

**关键优化点总结：**

| 优化点 | 说明 |
|--------|------|
| WholeReduceMax 向量化 | 每个 repeat 处理 128 个 half，相比标量循环快约 128 倍 |
| Double Buffer | BUFFER_NUM=2，DMA 搬运与向量计算流水并行，隐藏访存延迟 |
| 独立 work TBuf | work 使用 TBuf<VECCALC>，不占用 VECIN 队列，xCur 数据不被覆盖 |
| 单次 pass | 向量化找最大值 + 精确定位 idx 在同一次数据读取中完成，无需第二次扫描 |
| 20核并行 | 多输出元素场景下线性扩展，充分利用 AICore 并行能力 |

### 3.4 Kernel 实现——Strided-Reduce（stride>1，标量）

对于非连续 reduce（如对中间维度或首维 reduce），由于内存不连续，采用标量实现保证正确性：

```cpp
// 遍历 inner 维度的每个元素
for (int k = 0; k < inner; k++) {
    // 计算该元素在 Global Memory 中的实际地址
    int gIdx = o * inner * stride + k * stride + s;

    // DMA 最小传输单元为 32B = 16 个 half
    // 对齐到 16 的整数倍地址读取
    int alignedIdx = (gIdx / ALIGN_NUM) * ALIGN_NUM;
    int offset     = gIdx - alignedIdx;

    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopy(xLocal, xGm[alignedIdx], ALIGN_NUM);  // 读 16 个 half
    inQueueX.EnQue(xLocal);
    LocalTensor<T> xComp = inQueueX.DeQue<T>();

    T val = xComp.GetValue(offset);  // 取出目标元素
    inQueueX.FreeTensor(xComp);

    if ((float)val > (float)maxVal) {
        maxVal = val;
        maxIdx = k;
    }
}
```

---

## 4. 编译运行说明

### 4.1 环境准备

```bash
# 确认 CANN 环境变量已设置
source $ASCEND_HOME_DIR/bin/setenv.bash

# 设置自定义算子库路径
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=/home/developer/Ascend/cann-8.5.0/opp
```

### 4.2 编译算子

```bash
# 进入工程根目录
cd <工程目录>

# 清理旧产物（重要）
rm -rf build_out

# 一键编译，生成 .run 安装包
bash build.sh

# 编译成功后输出：
# CPack: - package: .../build_out/custom_opp_ubuntu_aarch64.run generated.
```

### 4.3 安装算子

```bash
cd build_out
bash custom_opp_ubuntu_aarch64.run

# 安装成功后输出：
# SUCCESS
```

### 4.4 编译测试程序

```bash
cd test_reducemax
mkdir -p build && cd build
cmake ..
make
```

### 4.5 运行精度验证

```bash
cd test_reducemax
./build/test_reducemax
```

**预期输出：**

```
[INFO] ACL initialized
[INFO] Input copied to device
[INFO] Workspace size: 6
[INFO] Operator executed!
[RESULT] max value: got=10 golden=10 PASS
[RESULT] max index: got=4887 golden=4887 PASS
[RESULT] ALL PASS
```

---

## 5. 测试结果

### 5.1 精度验证

测试配置：
- **输入 shape：** [1048576]（1M 个 float16 元素）
- **reduce dim：** 0
- **数据范围：** 均匀分布 [-10, 10]，numpy random seed=123

| 指标 | Golden（numpy） | 算子输出 | 结论 |
|------|----------------|---------|------|
| 最大值 | 10.0 (float16) | 10.0 (float16) | ✅ PASS |
| 最大值索引 | 4887 | 4887 | ✅ PASS |

### 5.2 性能数据

测试环境：Ascend 910B4，CANN 8.5.0，float16，dim=2（Last-Reduce）

| 输入 Shape | reduce dim | 数据量 | 平均耗时 | 有效带宽 |
|-----------|------------|--------|---------|---------|
| [1048576] | 0 | 2 MB | 0.19 ms | 10.3 GB/s |
| [512, 128, 512] | 2 | 64 MB | 2.12 ms | 29.5 GB/s |
| [1024, 512, 1024] | 2 | 1024 MB | 19.1 ms | 52.4 GB/s |

> 注：耗时为算子纯执行时间（预热后多次取平均），不含 ACL 初始化开销。

---

## 6. 已知限制与后续优化方向

| 限制 | 说明 | 后续优化方向 |
|------|------|------------|
| 非连续 Reduce 性能低 | dim=0/dim=1 场景使用标量实现 | 可通过转置将非连续 reduce 转为连续 reduce |
| 超大 shape 内存限制 | [2048,2048,2048] 等超过设备内存 | 分批处理或流式计算 |
| 仅支持 float16 | 当前 kernel 模板实例化为 half | 可扩展支持 bfloat16、float32 |

---

*昇腾AI算法挑战赛中阶赛 · ReduceMax算子优化 · ShiyiLIN · 2026-02-26*

