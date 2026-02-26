# ReduceMax Custom 算子优化说明

## 算子功能
对输入张量沿指定维度求最大值，同时输出最大值的索引。

## 实现方案

### 核心优化：向量化 Last-Reduce
- 使用 AscendC `WholeReduceMax` 向量指令
- 每个 repeat 处理 128 个 float16 元素
- Double Buffer 流水隐藏 DMA 搬运延迟
- 20 核并行，按输出元素数均匀分配任务

### Tiling 策略
- outer × inner × stride 分解 reduce 维度
- tileSize 根据 inner 大小动态调整（256~8192）
- blockNum 最多 20 核，不超过输出元素数

### 支持场景
- Last-Reduce（stride=1）：向量化加速
- 非连续 Reduce（stride>1）：标量实现

## 性能数据（Ascend 910B4）
| Shape | dim | 耗时 | 带宽 |
|-------|-----|------|------|
| [512,128,512] | 2 | 2.1ms | 29.5 GB/s |
| [1024,512,1024] | 2 | 19ms | 52.4 GB/s |

## 编译运行
```bash
bash build.sh
cd build_out && bash custom_opp_ubuntu_aarch64.run
```
