/**
 * @file reduce_sum.cpp (op_kernel)
 * Optimized ReduceSum kernel for Ascend 910B.
 *
 * Optimization points:
 *  1. Multi-core parallel: each AICore handles its own slice
 *  2. Double-buffer pipeline: overlap DataCopy and Compute
 *  3. Vectorized reduction: WholeReduceSum / BlockReduceSum API
 *  4. Three paths:
 *     - LastReduce  (tilingMode=0): reduce沿最后维度，向量化最优
 *     - NonLastReduce (tilingMode=1): 中间维度reduce，逐行处理
 *     - FullReduce  (tilingMode=2): 整个tensor求和，流式累加
 */

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

// ============================================================
// 常量定义
// ============================================================
constexpr int32_t BUFFER_NUM    = 2;     // 双buffer
constexpr int32_t BLOCK_BYTES   = 32;    // 最小访存单元
constexpr int32_t ALIGN_ELEM_FP32 = 8;  // 32B / 4B
constexpr int32_t ALIGN_ELEM_FP16 = 16; // 32B / 2B
constexpr int32_t UB_BUF_SIZE   = 8192; // 每个buffer的最大元素数（fp32）

// ============================================================
// 工具函数：向上对齐
// ============================================================
__aicore__ inline int32_t AlignUp(int32_t val, int32_t align) {
    return (val + align - 1) / align * align;
}

// ============================================================
// Mode 0: LastReduce
// 每个核负责若干"行"（outerSize方向的若干个），
// 每行长度=innerSize，对每行做向量归约，结果写一个标量。
// 双buffer流水：prefetch下一行数据，同时计算当前行。
// ============================================================
template <typename T>
class KernelLastReduce {
public:
    __aicore__ inline KernelLastReduce() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR axes, GM_ADDR y,
        uint32_t outerSize, uint32_t innerSize,
        uint32_t coreDataNum,   // 本核负责的行数
        uint32_t coreOffset,    // 本核起始行索引
        uint32_t tileDataNum)
    {
        this->outerSize  = outerSize;
        this->innerSize  = innerSize;
        this->coreRows   = coreDataNum;   // 本核行数
        this->coreOffset = coreOffset;
        this->tileRows   = tileDataNum;   // 每tile处理的行数

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), outerSize * innerSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), outerSize);

        // innerSize对齐到32B
        int32_t alignedInner = AlignUp((int32_t)innerSize, ALIGN_ELEM_FP32);

        // 每tile读取tileRows行，每行innerSize元素
        uint32_t tileElem = tileRows * alignedInner;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileElem * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, AlignUp((int32_t)tileRows, ALIGN_ELEM_FP32) * sizeof(T));
        pipe.InitBuffer(calcBuf, alignedInner * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 处理所有行，双buffer流水
        uint32_t rows = coreRows;
        uint32_t processed = 0;

        while (processed < rows) {
            uint32_t batchRows = (rows - processed < tileRows) ? (rows - processed) : tileRows;
            CopyIn(processed, batchRows);
            Compute(processed, batchRows);
            CopyOut(processed, batchRows);
            processed += batchRows;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowStart, uint32_t rowCount)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        uint32_t gmOffset = (coreOffset + rowStart) * innerSize;
        int32_t copyLen = AlignUp((int32_t)(rowCount * innerSize), ALIGN_ELEM_FP32);
        DataCopy(xLocal, xGm[gmOffset], copyLen);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t rowStart, uint32_t rowCount)
    {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> tmpBuf = calcBuf.Get<T>();

        int32_t alignedInner = AlignUp((int32_t)innerSize, ALIGN_ELEM_FP32);

        for (uint32_t r = 0; r < rowCount; r++) {
            // 对第r行做归约
            LocalTensor<T> rowSlice = xLocal[r * alignedInner];
            ReduceSum(yLocal[r], rowSlice, tmpBuf, (int32_t)innerSize);
        }

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowStart, uint32_t rowCount)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        int32_t copyLen = AlignUp((int32_t)rowCount, ALIGN_ELEM_FP32);
        DataCopy(yGm[coreOffset + rowStart], yLocal, copyLen);
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC>            calcBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t coreRows;
    uint32_t coreOffset;
    uint32_t tileRows;
};

// ============================================================
// Mode 2: FullReduce
// 整个tensor求和。各核先各自求一个局部和，
// 再用AtomicAdd写到输出（单元素）。
// ============================================================
template <typename T>
class KernelFullReduce {
public:
    __aicore__ inline KernelFullReduce() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
        uint32_t totalSize,
        uint32_t coreDataNum,   // 本核负责的元素数
        uint32_t coreOffset,    // 本核起始元素索引
        uint32_t tileDataNum)
    {
        this->totalSize   = totalSize;
        this->coreElems   = coreDataNum;
        this->coreOffset  = coreOffset;
        this->tileElems   = tileDataNum;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), totalSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), 1);
        // workspace用于原子加的目标（单个float累加器）
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace), 1);

        int32_t alignedTile = AlignUp((int32_t)tileElems, ALIGN_ELEM_FP32);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedTile * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, ALIGN_ELEM_FP32 * sizeof(T));
        pipe.InitBuffer(calcBuf, alignedTile * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint32_t processed = 0;
        // 每个核维护一个局部累加器（放在UB里）
        LocalTensor<T> localAcc;
        bool firstTile = true;

        while (processed < coreElems) {
            uint32_t batchElems = (coreElems - processed < tileElems) ?
                                   (coreElems - processed) : tileElems;

            // CopyIn
            LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
            int32_t copyLen = AlignUp((int32_t)batchElems, ALIGN_ELEM_FP32);
            DataCopy(xLocal, xGm[coreOffset + processed], copyLen);
            inQueueX.EnQue(xLocal);

            // Compute: 对这批数据求和，累加到localAcc
            xLocal = inQueueX.DeQue<T>();
            LocalTensor<T> partialOut = outQueueY.AllocTensor<T>();
            LocalTensor<T> tmpBuf = calcBuf.Get<T>();

            ReduceSum(partialOut[0], xLocal, tmpBuf, (int32_t)batchElems);

            if (firstTile) {
                // 第一个tile：直接保存
                localAcc = partialOut;
                firstTile = false;
                outQueueY.EnQue<T>(partialOut);
            } else {
                // 后续tile：累加
                LocalTensor<T> accLocal = outQueueY.DeQue<T>();
                Add(accLocal[0], accLocal[0], partialOut[0], 1);
                outQueueY.EnQue<T>(accLocal);
                outQueueY.FreeTensor(partialOut);
            }

            inQueueX.FreeTensor(xLocal);
            processed += batchElems;
        }

        // 将本核局部和用AtomicAdd写入全局输出
        LocalTensor<T> finalAcc = outQueueY.DeQue<T>();
        // 使用SetAtomicAdd确保多核安全累加
        SetAtomicAdd<T>();
        DataCopy(yGm[0], finalAcc, ALIGN_ELEM_FP32);
        SetAtomicNone();
        outQueueY.FreeTensor(finalAcc);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC>            calcBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> wsGm;

    uint32_t totalSize;
    uint32_t coreElems;
    uint32_t coreOffset;
    uint32_t tileElems;
};

// ============================================================
// Mode 1: NonLastReduce (通用多维reduce)
// 对于中间轴reduce，输出的每个元素需要从输入中跨步累加。
// 使用与baseline相同的逻辑，但加入多核并行（按outerSize切分）。
// ============================================================
template <typename T>
class KernelNonLastReduce {
public:
    __aicore__ inline KernelNonLastReduce() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR axes, GM_ADDR y,
        int32_t size,
        int32_t x_ndarray[], int32_t x_dimensional, int32_t axes_num,
        bool keep_dims, bool ignore_nan,
        uint32_t coreOutputNum,  // 本核负责的输出元素数
        uint32_t coreOutputOff)  // 本核起始输出元素索引
    {
        this->x_dimensional  = x_dimensional;
        this->axes_num       = axes_num;
        this->coreOutputNum  = coreOutputNum;
        this->coreOutputOff  = coreOutputOff;

        for (int i = 0; i < x_dimensional; i++) this->x_ndarray[i] = x_ndarray[i];

        axesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(axes), axes_num);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), size);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), size);

        // 读取axes值
        for (int i = 0; i < axes_num; i++) {
            dim[i] = axesGm.GetValue(i);
            if (dim[i] < 0) dim[i] += x_dimensional;
        }

        // 计算cycles / interval / loopCount
        int32_t lc = 1, cyc = 1, itv = 1;
        for (int i = 0; i < x_dimensional; i++) lc *= x_ndarray[i];
        for (int i = 0; i < axes_num; i++) lc /= x_ndarray[dim[i]];
        for (int i = 0; i < axes_num; i++) cyc *= x_ndarray[dim[i]];
        for (int i = dim[axes_num-1]+1; i < x_dimensional; i++) itv *= x_ndarray[i];

        this->cycles    = cyc;
        this->interval  = itv;
        this->loopCount = lc;
    }

    __aicore__ inline void Process()
    {
        // 按本核负责的输出元素范围处理
        uint32_t start = coreOutputOff;
        uint32_t end   = coreOutputOff + coreOutputNum;
        if (end > (uint32_t)loopCount) end = (uint32_t)loopCount;

        for (uint32_t z = start; z < end; z++) {
            int32_t x_num = (int32_t)z / interval;
            x_num = x_num * cycles * interval + (int32_t)z % interval;
            float temp_sum = 0.0f;
            for (int32_t i = 0; i < cycles; i++) {
                int32_t temp_num = x_num + i * interval;
                float temp_add = (float)xGm.GetValue(temp_num);
                temp_sum += temp_add;
            }
            yGm.SetValue(z, (T)temp_sum);
        }
    }

private:
    GlobalTensor<T>       xGm;
    GlobalTensor<int32_t> axesGm;
    GlobalTensor<T>       yGm;

    int32_t x_ndarray[20];
    int32_t x_dimensional;
    int32_t axes_num;
    int32_t dim[20];
    int32_t cycles;
    int32_t interval;
    int32_t loopCount;
    uint32_t coreOutputNum;
    uint32_t coreOutputOff;
};

// ============================================================
// Kernel入口
// ============================================================
extern "C" __global__ __aicore__ void reduce_sum(
    GM_ADDR x, GM_ADDR axes, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t blockIdx    = GetBlockIdx();
    uint32_t coreNum     = tiling_data.coreNum;
    uint32_t bigCoreNum  = tiling_data.bigCoreNum;

    // 计算本核的数据偏移和数据量
    uint32_t coreDataNum, coreOffset;
    if (blockIdx < bigCoreNum) {
        coreDataNum = tiling_data.bigCoreDataNum;
        coreOffset  = blockIdx * tiling_data.bigCoreDataNum;
    } else {
        coreDataNum = tiling_data.smallCoreDataNum;
        coreOffset  = bigCoreNum * tiling_data.bigCoreDataNum
                    + (blockIdx - bigCoreNum) * tiling_data.smallCoreDataNum;
    }

    if (coreDataNum == 0) return;

    uint32_t tilingMode = tiling_data.tilingMode;

    if (TILING_KEY_IS(1)) {
        // LastReduce: coreDataNum = 行数, 每行innerSize元素
        KernelLastReduce<DTYPE_X> op;
        op.Init(x, axes, y,
                tiling_data.outerSize,
                tiling_data.innerSize,
                coreDataNum,           // 本核行数
                coreOffset,            // 本核起始行
                tiling_data.tileDataNum);
        op.Process();
    }
    else if (TILING_KEY_IS(3)) {
        // FullReduce: coreDataNum = 元素数
        KernelFullReduce<DTYPE_X> op;
        op.Init(x, y, workspace,
                tiling_data.size,
                coreDataNum,
                coreOffset,
                tiling_data.tileDataNum);
        op.Process();
    }
    else {
        // NonLastReduce (tilingKey=2): 通用路径，多核按输出元素切分
        KernelNonLastReduce<DTYPE_X> op;
        op.Init(x, axes, y,
                (int32_t)tiling_data.size,
                tiling_data.x_ndarray,
                tiling_data.x_dimensional,
                tiling_data.axes_num,
                tiling_data.keep_dims,
                tiling_data.ignore_nan,
                coreDataNum,
                coreOffset);
        op.Process();
    }
}
