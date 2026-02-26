#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM  = 2;
constexpr int32_t ALIGN_NUM   = 16;
constexpr int32_t MASK        = 128;
constexpr int32_t REP_STRIDE  = 8;

template<typename T>
class ReduceMaxKernel {
public:
    __aicore__ inline ReduceMaxKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx,
                                GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(td, tiling);
        outer    = td.outer;
        inner    = td.inner;
        stride   = td.stride;
        blockNum = td.blockNum;
        tileSize = td.tileSize;

        uint32_t totalOut = outer * stride;
        uint32_t blockId  = GetBlockIdx();
        uint32_t base     = totalOut / blockNum;
        uint32_t tail     = totalOut % blockNum;
        taskLen   = base + (blockId < tail ? 1 : 0);
        taskStart = blockId * base + (blockId < tail ? blockId : tail);

        xGm.SetGlobalBuffer((__gm__ T*)x,           outer * inner * stride);
        yGm.SetGlobalBuffer((__gm__ T*)y,            totalOut);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)idx,  totalOut);

        pipe.InitBuffer(inQueueX,    BUFFER_NUM, tileSize * sizeof(T));
        pipe.InitBuffer(outQueueY,   1, ALIGN_NUM * sizeof(T));
        pipe.InitBuffer(outQueueIdx, 1, ALIGN_NUM * sizeof(int32_t));
        // workBuf: ONLY_VALUE每个repeat输出1个值
        uint32_t workLen = ((tileSize / MASK) + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM;
        pipe.InitBuffer(workBuf, workLen * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = taskStart; t < taskStart + taskLen; t++) {
            uint32_t o = t / stride;
            uint32_t s = t % stride;

            T       maxVal = static_cast<T>(-65504.0f);
            int32_t maxIdx = 0;

            if (stride == 1) {
                ComputeLastReduce(o, maxVal, maxIdx);
            } else {
                ComputeStridedReduce(o, s, maxVal, maxIdx);
            }

            LocalTensor<T>       yBuf   = outQueueY.AllocTensor<T>();
            LocalTensor<int32_t> idxBuf = outQueueIdx.AllocTensor<int32_t>();
            yBuf.SetValue(0, maxVal);
            idxBuf.SetValue(0, maxIdx);
            outQueueY.EnQue(yBuf);
            outQueueIdx.EnQue(idxBuf);
            LocalTensor<T>       yOut   = outQueueY.DeQue<T>();
            LocalTensor<int32_t> idxOut = outQueueIdx.DeQue<int32_t>();
            DataCopy(yGm[t],   yOut,   ALIGN_NUM);
            DataCopy(idxGm[t], idxOut, ALIGN_NUM);
            outQueueY.FreeTensor(yOut);
            outQueueIdx.FreeTensor(idxOut);
        }
    }

private:
    __aicore__ inline void ComputeLastReduce(uint32_t o,
                                              T& maxVal, int32_t& maxIdx)
    {
        uint32_t base    = o * inner;
        int32_t  loops   = inner / tileSize;
        int32_t  tailLen = inner % tileSize;

        maxVal = static_cast<T>(-65504.0f);
        maxIdx = 0;

        if (loops > 0) {
            LocalTensor<T> x0 = inQueueX.AllocTensor<T>();
            DataCopy(x0, xGm[base], tileSize);
            inQueueX.EnQue(x0);
        }

        for (int32_t i = 0; i < loops; i++) {
            if (i + 1 < loops) {
                LocalTensor<T> xNext = inQueueX.AllocTensor<T>();
                DataCopy(xNext, xGm[base + (i + 1) * tileSize], tileSize);
                inQueueX.EnQue(xNext);
            }

            LocalTensor<T> xCur      = inQueueX.DeQue<T>();
            LocalTensor<T> work      = workBuf.Get<T>();
            int32_t        repeatTime = tileSize / MASK;

            // 向量化：每个repeat求最大值存到work[r]
            // xCur数据保留不变（work是独立buffer）
            WholeReduceMax(work, xCur, MASK, repeatTime,
                           1, 1, REP_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();

            // 找最大值所在的repeat，然后在该repeat里精确找idx
            for (int32_t r = 0; r < repeatTime; r++) {
                T v = work.GetValue(r);
                if ((float)v > (float)maxVal) {
                    maxVal = v;
                    // 在repeat r里精确找位置，xCur数据还在
                    int32_t rBase = r * MASK;
                    for (int32_t k = rBase; k < rBase + MASK; k++) {
                        if ((float)xCur.GetValue(k) == (float)v) {
                            maxIdx = i * tileSize + k;
                            break;
                        }
                    }
                }
            }

            inQueueX.FreeTensor(xCur);
        }

        if (tailLen > 0) {
            int32_t padLen     = (tailLen + MASK - 1) / MASK * MASK;
            int32_t repeatTime = padLen / MASK;

            LocalTensor<T> xTail = inQueueX.AllocTensor<T>();
            DataCopy(xTail, xGm[base + loops * tileSize], padLen);
            inQueueX.EnQue(xTail);
            LocalTensor<T> xComp = inQueueX.DeQue<T>();
            LocalTensor<T> work  = workBuf.Get<T>();

            WholeReduceMax(work, xComp, MASK, repeatTime,
                           1, 1, REP_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();

            for (int32_t r = 0; r < repeatTime; r++) {
                T v = work.GetValue(r);
                if ((float)v > (float)maxVal) {
                    maxVal = v;
                    int32_t rBase = r * MASK;
                    int32_t rEnd  = rBase + MASK;
                    if (rEnd > tailLen) rEnd = tailLen;
                    for (int32_t k = rBase; k < rEnd; k++) {
                        if ((float)xComp.GetValue(k) == (float)v) {
                            maxIdx = loops * tileSize + k;
                            break;
                        }
                    }
                }
            }

            inQueueX.FreeTensor(xComp);
        }
    }

    __aicore__ inline void ComputeStridedReduce(uint32_t o, uint32_t s,
                                                 T& maxVal, int32_t& maxIdx)
    {
        maxVal = static_cast<T>(-65504.0f);
        maxIdx = 0;
        for (int32_t k = 0; k < (int32_t)inner; k++) {
            int32_t gIdx       = o * inner * stride + k * stride + s;
            int32_t alignedIdx = (gIdx / ALIGN_NUM) * ALIGN_NUM;
            int32_t offset     = gIdx - alignedIdx;
            LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
            DataCopy(xLocal, xGm[alignedIdx], ALIGN_NUM);
            inQueueX.EnQue(xLocal);
            LocalTensor<T> xComp = inQueueX.DeQue<T>();
            T val = xComp.GetValue(offset);
            inQueueX.FreeTensor(xComp);
            if ((float)val > (float)maxVal) {
                maxVal = val;
                maxIdx = k;
            }
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, 1>          outQueueY;
    TQue<QuePosition::VECOUT, 1>          outQueueIdx;
    TBuf<QuePosition::VECCALC>            workBuf;

    GlobalTensor<T>       xGm;
    GlobalTensor<T>       yGm;
    GlobalTensor<int32_t> idxGm;

    uint32_t outer, inner, stride, blockNum, tileSize;
    uint32_t taskStart, taskLen;
};

extern "C" __global__ __aicore__ void reduce_max_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx,
    GM_ADDR workspace, GM_ADDR tiling)
{
    ReduceMaxKernel<half> kernel;
    kernel.Init(x, y, idx, workspace, tiling);
    kernel.Process();
}
