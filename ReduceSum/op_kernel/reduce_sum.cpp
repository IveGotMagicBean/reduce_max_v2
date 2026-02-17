/**
* @file reduce_sum.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <type_traits>
#include <cmath>
#include "kernel_operator.h"

#define K_MAX_SHAPE_DIM 0

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelReduceSum
{
public:
    __aicore__ inline KernelReduceSum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR axes, GM_ADDR y,int32_t size, int32_t x_ndarray[], int32_t x_dimensional, int32_t axes_num,bool keep_dims, bool ignore_nan, uint8_t dtype)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->x_dimensional = x_dimensional;
        this->axes_num = axes_num;
        this->keep_dims = keep_dims;
        this->ignore_nan = ignore_nan;
        this->dtype = dtype;
        axesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_AXES *>(axes), 1);

        size = 1;
        for (int i = 0; i < x_dimensional; i++)
        {
            this->x_ndarray[i] = x_ndarray[i];
            size *= x_ndarray[i];
        }

        this->size = size;

        for (int i = 0; i < this->axes_num; i++)
        {
            this->dim[i] = axesGm.GetValue(i);
            if (this->dim[i] < 0)
            {
                this->dim[i] += this->x_dimensional;
            }
        }

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x), size);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y), size);

        int32_t cycles = 1;
        int32_t interval = 1;
        int32_t loopCount = 1;

        for (int i = 0; i < this->x_dimensional; i++)
        {
            loopCount *= this->x_ndarray[i];
        }
        for (int i = 0; i < this->axes_num; i++)
        {
            loopCount = loopCount / this->x_ndarray[this->dim[i]];
        }

        for (int i = 0; i < this->axes_num; i++)
        {
            cycles *= this->x_ndarray[this->dim[i]];
        }

        for (int i = this->dim[this->axes_num - 1] + 1; i < this->x_dimensional; i++)
        {
            interval *= this->x_ndarray[i];
        }

        this->cycles = cycles;
        this->interval = interval;
        this->loopCount = loopCount;

        if (this->interval == 1)
        {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, BUFFER_SIZE_X * sizeof(DTYPE_X));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, BUFFER_SIZE_Y * sizeof(DTYPE_Y));
            pipe.InitBuffer(QueueBuff, BUFFER_SIZE_Y * sizeof(DTYPE_X));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<DTYPE_Y, float>)
        {
            if ((this->interval == 1) && (BUFFER_SIZE_X % this->cycles == 0) && (this->cycles <= BUFFER_SIZE_X))
            {
                int capacity = BUFFER_SIZE_X / this->cycles;
                for (int32_t i = 0; i < this->loopCount; i++)
                {
                    if ((this->loopCount - i) >= capacity)
                    {
                        CopyIn(i, capacity);
                        Compute(i, capacity);
                        CopyOut(i, capacity);
                    }
                    else
                    {
                        CopyIn(i, this->loopCount - i);
                        Compute(i, this->loopCount - i);
                        CopyOut(i, this->loopCount - i);
                    }
                    i += capacity;
                }
            }
            else
            {
                DTYPE_Y temp_sum;
                DTYPE_Y temp_add;
                for (int z = 0; z < this->loopCount; z++)
                {
                    int32_t x_num = z / this->interval;
                    x_num = x_num * this->cycles * this->interval + z % this->interval;
                    for (int i = 0; i < this->cycles; i++)
                    {
                        int32_t temp_num = x_num + i * this->interval;
                        temp_add = xGm.GetValue(temp_num);

                        if (i == 0)
                        {
                            temp_sum = (float)temp_add;
                        }
                        else
                        {
                            temp_sum = (float)temp_sum + (float)temp_add;
                        }
                    }
                    yGm.SetValue(z, (DTYPE_Y)temp_sum);
                }
            }
        }
        else
        {
            DTYPE_Y temp_sum;
            DTYPE_Y temp_add;
            for (int z = 0; z < this->loopCount; z++)
            {
                int32_t x_num = z / this->interval;
                x_num = x_num * this->cycles * this->interval + z % this->interval;
                for (int i = 0; i < this->cycles; i++)
                {
                    int32_t temp_num = x_num + i * this->interval;
                    temp_add = xGm.GetValue(temp_num);

                    if (i == 0)
                    {
                        temp_sum = (float)temp_add;
                    }
                    else
                    {
                        temp_sum = (float)temp_sum + (float)temp_add;
                    }
                }
                yGm.SetValue(z, (DTYPE_Y)temp_sum);
            }
        }
    }

    private:
        static constexpr int CYCLE_UNIT_8 = 8;
        static constexpr int CYCLE_UNIT_64 = 64;
        static constexpr int CYCLE_UNIT_128 = 128;
        static constexpr int BUFFER_SIZE_X = 8192;
        static constexpr int BUFFER_SIZE_Y = 1024;
    __aicore__ inline void CopyIn(int32_t i, int32_t capacity)
    {
        static constexpr int32_t TensorAlignment = 8;

        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[i * this->cycles], (this->cycles * capacity + TensorAlignment - 1) / TensorAlignment * TensorAlignment);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t i, int32_t capacity)
    {
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        auto buff = QueueBuff.Get<float>();
        int cyl = this->cycles;
        if (cyl % CYCLE_UNIT_8 == 0 && cyl >= CYCLE_UNIT_64)
        {
            cyl /= CYCLE_UNIT_8;
            BlockReduceSum(buff, xLocal, CYCLE_UNIT_128, CYCLE_UNIT_64, 1, 1, CYCLE_UNIT_8);
            if (cyl % CYCLE_UNIT_8 == 0 && cyl >= CYCLE_UNIT_64)
            {
                cyl /= CYCLE_UNIT_8;
                BlockReduceSum(xLocal, buff, CYCLE_UNIT_64, CYCLE_UNIT_64, 1, 1, CYCLE_UNIT_8);
                WholeReduceSum(yLocal, xLocal, cyl, capacity, 1, 1, cyl / CYCLE_UNIT_8);
            }
            else
            {
                WholeReduceSum(yLocal, buff, cyl, capacity, 1, 1, cyl / CYCLE_UNIT_8);
            }
        }
        else
        {
            for (int z = 0; z < capacity; z++)
            {
                ReduceSum(yLocal[z], xLocal[this->cycles * z], buff, this->cycles);
            }
        }
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t i, int32_t j)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        constexpr int CYCLE_ADJUSTMENT = 8;
        DataCopy(yGm[i], yLocal, (j + CYCLE_ADJUSTMENT - 1) / CYCLE_ADJUSTMENT * CYCLE_ADJUSTMENT);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> QueueBuff;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_AXES> axesGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t size;
    int32_t x_ndarray[10];
    int32_t x_dimensional;
    int32_t axes_num;
    int32_t dim[10];

    bool keep_dims;
    bool ignore_nan;
    uint8_t dtype;

    int32_t cycles;
    int32_t interval;
    int32_t loopCount;
};

template <typename T>
class KernelReduceSumBroadcast
{
public:
    __aicore__ inline KernelReduceSumBroadcast() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR axes, GM_ADDR y,
                                int32_t size, int32_t x_ndarray[], int32_t x_dimensional, int32_t axes_num,
                                bool keep_dims, bool ignore_nan, uint8_t dtype)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->x_dimensional = x_dimensional;
        this->axes_num = axes_num;
        this->keep_dims = keep_dims;
        this->ignore_nan = ignore_nan;
        this->dtype = dtype;

        axesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_AXES *>(axes), 1);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x), 1);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y), 1);

        for (int i = 0; i < this->axes_num; i++)
        {
            this->dim[i] = axesGm.GetValue(i);
            if (this->dim[i] < 0)
                this->dim[i] += this->x_dimensional;
        }

        for (int i = 0; i < x_dimensional; i++)
        {
            int flag = 0;
            this->x_shape[x_dimensional - i - 1] = x_ndarray[i];
            for (int j = 0; j < axes_num; j++)
            {
                if (this->dim[j] == i)
                {
                    flag = 1;
                    break;
                }
            }
            if (flag == 1)
            {
                this->y_shape[x_dimensional - i - 1] = 1;
            }
            else
            {
                this->y_shape[x_dimensional - i - 1] = x_ndarray[i];
            }
        }
        y_sumndarray[0] = 1;
        x_sumndarray[0] = 1;
        for (int i = 1; i <= x_dimensional; i++)
        {
            y_sumndarray[i] = y_sumndarray[i - 1] * this->y_shape[i - 1];
            x_sumndarray[i] = x_sumndarray[i - 1] * this->x_shape[i - 1];
        }
    }

    __aicore__ inline void Process()
    {
        DTYPE_Y temp_sum, temp_add1, temp_add2;
        int dim = this->x_dimensional;
        for (int j = 0; j < this->y_sumndarray[dim]; j++)
        {
            yGm.SetValue(j, (DTYPE_Y)0);
        }

        for (int32_t j = 0; j < this->x_sumndarray[dim]; j++)
        {
            int32_t y_start = 0;
            for (int k = 0; k < dim; k++)
            {
                if (this->y_shape[k] != 1)
                {
                    y_start += this->y_sumndarray[k] * (j / this->x_sumndarray[k] % this->x_shape[k]);
                }
            }
            temp_add1 = xGm.GetValue(j);
            temp_add2 = yGm.GetValue(y_start);
            temp_sum = (float)temp_add1 + (float)temp_add2;
            yGm.SetValue(y_start, (DTYPE_Y)temp_sum);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> QueueBuff;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_AXES> axesGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t size;
    int32_t x_shape[20];
    int32_t x_sumndarray[20];
    int32_t x_dimensional;
    int32_t y_shape[20];
    int32_t y_sumndarray[20];
    int32_t y_dimensional;
    int32_t axes_num;
    int32_t dim[20];

    bool keep_dims;
    bool ignore_nan;
    uint8_t dtype;
};
extern "C" __global__ __aicore__ void reduce_sum(GM_ADDR x, GM_ADDR axes, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(1))
    {
        KernelReduceSum<DTYPE_X> op;
        op.Init(x, axes, y,
                tiling_data.size, tiling_data.x_ndarray, tiling_data.x_dimensional, tiling_data.axes_num,
                tiling_data.keep_dims, tiling_data.ignore_nan, tiling_data.dtype);
        op.Process();
    }
    else if (TILING_KEY_IS(2))
    {
        KernelReduceSumBroadcast<DTYPE_X> op;
        op.Init(x, axes, y,
                tiling_data.size, tiling_data.x_ndarray, tiling_data.x_dimensional, tiling_data.axes_num,
                tiling_data.keep_dims, tiling_data.ignore_nan, tiling_data.dtype);
        op.Process();
    }
}
