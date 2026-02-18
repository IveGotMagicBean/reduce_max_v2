/**
* @file reduce_sum_tiling.h
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef REDUCE_SUM_TILING_H
#define REDUCE_SUM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceSumTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x_ndarray);
  TILING_DATA_FIELD_DEF(int32_t, x_dimensional);
  TILING_DATA_FIELD_DEF(int32_t, axes_num);
  TILING_DATA_FIELD_DEF(bool, keep_dims);
  TILING_DATA_FIELD_DEF(bool, ignore_nan);
  TILING_DATA_FIELD_DEF(uint8_t, dtype);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceSum, ReduceSumTilingData)
}

#endif // REDUCE_SUM_TILING_H