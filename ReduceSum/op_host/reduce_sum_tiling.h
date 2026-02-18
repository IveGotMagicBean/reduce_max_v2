/**
 * @file reduce_sum_tiling.h
 * 精简版tiling - 控制在96字节以内
 */

#ifndef REDUCE_SUM_TILING_H
#define REDUCE_SUM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ReduceSumTilingData)
  // 基础信息 (必须保留，兼容baseline)
  TILING_DATA_FIELD_DEF(uint32_t, size);                    // 4B
  TILING_DATA_FIELD_DEF_ARR(int32_t, 20, x_ndarray);        // 80B
  TILING_DATA_FIELD_DEF(int32_t, x_dimensional);            // 4B  
  TILING_DATA_FIELD_DEF(int32_t, axes_num);                 // 4B
  TILING_DATA_FIELD_DEF(bool, keep_dims);                   // 1B
  TILING_DATA_FIELD_DEF(bool, ignore_nan);                  // 1B
  TILING_DATA_FIELD_DEF(uint8_t, dtype);                    // 1B
  // 到这里已经95字节，不能再加了！
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReduceSum, ReduceSumTilingData)

}

#endif
