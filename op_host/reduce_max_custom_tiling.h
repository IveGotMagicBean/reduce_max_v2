#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceMaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outer);      // reduce维度前面所有维度之积
  TILING_DATA_FIELD_DEF(uint32_t, inner);      // reduce维度的大小
  TILING_DATA_FIELD_DEF(uint32_t, stride);     // reduce维度后面所有维度之积
  TILING_DATA_FIELD_DEF(uint32_t, blockNum);   // 实际使用的核数
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);   // 每次向量计算的tile大小
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ReduceMaxCustom, ReduceMaxCustomTilingData)
}
