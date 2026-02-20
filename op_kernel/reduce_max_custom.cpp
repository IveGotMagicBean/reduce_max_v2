#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void reduce_max_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tiling_data, tiling);
    
    // 只在核0上工作
    if (GetBlockIdx() != 0) {
        return;
    }
    
    // 使用GlobalTensor访问全局内存
    GlobalTensor<half> inputGM;
    GlobalTensor<half> outputGM;
    GlobalTensor<int32_t> idxGM;
    
    inputGM.SetGlobalBuffer((__gm__ half*)x, tiling_data.size);
    outputGM.SetGlobalBuffer((__gm__ half*)y, 1);
    idxGM.SetGlobalBuffer((__gm__ int32_t*)idx, 1);
    
    // 找最大值 - 转为float比较
    half max_val_h = inputGM.GetValue(0);
    float max_val = static_cast<float>(max_val_h);
    int32_t max_idx = 0;
    
    for (uint32_t i = 1; i < tiling_data.size; i++) {
        half val_h = inputGM.GetValue(i);
        float val = static_cast<float>(val_h);
        if (val > max_val) {
            max_val = val;
            max_val_h = val_h;
            max_idx = i;
        }
    }
    
    // 写回结果
    outputGM.SetValue(0, max_val_h);
    idxGM.SetValue(0, max_idx);
}
