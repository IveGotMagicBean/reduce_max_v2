#include <iostream>
#include <vector>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_reduce_max_custom.h"
using namespace std;

void testShape(aclrtStream stream, int64_t N, const char* label) {
    void *devX, *devY, *devIdx;
    aclrtMalloc(&devX,   N * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY,   16 * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devIdx, 16 * sizeof(int32_t),  ACL_MEM_MALLOC_HUGE_FIRST);

    int64_t inShape[1]  = {N};
    int64_t outShape[1] = {1};
    int64_t reduceDim = 0, isKeepDim = 0;

    aclTensor* xT   = aclCreateTensor(inShape,  1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, inShape,  1, devX);
    aclTensor* yT   = aclCreateTensor(outShape, 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, outShape, 1, devY);
    aclTensor* idxT = aclCreateTensor(outShape, 1, ACL_INT32,   nullptr, 0, ACL_FORMAT_ND, outShape, 1, devIdx);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnReduceMaxCustomGetWorkspaceSize(xT, reduceDim, isKeepDim, yT, idxT, &wsSize, &exec);
    void* ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 预热
    aclnnReduceMaxCustom(ws, wsSize, exec, stream);
    aclrtSynchronizeStream(stream);

    // 计时100次
    const int RUNS = 100;
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) {
        aclnnReduceMaxCustomGetWorkspaceSize(xT, reduceDim, isKeepDim, yT, idxT, &wsSize, &exec);
        aclnnReduceMaxCustom(ws, wsSize, exec, stream);
    }
    aclrtSynchronizeStream(stream);
    auto t1 = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, milli>(t1 - t0).count() / RUNS;
    double gb = (double)N * 2 / 1024 / 1024 / 1024;
    cout << label << " N=" << N
         << " avg=" << ms << "ms"
         << " BW=" << gb/ms*1000 << " GB/s" << endl;

    aclDestroyTensor(xT); aclDestroyTensor(yT); aclDestroyTensor(idxT);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devIdx);
    if (ws) aclrtFree(ws);
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 模拟赛题 Shape [M,K,N] flatten 后的 inner 维度大小
    testShape(stream, 2048L,              "[small]   2048");
    testShape(stream, 2048L*2048,         "[mid]     2048x2048");
    testShape(stream, 2048L*2048*2048,    "[S1]      2048^3 flatten");
    testShape(stream, 4096L*1024*4096,    "[S2]      4096x1024x4096 flatten");

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
