#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_reduce_max_custom.h"
using namespace std;

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    const size_t N = 1024 * 1024;
    vector<uint16_t> hostInput(N);
    // 填充测试数据
    for (size_t i = 0; i < N; i++) hostInput[i] = (uint16_t)(i % 65504);

    void *devX, *devY, *devIdx;
    aclrtMalloc(&devX,   N * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY,   16 * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devIdx, 16 * sizeof(int32_t),  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devX, N * sizeof(uint16_t), hostInput.data(),
                N * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);

    int64_t inShape[1]  = {(int64_t)N};
    int64_t outShape[1] = {1};
    int64_t reduceDim   = 0;
    int64_t isKeepDim   = 0;

    aclTensor* xTensor   = aclCreateTensor(inShape,  1, ACL_FLOAT16, nullptr, 0,
                                            ACL_FORMAT_ND, inShape,  1, devX);
    aclTensor* yTensor   = aclCreateTensor(outShape, 1, ACL_FLOAT16, nullptr, 0,
                                            ACL_FORMAT_ND, outShape, 1, devY);
    aclTensor* idxTensor = aclCreateTensor(outShape, 1, ACL_INT32,   nullptr, 0,
                                            ACL_FORMAT_ND, outShape, 1, devIdx);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnReduceMaxCustomGetWorkspaceSize(xTensor, reduceDim, isKeepDim,
                                         yTensor, idxTensor,
                                         &workspaceSize, &executor);
    void* workspace = nullptr;
    if (workspaceSize > 0)
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 预热1次
    aclnnReduceMaxCustom(workspace, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);

    // 正式计时：跑100次
    const int RUNS = 100;
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) {
        aclnnReduceMaxCustomGetWorkspaceSize(xTensor, reduceDim, isKeepDim,
                                             yTensor, idxTensor,
                                             &workspaceSize, &executor);
        aclnnReduceMaxCustom(workspace, workspaceSize, executor, stream);
    }
    aclrtSynchronizeStream(stream);
    auto t1 = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, milli>(t1 - t0).count();
    cout << "[PERF] " << RUNS << " runs, total=" << ms << "ms"
         << ", avg=" << ms/RUNS << "ms per run" << endl;
    cout << "[PERF] Input size: " << N << " elements ("
         << N*2/1024/1024 << " MB)" << endl;

    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(idxTensor);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devIdx);
    if (workspace) aclrtFree(workspace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
