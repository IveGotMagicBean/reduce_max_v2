#include <iostream>
#include <vector>
#include <chrono>
#include "acl/acl.h"
#include "aclnn_reduce_max_custom.h"
using namespace std;

void testShape3D(aclrtStream stream,
                 int64_t M, int64_t K, int64_t N,
                 int64_t reduceDim, const char* label)
{
    int64_t totalIn  = M * K * N;
    int64_t totalOut = (reduceDim == 0) ? K*N :
                       (reduceDim == 1) ? M*N : M*K;

    // 检查内存是否合理（超过4GB跳过）
    if (totalIn * 2 > 4LL * 1024 * 1024 * 1024) {
        cout << label << " SKIP (too large: " << totalIn*2/1024/1024 << " MB)" << endl;
        return;
    }

    void *devX, *devY, *devIdx;
    aclrtMalloc(&devX,   totalIn  * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY,   totalOut * sizeof(uint16_t) + 32, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devIdx, totalOut * sizeof(int32_t)  + 32, ACL_MEM_MALLOC_HUGE_FIRST);

    int64_t inShape[3]   = {M, K, N};
    int64_t outShape2D_0[2] = {K, N};  // reduce dim0
    int64_t outShape2D_1[2] = {M, N};  // reduce dim1
    int64_t outShape2D_2[2] = {M, K};  // reduce dim2

    int64_t* outShape = (reduceDim == 0) ? outShape2D_0 :
                        (reduceDim == 1) ? outShape2D_1 : outShape2D_2;

    aclTensor* xT = aclCreateTensor(inShape,   3, ACL_FLOAT16, nullptr, 0,
                                     ACL_FORMAT_ND, inShape,   3, devX);
    aclTensor* yT = aclCreateTensor(outShape,  2, ACL_FLOAT16, nullptr, 0,
                                     ACL_FORMAT_ND, outShape,  2, devY);
    aclTensor* idxT = aclCreateTensor(outShape, 2, ACL_INT32,   nullptr, 0,
                                       ACL_FORMAT_ND, outShape,  2, devIdx);

    int64_t isKeepDim = 0;
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnReduceMaxCustomGetWorkspaceSize(
        xT, reduceDim, isKeepDim, yT, idxT, &wsSize, &exec);
    if (ret != 0) {
        cout << label << " GetWorkspaceSize FAILED ret=" << ret << endl;
        goto cleanup;
    }

    {
        void* ws = nullptr;
        if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

        // 预热
        aclnnReduceMaxCustom(ws, wsSize, exec, stream);
        aclrtSynchronizeStream(stream);

        const int RUNS = 20;
        auto t0 = chrono::high_resolution_clock::now();
        for (int i = 0; i < RUNS; i++) {
            aclnnReduceMaxCustomGetWorkspaceSize(
                xT, reduceDim, isKeepDim, yT, idxT, &wsSize, &exec);
            aclnnReduceMaxCustom(ws, wsSize, exec, stream);
        }
        aclrtSynchronizeStream(stream);
        auto t1 = chrono::high_resolution_clock::now();

        double ms = chrono::duration<double, milli>(t1 - t0).count() / RUNS;
        double gb = (double)totalIn * 2 / 1024 / 1024 / 1024;
        cout << label
             << " shape=[" << M << "," << K << "," << N << "] dim=" << reduceDim
             << " in=" << totalIn*2/1024/1024 << "MB"
             << " avg=" << ms << "ms"
             << " BW=" << gb/ms*1000 << " GB/s" << endl;

        if (ws) aclrtFree(ws);
    }

cleanup:
    aclDestroyTensor(xT); aclDestroyTensor(yT); aclDestroyTensor(idxT);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devIdx);
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    cout << "=== 赛题 Shape 性能测试 ===" << endl;

    // S3: (1024, 512, 1024) 中维
    testShape3D(stream, 1024, 512, 1024, 2, "[S3] dim=2");
    testShape3D(stream, 1024, 512, 1024, 1, "[S3] dim=1");
    testShape3D(stream, 1024, 512, 1024, 0, "[S3] dim=0");

    // S4: (512, 128, 512) 低维
    testShape3D(stream, 512, 128, 512, 2, "[S4] dim=2");
    testShape3D(stream, 512, 128, 512, 1, "[S4] dim=1");

    // S1: (2048, 2048, 2048) 高维 — 内存约 16GB，可能装不下
    testShape3D(stream, 2048, 2048, 2048, 2, "[S1] dim=2");

    // S2: (4096, 1024, 4096) 高维
    testShape3D(stream, 4096, 1024, 4096, 2, "[S2] dim=2");

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
