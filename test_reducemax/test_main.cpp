#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_reduce_max_custom.h"
using namespace std;

// uint16_t (float16) 转 float
float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    if (exp == 0) return (sign ? -1 : 1) * ldexp((float)frac, -24);
    if (exp == 31) return frac ? NAN : (sign ? -INFINITY : INFINITY);
    return (sign ? -1 : 1) * ldexp((float)(frac + 1024), (int)exp - 25);
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    cout << "[INFO] ACL initialized" << endl;

    const size_t N = 1024 * 1024;
    vector<uint16_t> hostX(N);
    ifstream fin("./input/input_x.bin", ios::binary);
    fin.read((char*)hostX.data(), N * 2);

    void *devX, *devY, *devIdx;
    aclrtMalloc(&devX,   N * 2,  ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY,   16 * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devIdx, 16 * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devX, N*2, hostX.data(), N*2, ACL_MEMCPY_HOST_TO_DEVICE);
    cout << "[INFO] Input copied to device" << endl;

    int64_t inShape[]  = {(int64_t)N};
    int64_t outShape[] = {1};
    int64_t reduceDim = 0, isKeepDim = 0;

    aclTensor* xT = aclCreateTensor(inShape,  1, ACL_FLOAT16, nullptr, 0,
                                     ACL_FORMAT_ND, inShape,  1, devX);
    aclTensor* yT = aclCreateTensor(outShape, 1, ACL_FLOAT16, nullptr, 0,
                                     ACL_FORMAT_ND, outShape, 1, devY);
    aclTensor* iT = aclCreateTensor(outShape, 1, ACL_INT32,   nullptr, 0,
                                     ACL_FORMAT_ND, outShape, 1, devIdx);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnReduceMaxCustomGetWorkspaceSize(xT, reduceDim, isKeepDim, yT, iT, &wsSize, &exec);
    cout << "[INFO] Workspace size: " << wsSize << endl;

    void* ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclnnReduceMaxCustom(ws, wsSize, exec, stream);
    aclrtSynchronizeStream(stream);
    cout << "[INFO] Operator executed!" << endl;

    uint16_t hostY = 0; int32_t hostIdx = 0;
    aclrtMemcpy(&hostY,   2, devY,   2, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(&hostIdx, 4, devIdx, 4, ACL_MEMCPY_DEVICE_TO_HOST);

    uint16_t goldenY = 0; int32_t goldenIdx = 0;
    ifstream fy("./output/golden_y.bin",   ios::binary); fy.read((char*)&goldenY,   2);
    ifstream fi("./output/golden_idx.bin", ios::binary); fi.read((char*)&goldenIdx, 4);

    float outVal    = f16_to_f32(hostY);
    float goldenVal = f16_to_f32(goldenY);
    bool valOK = fabs(outVal - goldenVal) < 0.1f;
    bool idxOK = (hostIdx == goldenIdx);

    cout << "[RESULT] max value: got=" << outVal << " golden=" << goldenVal
         << (valOK ? " PASS" : " FAIL") << endl;
    cout << "[RESULT] max index: got=" << hostIdx << " golden=" << goldenIdx
         << (idxOK ? " PASS" : " FAIL") << endl;
    cout << "[RESULT] " << (valOK && idxOK ? "ALL PASS" : "FAIL") << endl;

    aclDestroyTensor(xT); aclDestroyTensor(yT); aclDestroyTensor(iT);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devIdx);
    if (ws) aclrtFree(ws);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return (valOK && idxOK) ? 0 : 1;
}
