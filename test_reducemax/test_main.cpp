#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include "acl/acl.h"
#include "aclnn_reduce_max_custom.h"
using namespace std;

float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    if (exp == 0)  return (sign ? -1 : 1) * ldexp((float)frac, -24);
    if (exp == 31) return frac ? NAN : (sign ? -INFINITY : INFINITY);
    return (sign ? -1 : 1) * ldexp((float)(frac + 1024), (int)exp - 25);
}

bool run_test(const vector<int64_t>& inShape, const vector<int64_t>& outShape,
              int64_t reduceDim, const string& tag)
{
    cout << "\n===== [" << tag << "] =====" << endl;

    // 读输入
    size_t inSize = 1;
    for (auto d : inShape) inSize *= d;
    vector<uint16_t> hostX(inSize);
    ifstream fin("../input_" + tag + "/input_x.bin", ios::binary);
    if (!fin) { cout << "SKIP: input not found" << endl; return true; }
    fin.read((char*)hostX.data(), inSize * 2);

    // 读 golden
    size_t outSize = 1;
    for (auto d : outShape) outSize *= d;
    vector<uint16_t> goldenY(outSize);
    vector<int32_t>  goldenIdx(outSize);
    ifstream fy("../output_" + tag + "/golden_y.bin",   ios::binary);
    ifstream fi("../output_" + tag + "/golden_idx.bin", ios::binary);
    fy.read((char*)goldenY.data(),   outSize * 2);
    fi.read((char*)goldenIdx.data(), outSize * 4);

    // 分配设备内存
    void *devX, *devY, *devIdx;
    aclrtMalloc(&devX,   inSize  * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devY,   outSize * 2 * 16, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devIdx, outSize * 4 * 16, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devX, inSize*2, hostX.data(), inSize*2, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    int64_t isKeepDim = 0;
    aclTensor* xT = aclCreateTensor(inShape.data(),  inShape.size(),  ACL_FLOAT16, nullptr, 0,
                                     ACL_FORMAT_ND,   inShape.data(),  inShape.size(),  devX);
    aclTensor* yT = aclCreateTensor(outShape.data(), outShape.size(), ACL_FLOAT16, nullptr, 0,
                                     ACL_FORMAT_ND,   outShape.data(), outShape.size(), devY);
    aclTensor* iT = aclCreateTensor(outShape.data(), outShape.size(), ACL_INT32,   nullptr, 0,
                                     ACL_FORMAT_ND,   outShape.data(), outShape.size(), devIdx);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    aclnnReduceMaxCustomGetWorkspaceSize(xT, reduceDim, isKeepDim, yT, iT, &wsSize, &exec);
    void* ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclnnReduceMaxCustom(ws, wsSize, exec, stream);
    aclrtSynchronizeStream(stream);

    // 拷回结果
    vector<uint16_t> hostY(outSize * 16);
    vector<int32_t>  hostIdx(outSize * 16);
    aclrtMemcpy(hostY.data(),   outSize*2*16, devY,   outSize*2*16, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hostIdx.data(), outSize*4*16, devIdx, outSize*4*16, ACL_MEMCPY_DEVICE_TO_HOST);

    // 逐元素比对
    int valFail = 0, idxFail = 0;
    float maxDiff = 0;
    for (size_t i = 0; i < outSize; i++) {
        float got    = f16_to_f32(hostY[i * 16]);
        float golden = f16_to_f32(goldenY[i]);
        float diff   = fabs(got - golden);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > 0.1f) {
            if (valFail < 3)
                cout << "  val FAIL at [" << i << "]: got=" << got << " golden=" << golden << endl;
            valFail++;
        }
        if (hostIdx[i * 16] != goldenIdx[i]) {
            if (idxFail < 3)
                cout << "  idx FAIL at [" << i << "]: got=" << hostIdx[i] << " golden=" << goldenIdx[i] << endl;
            idxFail++;
        }
    }

    cout << "[RESULT] 输出元素数=" << outSize << endl;
    cout << "[RESULT] 最大误差=" << maxDiff << endl;
    cout << "[RESULT] 值错误数=" << valFail << "/" << outSize << endl;
    cout << "[RESULT] 索引错误数=" << idxFail << "/" << outSize << endl;
    cout << "[RESULT] 精度结论: " << (valFail==0 && idxFail==0 ? "✓ PASS" : "✗ FAIL") << endl;

    aclDestroyTensor(xT); aclDestroyTensor(yT); aclDestroyTensor(iT);
    aclrtFree(devX); aclrtFree(devY); aclrtFree(devIdx);
    if (ws) aclrtFree(ws);
    aclrtDestroyStream(stream);
    return (valFail == 0 && idxFail == 0);
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    bool all = true;
    all &= run_test({1048576},         {1},        0, "1048576_dim0");
    all &= run_test({512, 128, 512},   {512, 512}, 1, "512x128x512_dim1");
    all &= run_test({512, 128, 512},   {512, 128}, 2, "512x128x512_dim2");
    all &= run_test({512, 4096},       {512},      1, "512x4096_dim1");
    all &= run_test({512, 4096},       {4096},     0, "512x4096_dim0");

    cout << "\n===== TOTAL: " << (all ? "ALL PASS" : "SOME FAILED") << " =====" << endl;

    aclrtResetDevice(0);
    aclFinalize();
    return all ? 0 : 1;
}
