#include "reduce_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ReduceMaxCustomTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    auto& shape = x_shape->GetStorageShape();
    int32_t ndim = shape.GetDimNum();

    // 获取 reduceDim 属性
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    int64_t reduceDim = *attrs->GetAttrPointer<int64_t>(0);
    if (reduceDim < 0) reduceDim += ndim;

    // 计算 outer / inner / stride
    uint32_t outer = 1, inner = 1, stride = 1;
    for (int i = 0; i < (int)reduceDim; i++)        outer  *= shape.GetDim(i);
    inner = shape.GetDim(reduceDim);
    for (int i = (int)reduceDim + 1; i < ndim; i++) stride *= shape.GetDim(i);

    uint32_t totalOut = outer * stride;

    // 动态决定核数：最多20核，但不超过输出元素数
    uint32_t blockNum = 20;
    if (totalOut < blockNum) blockNum = totalOut;
    if (blockNum == 0) blockNum = 1;

    // tileSize：UB一次能装多少元素，256对齐
    // UB约192KB，half=2B，留一半给work buffer
    // 192*1024/2/2 = 49152，保守取8192
    uint32_t tileSize = 256;
    if (inner >= 1024) tileSize = 1024;
    if (inner >= 4096) tileSize = 4096;
    if (inner >= 8192) tileSize = 8192;
    // 对齐到16（half的向量对齐要求）
    tileSize = (tileSize / 16) * 16;
    if (tileSize == 0) tileSize = 16;

    tiling.set_outer(outer);
    tiling.set_inner(inner);
    tiling.set_stride(stride);
    tiling.set_blockNum(blockNum);
    tiling.set_tileSize(tileSize);

    context->SetBlockDim(blockNum);

    // workspace：用于核间汇总（每核写一个half+一个int32）
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = blockNum * (sizeof(uint16_t) + sizeof(int32_t));

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape   = context->GetOutputShape(0);
    gert::Shape* idx_shape = context->GetOutputShape(1);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    int64_t reduceDim = *attrs->GetAttrPointer<int64_t>(0);
    int32_t ndim = x_shape->GetDimNum();
    if (reduceDim < 0) reduceDim += ndim;

    int32_t out_ndim = 0;
    for (int i = 0; i < ndim; i++) {
        if (i == (int)reduceDim) continue;
        y_shape->SetDim(out_ndim, x_shape->GetDim(i));
        idx_shape->SetDim(out_ndim, x_shape->GetDim(i));
        out_ndim++;
    }
    if (out_ndim == 0) {
        out_ndim = 1;
        y_shape->SetDim(0, 1);
        idx_shape->SetDim(0, 1);
    }
    y_shape->SetDimNum(out_ndim);
    idx_shape->SetDimNum(out_ndim);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT16);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class ReduceMaxCustom : public OpDef {
public:
    explicit ReduceMaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("reduceDim").Int();
        this->Attr("isKeepDim").AttrType(OPTIONAL).Int(1);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ReduceMaxCustom);
} // namespace ops
