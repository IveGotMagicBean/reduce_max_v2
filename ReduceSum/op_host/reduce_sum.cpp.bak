/**
* @file reduce_sum.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "reduce_sum_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ReduceSumTilingData tiling;

        auto shape_x = context->GetInputShape(0)->GetOriginShape();
        auto shape_y = context->GetOutputShape(0)->GetOriginShape();
        int32_t axes_num = context->GetInputShape(1)->GetStorageShape().GetShapeSize();

        int32_t x_ndarray[20];
        int32_t x_dimensional;
        int32_t size = 1;

        x_dimensional = shape_x.GetDimNum();

        for (int i = 0; i < x_dimensional; i++)
        {
            x_ndarray[i] = shape_x.GetDim(i);
            size *= x_ndarray[i];
        }

        bool keep_dims = *context->GetAttrs()->GetBool(0);
        bool ignore_nan = *context->GetAttrs()->GetBool(1);
        const char *dtype_str = context->GetAttrs()->GetStr(2);

        uint32_t ySize = context->GetOutputShape(0)->GetStorageShape().GetShapeSize();

        constexpr int SET_TILING_KEY = 1;
        constexpr int QUERY_TILING_KEY = 2;
        if (axes_num == 1 || ySize == 1)
        {
            context->SetTilingKey(SET_TILING_KEY);
        }
        else
        {
            context->SetTilingKey(QUERY_TILING_KEY);
        }

        tiling.set_size(size);
        tiling.set_x_ndarray(x_ndarray);
        tiling.set_x_dimensional(x_dimensional);
        tiling.set_keep_dims(keep_dims);
        tiling.set_ignore_nan(ignore_nan);
        tiling.set_axes_num(axes_num);

        context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class ReduceSum : public OpDef
    {
    public:
        explicit ReduceSum(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("axes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
            this->Attr("ignore_nan").AttrType(OPTIONAL).Bool(false);
            this->Attr("dtype").AttrType(OPTIONAL).String("float");

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310b");
        }
    };

    OP_ADD(ReduceSum);
}
