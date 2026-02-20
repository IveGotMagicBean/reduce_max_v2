#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ReduceMaxCustom)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(idx, ge::TensorType::ALL())
    .REQUIRED_ATTR(reduceDim, Int)
    .ATTR(isKeepDim, Int, 1)
    .OP_END_FACTORY_REG(ReduceMaxCustom);

}

#endif
