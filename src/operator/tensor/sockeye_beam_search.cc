/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file sockeye_beam_search.cc
 * \brief CPU-Operators for Sockeye beam search.
 */
#include "./sockeye_beam_search.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SockeyeBeamSearchParam);

NNVM_REGISTER_OP(_sockeye_beam_search)
.add_alias("sockeye_beam_search")
.describe(R"code(Performs state update in beam search.
)code" ADD_FILELINE)
.set_num_inputs(9)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SockeyeBeamSearchParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"SEQ", "LEN", "FIN", "VOC", "SCORE", "SCORE_ACC", "ATT", "ATT_SCORE", "HYP"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", SockeyeBeamSearchShape)
.set_attr<nnvm::FInferType>("FInferType", SockeyeBeamSearchType)
.set_attr<FCompute>("FCompute<cpu>", SockeyeBeamSearchForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("SEQ", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("LEN", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("FIN", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("VOC", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("SCORE", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("SCORE_ACC", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("ATT", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("ATT_SCORE", "NDArray-or-Symbol", "Tensor of input matrices")
.add_argument("HYP", "NDArray-or-Symbol", "Tensor of input matrices")
.add_arguments(SockeyeBeamSearchParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
