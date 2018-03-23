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
 * \file sockeye_beam_search.h
 * \brief Operator performing state transition for beam search in Sockeye.
 */
#ifndef MXNET_OPERATOR_TENSOR_SOCKEYE_BEAM_SEARCH_H_
#define MXNET_OPERATOR_TENSOR_SOCKEYE_BEAM_SEARCH_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

namespace beam {
enum BeamOpInputs { kseq, klen, kfin, kvoc, kscore, kscore_acc, katt, katt_score, khyp };
}

struct SockeyeBeamSearchParam : public dmlc::Parameter<SockeyeBeamSearchParam> {
  int batch_size;
  int beam_size;
  int pad_id;
  int eos_id;
  int step;
  bool restrict_vocab;
  float alpha;
  float beta;
  DMLC_DECLARE_PARAMETER(SockeyeBeamSearchParam) {
    DMLC_DECLARE_FIELD(batch_size)
      .set_default(1)
      .describe("Batch size used in beam search.");
    DMLC_DECLARE_FIELD(beam_size)
      .set_default(1)
      .describe("Beam size used in beam search.");
    DMLC_DECLARE_FIELD(pad_id)
      .set_default(0)
      .describe("C.PAD_ID in Sockeye.");
    DMLC_DECLARE_FIELD(eos_id)
      .set_default(0)
      .describe("vocab_target[C.EOS_SYMBOL] in Sockeye.");
    DMLC_DECLARE_FIELD(step)
      .set_default(0)
      .describe("Decoding step");
    DMLC_DECLARE_FIELD(restrict_vocab)
      .set_default(false)
      .describe("Whether to restrict target vocabulary.");
    DMLC_DECLARE_FIELD(alpha)
      .set_default(1.0)
      .describe("Alpha parameter of length penalty.");
    DMLC_DECLARE_FIELD(beta)
      .set_default(0.0)
      .describe("Beta parameter of length penalty.");
  }
};

inline bool SockeyeBeamSearchType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int> *in_type,
                                  std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 9);
  CHECK_EQ(out_type->size(), 1);
  std::for_each(in_type->begin(), in_type->end(), [](int t) { CHECK_NE(t, -1); });
  CHECK_EQ((*in_type)[beam::kscore], (*in_type)[beam::kscore_acc]);
  CHECK_EQ((*in_type)[beam::kscore], (*in_type)[beam::klen]);
  CHECK_EQ((*in_type)[beam::kscore], (*in_type)[beam::katt]);
  CHECK_EQ((*in_type)[beam::kscore], (*in_type)[beam::katt_score]);
  CHECK_EQ((*in_type)[beam::kscore], (*in_type)[beam::kvoc]);
  CHECK_EQ((*in_type)[beam::kfin], mshadow::kInt32);
  CHECK_EQ((*in_type)[beam::kseq], mshadow::kInt32);
  CHECK_EQ((*in_type)[beam::khyp], mshadow::kInt32);
  TYPE_ASSIGN_CHECK(*out_type, 0, mshadow::kInt32);
  return true;
}


inline bool SockeyeBeamSearchShape(const nnvm::NodeAttrs& attrs,
                                   std::vector<TShape>* in_attrs,
                                   std::vector<TShape>* out_attrs) {
  using namespace mshadow;
  auto parms(nnvm::get<SockeyeBeamSearchParam>(attrs.parsed));
  const int beam_size(parms.beam_size), batch_size(parms.batch_size);
  const int M(beam_size*batch_size);

  CHECK_EQ(in_attrs->size(), 9);
  CHECK_EQ(out_attrs->size(), 1);
  CHECK_EQ((*in_attrs)[beam::katt].ndim(), 3);
  CHECK_EQ((*in_attrs)[beam::katt_score].ndim(), 2);

  // Infer some values. 
  const int max_output_length((*in_attrs)[beam::katt][1]);
  const int encoded_source_length((*in_attrs)[beam::katt][2]);
  CHECK_GT(max_output_length, 0);
  CHECK_GT(encoded_source_length, 0);
  CHECK_EQ((*in_attrs)[beam::khyp].Size(), M);
  CHECK_EQ((*in_attrs)[beam::klen].Size(), M);
  CHECK_EQ((*in_attrs)[beam::kfin].Size(), M);
  CHECK_EQ((*in_attrs)[beam::kscore][0], M);
  CHECK_EQ((*in_attrs)[beam::kscore_acc].Size(), M);
  CHECK_EQ((*in_attrs)[beam::kseq][0], M);
  CHECK_EQ((*in_attrs)[beam::katt][0], M);
  CHECK_EQ((*in_attrs)[beam::katt_score][0], M);
  CHECK_EQ((*in_attrs)[beam::katt_score][1], encoded_source_length);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape1(1));
  return true;
}

// In-place take operation for rows of a matrix. Targeted for small number of
// rows. Huge number of rows would require inverse indexing as well. 
template<typename DType>
void TakeRows(DType *matrix, int *rows, int M, int N) {
  std::vector<int> contains(M, -1);
  std::for_each(rows, rows+M, [&](int& i) { contains[i] = i; });
  for (int i = 0; i < M; ++i ) {
    // Row that we want to take here. 
    const int row(std::find(contains.begin(), contains.end(), rows[i])-contains.begin());
    CHECK_LT(row, M);
    if (row < i ) {
      const int free(std::find(contains.begin()+i, contains.end(), -1)-contains.begin());
      CHECK_LT(free, M);
      std::copy(matrix+i*N, matrix+(i+1)*N, matrix+free*N);
      std::copy(matrix+row*N, matrix+(row+1)*N, matrix+i*N);
      contains[free] = contains[i];
    } else {
      std::swap_ranges(matrix+i*N, matrix+(i+1)*N, matrix+row*N);
      contains[row] = contains[i];
    }
    contains[i] = rows[i];
  }
}

template<typename DType>
void SockeyeBeamSearchForwardCpu(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow;
  using namespace mxnet_op;

  auto parms(nnvm::get<SockeyeBeamSearchParam>(attrs.parsed));
  const int beam_size(parms.beam_size), batch_size(parms.batch_size);
  const int M(beam_size*batch_size);

  // Infer some values. 
  const int max_output_length(inputs[beam::katt].shape_[1]);
  const int encoded_source_length(inputs[beam::katt].shape_[2]);
 
  // Use threading to speed up
  const int omp_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount());

  DType *scores     = inputs[beam::kscore].dptr<DType>();
  DType *scores_acc = inputs[beam::kscore_acc].dptr<DType>();
  DType *lengths    = inputs[beam::klen].dptr<DType>();
  DType *att        = inputs[beam::katt].dptr<DType>();
  DType *att_scores = inputs[beam::katt_score].dptr<DType>();
  DType *vocab_id   = inputs[beam::kvoc].dptr<DType>();
  int   *finished   = inputs[beam::kfin].dptr<int>();
  int   *best_hyp   = inputs[beam::khyp].dptr<int>();
  int   *sequences  = inputs[beam::kseq].dptr<int>();
    
  // # (2) compute length-normalized accumulated scores in place
  // if t == 1 and self.batch_size == 1:  # only one hypothesis at t==1
  //     scores = scores[:1] / self.length_penalty(lengths[:1])
  // else:
  //     # renormalize scores by length ...
  //     scores = (scores + scores_accumulated * self.length_penalty(lengths - 1)) / self.length_penalty(lengths)
  //     # ... but not for finished hyps.
  //     # their predicted distribution is set to their accumulated scores at C.PAD_ID.
  //     pad_dist[:, C.PAD_ID] = scores_accumulated[:, 0]
  //     # this is equivalent to doing this in numpy:
  //     #   pad_dist[finished, :] = np.inf
  //     #   pad_dist[finished, C.PAD_ID] = scores_accumulated[finished]
  //     scores = mx.nd.where(finished, pad_dist, scores)
  auto length_penalty = [&parms](int length) {
    return pow((parms.beta + length)/(parms.beta+1), parms.alpha);
  };
  DType infty(red::limits::MaxValue<DType>());
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < M; ++i) {
    int N(inputs[beam::kscore].shape_[1]);
    if (finished[i]) {    
      std::fill(scores+i*N, scores+(i+1)*N, infty);
      scores[i*N+parms.pad_id] = scores_acc[i];
    } else {
      DType lp1(length_penalty(lengths[i]-1)), lp2(length_penalty(lengths[i]));
      std::for_each(scores+i*N, scores+(i+1)*N, [&](DType &val){ val = (val + scores_acc[i]*lp1)/lp2; });
    }
  }

  // # (3) get beam_size winning hypotheses for each sentence block separately
  // scores = scores.asnumpy()  # convert to numpy once to minimize cross-device copying
  // for sent in range(self.batch_size):
  //     rows = slice(sent * self.beam_size, (sent + 1) * self.beam_size)
  //     sliced_scores = scores if t == 1 and self.batch_size == 1 else scores[rows]
  //     # TODO we could save some tiny amount of time here by not running smallest_k for a finished sent
  //     (best_hyp_indices_np[rows], best_word_indices_np[rows]), 
  //         scores_accumulated_np[rows] = utils.smallest_k(sliced_scores, self.beam_size, t == 1)
  //     # offsetting since the returned smallest_k() indices were slice-relative
  //     best_hyp_indices_np[rows] += rows.start
  // # convert back to mx.ndarray again
  // best_hyp_indices[:] = best_hyp_indices_np
  // best_word_indices[:] = best_word_indices_np
  // scores_accumulated[:] = np.expand_dims(scores_accumulated_np, axis=1)
  // # Map from restricted to full vocab ids if needed
  // if self.restrict_lexicon:
  //     best_word_indices[:] = vocab_slice_ids.take(best_word_indices)
  std::vector<int> best_word_indices(M);
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < M; i += beam_size) {
    int N(inputs[beam::kscore].shape_[1]);
    // Examine only first row for t==1. 
    const int first(i*N), last(parms.step == 1 ? (i+1)*N : (i+beam_size)*N);
    // Find smallest k elements in sorted order.
    std::vector<int> tmp(last-first);
    std::iota(tmp.begin(), tmp.end(), first);
    std::partial_sort(tmp.begin(), tmp.begin()+beam_size, tmp.end(),
                      [&](const int& i1, const int& i2){ return scores[i1] < scores[i2]; });
    // Select associated rows, cols, scores.
    for (int j = 0; j < beam_size; ++j ) {
      scores_acc[i+j] = scores[tmp[j]]; 
      best_hyp[i+j] = tmp[j]/N;
      best_word_indices[i+j] = tmp[j]%N;
      if (parms.restrict_vocab) {
        best_word_indices[i+j] = int(vocab_id[best_word_indices[i+j]]);
      }
    } 
  }
  
  // # (4) get hypotheses and their properties for beam_size winning hypotheses (ascending) 
  #pragma omp parallel sections num_threads(omp_threads)
  { 
    #pragma omp section
    {
      // sequences = mx.nd.take(sequences, best_hyp_indices)
      TakeRows(sequences, best_hyp, M, inputs[beam::kseq].shape_.Size()/M);
    }
    #pragma omp section
    {
      // lengths = mx.nd.take(lengths, best_hyp_indices)  
      TakeRows(lengths, best_hyp, M, inputs[beam::klen].shape_.Size()/M);
    }
    #pragma omp section
    {
      // finished = mx.nd.take(finished, best_hyp_indices)
      TakeRows(finished, best_hyp, M, inputs[beam::kfin].shape_.Size()/M);
    }
    #pragma omp section
    {
      // attentions = mx.nd.take(attentions, best_hyp_indices)
      TakeRows(att, best_hyp, M, inputs[beam::katt].shape_.Size()/M);
    }
    #pragma omp section
    {
      // attention_scores = mx.nd.take(attention_scores, best_hyp_indices)
      TakeRows(att_scores, best_hyp, M, inputs[beam::katt_score].shape_.Size()/M);
    }
  }

  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < M; ++i) {
    const int t(parms.step);
    // # (5) update best hypotheses, their attention lists and lengths (only for non-finished hyps)
    // sequences[:, t] = best_word_indices
    sequences[i*max_output_length+t] = best_word_indices[i];
    // attentions[:, t, :] = attention_scores
    std::copy(att_scores+i*encoded_source_length, att_scores+(i+1)*encoded_source_length, 
              att+(i*max_output_length+t)*encoded_source_length);
    //lengths += mx.nd.cast(1 - mx.nd.expand_dims(finished, axis=1), dtype='float32')
    lengths[i] += 1.0 - finished[i];
    // # (6) determine which hypotheses in the beam are now finished
    // finished = ((best_word_indices == C.PAD_ID) + (best_word_indices == self.vocab_target[C.EOS_SYMBOL]))
    finished[i] = ((best_word_indices[i] == parms.pad_id) || (best_word_indices[i] == parms.eos_id) ? 1 : 0);
  }
  const bool done(std::find(finished, finished+M, 0) == finished+M);
  (*outputs[0].dptr<int>()) = done;
}

template<typename xpu>
void SockeyeBeamSearchForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {}

template<>
void SockeyeBeamSearchForward<cpu>(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  // Factor out code into a separate function as MSHADOW_REAL_TYPE_SWITCH
  // and #pragma mess up together.
  MSHADOW_REAL_TYPE_SWITCH(inputs[beam::kscore].type_flag_, DType, {
    SockeyeBeamSearchForwardCpu<DType>(attrs, ctx, inputs, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SOCKEYE_BEAM_SEARCH_H_
