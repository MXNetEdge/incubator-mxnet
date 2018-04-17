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
  //const int omp_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount());
  // FIXIT: Use fixed minimum number of threads as engine will restrict it to 1 on GPU
  const int omp_threads(std::max(2, engine::OpenMP::Get()->GetRecommendedOMPThreadCount()));

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
 //printf("REAL TOPK i = %d, val = %d / %f\n",j, tmp[j], scores[tmp[j]]);
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
MSHADOW_FORCE_INLINE void SockeyeBeamSearchForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {}

template<>
MSHADOW_FORCE_INLINE void SockeyeBeamSearchForward<cpu>(const nnvm::NodeAttrs& attrs,
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

#ifdef __CUDACC__

template<typename DType>
MSHADOW_XINLINE void MergeTopK(int K, DType *val1, int *ind1, DType *val2, int *ind2) {
  // In-place merge into val1/ind1
  int i1(K-1), i2(K-1);
  for( int i = 0; i < K; ++i ) {
    if( val1[i1] < val2[i2] ) {
      --i2;
    } else { 
      --i1;
    }
  }
  for( int i = K; i--; ) {
    if( i2 < 0 || i1 >= 0 && val1[i1] > val2[i2] ) {
      val1[i] = val1[i1];
      ind1[i] = ind1[i1];
      --i1;
    } else {
      val1[i] = val2[i2];
      ind1[i] = ind2[i2];
      --i2;
    }
  }
}

template<typename DType>
__global__ void find_best_hyp(int t, int N, int beam_size, int pad_id, int restrict_vocab, DType infty, 
                              int *best_hyp, int *best_word_indices, int *finished, 
                              DType *scores, DType *scores_acc, DType *vocab_id, DType *penalties) {
  // "beam" refers to "beam_size" rows of the data. 
  const int beam(blockIdx.x);
  // First/last element in the "scores" array that will be processed.
  const int first(beam*beam_size*N+threadIdx.x), last(N*(beam*beam_size+(t == 1 ? 1 : beam_size)));
  // Buffer for blockwise reduction. Partitioned into a section for indices and a section for scores.
  // beam_size elements per thread of either data, i.e. for a beam size of 5 this are 40 byte/thread. 
  extern __shared__ int buff[];
  // Determine start of the buffer sections for this thread. 
  const int offset(threadIdx.x*beam_size);
  int *ind_buff = &buff[offset];
  DType *val_buff = ((DType *)&buff[blockDim.x*beam_size])+offset;
  // Initialize top-K values for this thread. 
  for( int i = 0; i < beam_size; ++i ) {
    val_buff[i] = infty;
  }
  // Process all score values associated with this thread. 
  for( int i = first; i < last; i += blockDim.x ) {
    // Data row of this element.
    const int row(i/N);
    // Adjust score values
    DType val(scores[i]);
    val = finished[row] ? (i == row*N+pad_id ? scores_acc[row] : infty)
                        : (val+scores_acc[row]*penalties[2*row])/penalties[2*row+1];  
    scores[i] = val;
    for (int k = beam_size; k-- && val_buff[k] > val; ) {
      if( k+1 < beam_size ) {
        val_buff[k+1] = val_buff[k];
        ind_buff[k+1] = ind_buff[k];
      }
      val_buff[k]   = val;
      ind_buff[k]   = i;
    }
  }
  // Recursive merge on thread block.
  for (unsigned int s = (blockDim.x+1)/2, last_s = blockDim.x; last_s > 1; last_s = s, s = (s+1)/2) {
    __syncthreads();
    if (threadIdx.x < s && threadIdx.x+s < last_s ) {
      MergeTopK(beam_size, val_buff, ind_buff, val_buff+s*beam_size, ind_buff+s*beam_size);
    }
  }
  // Final updates on master thread. 
  if( threadIdx.x == 0 ) {
    for (int i = 0; i < beam_size; ++i) {
      const int row(beam*beam_size+i);
      scores_acc[row] = val_buff[i];
      best_hyp[row] = ind_buff[i]/N;
      best_word_indices[row] = ind_buff[i]%N;
      if (restrict_vocab) {
        best_word_indices[row] = int(vocab_id[best_word_indices[row]]);
      }
    } 
  }
}

template<typename DType>
__global__ void update_with_best(int t, int max_output_length, int encoded_source_length, int pad_id, int eos_id,
                                 int *sequences, int *best_word_indices, int *finished, int *out, 
                                 DType *lengths, DType *att_scores, DType *att) {
  extern __shared__ int done[];
  // We will only call it with a single thread block. 
  // "beam" refers to one row of the data. 
  const int beam(threadIdx.x);
  // # (5) update best hypotheses, their attention lists and lengths (only for non-finished hyps)
  // sequences[:, t] = best_word_indices
  sequences[beam*max_output_length+t] = best_word_indices[beam];
  // attentions[:, t, :] = attention_scores
  att += (beam*max_output_length+t)*encoded_source_length;
  att_scores += beam*encoded_source_length;
  for( int i = 0; i < encoded_source_length; ++i ) {
    att[i] = att_scores[i];
  }
  //lengths += mx.nd.cast(1 - mx.nd.expand_dims(finished, axis=1), dtype='float32')
  lengths[beam] += 1.0 - finished[beam];
  // # (6) determine which hypotheses in the beam are now finished
  // finished = ((best_word_indices == C.PAD_ID) + (best_word_indices == self.vocab_target[C.EOS_SYMBOL]))
  finished[beam] = ((best_word_indices[beam] == pad_id) || (best_word_indices[beam] == eos_id) ? 1 : 0);
  done[beam] = finished[beam];
  for (unsigned int s = (blockDim.x+1)/2, last_s = blockDim.x; last_s > 1; last_s = s, s = (s+1)/2) {
    __syncthreads();
    if (beam < s && beam+s < last_s ) {
      done[beam] = done[beam] && done[beam+s];
    }
  }
  if( beam == 0 ) {
    *out = done[beam];
  }
}

template<typename DType>
__device__ void take_rows(int M, int N, DType *matrix, DType *work, int *rows) {
  const int first(threadIdx.x), last(M*N);
  for( int i = first; i < last; i += blockDim.x ) {
    work[i] = matrix[rows[i/N]*N+i%N];
  }
  __syncthreads();
  for( int i = first; i < last; i += blockDim.x ) {
    matrix[i] = work[i];
  }
}

template<typename DType0, typename DType1, typename DType2, typename DType3, typename DType4>
__global__ void take_rows_batch(int M, int N0, int N1, int N2, int N3, int N4, int *rows, 
                                DType0 *matrix0, DType1 *matrix1, DType2 *matrix2, DType3 *matrix3, DType4 *matrix4, 
                                DType0 *work0, DType1* work1, DType2 *work2, DType3 *work3, DType4 *work4) { 
  if( blockIdx.x == 0 && matrix0) take_rows(M, N0, matrix0, work0, rows);
  if( blockIdx.x == 1 && matrix1) take_rows(M, N1, matrix1, work1, rows);
  if( blockIdx.x == 2 && matrix2) take_rows(M, N2, matrix2, work2, rows);
  if( blockIdx.x == 3 && matrix3) take_rows(M, N3, matrix3, work3, rows);
  if( blockIdx.x == 4 && matrix4) take_rows(M, N4, matrix4, work4, rows);
}

struct TakeRowsGpu {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int N, DType *in, DType *out, int *rows) {
    out[i] = rows ? in[rows[i/N]*N+i%N] : in[i];
  }
};


template<typename DType>
void TakeRows(DType *matrix, DType *buffer, int *rows, int M, int N, mshadow::Stream<gpu> *s) {
  mxnet_op::Kernel<TakeRowsGpu, gpu>::Launch(s, M*N, N, matrix, buffer, rows);
  mxnet_op::Kernel<TakeRowsGpu, gpu>::Launch(s, M*N, N, buffer, matrix, (int *)0);
}

struct ComputeLengthPenalties {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType alpha, DType beta, DType *lengths, DType *out) {
    out[2*i]   = pow((beta + lengths[i]-1)/(beta + 1), alpha);
    out[2*i+1] = pow((beta + lengths[i])/(beta + 1), alpha);
  }
};

template<typename DType>
void SockeyeBeamSearchForwardGpu(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<gpu> *s = ctx.get_stream<gpu>();
  auto parms(nnvm::get<SockeyeBeamSearchParam>(attrs.parsed));
  const int beam_size(parms.beam_size), batch_size(parms.batch_size);
  const int M(beam_size*batch_size);

  // Infer some values. 
  const int max_output_length(inputs[beam::katt].shape_[1]);
  const int encoded_source_length(inputs[beam::katt].shape_[2]);
 
  DType *scores     = inputs[beam::kscore].dptr<DType>();
  DType *scores_acc = inputs[beam::kscore_acc].dptr<DType>();
  DType *lengths    = inputs[beam::klen].dptr<DType>();
  DType *att        = inputs[beam::katt].dptr<DType>();
  DType *att_scores = inputs[beam::katt_score].dptr<DType>();
  DType *vocab_id   = inputs[beam::kvoc].dptr<DType>();
  int   *finished   = inputs[beam::kfin].dptr<int>();
  int   *best_hyp   = inputs[beam::khyp].dptr<int>();
  int   *sequences  = inputs[beam::kseq].dptr<int>();
  
  // Compute row lengths of the matrices where we later perform a take-operation.
  // Compute workspace requirements for these takes. 
  const int N_seq(inputs[beam::kseq].shape_.Size()/M),
            N_fin(inputs[beam::kfin].shape_.Size()/M),
            N_len(inputs[beam::klen].shape_.Size()/M),
            N_att(inputs[beam::katt].shape_.Size()/M),
            N_att_score(inputs[beam::katt_score].shape_.Size()/M),
            W_seq(0),
            W_fin(W_seq+sizeof(int)*M*N_seq),
            W_len(W_fin+sizeof(int)*M*N_fin),
            W_att(W_len+sizeof(DType)*M*N_len),
            W_att_score(W_att+sizeof(DType)*M*N_att),
            W_all(W_att_score+sizeof(DType)*M*N_att_score);

  // Allocate workspace for the in-place take operations and length penalties and best_word_indices.
  const int wsize(M*sizeof(int)+std::max(2*M*sizeof(DType), size_t(W_all)));
  char *wspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(wsize), s).dptr_;
  int *best_word_indices = (int *)(wspace);
  wspace += sizeof(int)*M;
  DType *wspace_d = (DType *)wspace;
      
  Kernel<ComputeLengthPenalties, gpu>::Launch(s, M, DType(parms.alpha), DType(parms.beta), lengths, wspace_d);

  // Every element of the batch gets one thread block assigned. 
  find_best_hyp<<<batch_size, 1024, 1024*beam_size*(sizeof(int)+sizeof(DType)), mshadow::Stream<gpu>::GetStream(s)>>>
     (parms.step, inputs[beam::kscore].shape_[1], beam_size, parms.pad_id, parms.restrict_vocab, red::limits::MaxValue<DType>(), 
      best_hyp, best_word_indices, finished, scores, scores_acc, vocab_id, wspace_d);

  // # (4) get hypotheses and their properties for beam_size winning hypotheses (ascending) 
  // sequences = mx.nd.take(sequences, best_hyp_indices)
  // finished = mx.nd.take(finished, best_hyp_indices)
  // lengths = mx.nd.take(lengths, best_hyp_indices)  
  // attentions = mx.nd.take(attentions, best_hyp_indices)
  // attention_scores = mx.nd.take(attention_scores, best_hyp_indices)
  if( M * N_att > 20000 ) {
    TakeRows(att, (DType *)(wspace+W_att), best_hyp, M, N_att, s);
    take_rows_batch<<<4, 1024, 0, mshadow::Stream<gpu>::GetStream(s)>>>
       (M, N_seq, N_fin, N_len, N_att_score, 0, best_hyp, sequences, finished, lengths, att_scores, (int *)0, 
        (int *)(wspace+W_seq), (int *)(wspace+W_fin), (DType *)(wspace+W_len), (DType *)(wspace+W_att_score), (int *)0);
  } else {
    take_rows_batch<<<5, 1024, 0, mshadow::Stream<gpu>::GetStream(s)>>>
       (M, N_seq, N_fin, N_len, N_att, N_att_score, best_hyp, sequences, finished, lengths, att, att_scores, 
        (int *)(wspace+W_seq), (int *)(wspace+W_fin), (DType *)(wspace+W_len), (DType *)(wspace+W_att), (DType *)(wspace+W_att_score));
  }

  // Must use a single thread block and one thread/data row. 
  CHECK_LE(M, 700)<<"too big batch size"; 
  update_with_best<<<1, M, sizeof(int)*M, mshadow::Stream<gpu>::GetStream(s)>>>
      (parms.step, max_output_length, encoded_source_length, parms.pad_id, parms.eos_id,
       sequences, best_word_indices, finished, outputs[0].dptr<int>(), 
       lengths, att_scores, att);
}

template<>
MSHADOW_FORCE_INLINE void SockeyeBeamSearchForward<gpu>(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  // Factor out code into a separate function as MSHADOW_REAL_TYPE_SWITCH
  // and #pragma mess up together.
  MSHADOW_REAL_TYPE_SWITCH(inputs[beam::kscore].type_flag_, DType, {
    SockeyeBeamSearchForwardGpu<DType>(attrs, ctx, inputs, req, outputs);
  });
}

#endif

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SOCKEYE_BEAM_SEARCH_H_
