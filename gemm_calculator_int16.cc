#include "gemm_calculator_int16.h"
#include <malloc.h>
#include <memory.h>
#include <time.h>
#include <xmmintrin.h>
#include <immintrin.h>

static const int32 g_num_rows_of_block_a = 4;
static const int32 g_num_cols_of_block_a = 4;
static const int32 g_num_elem_of_block_a = 16;
static const int32 g_num_rows_of_block_b = 4;
static const int32 g_num_cols_of_block_b = 4;
static const int32 g_num_elem_of_block_b = 16;
static const int32 g_num_rows_of_block_c = 4;
static const int32 g_num_cols_of_block_c = 4;
static const int32 g_num_elem_of_block_c = 16;

GemmCalculatorInt16::GemmCalculatorInt16() {
  ready_ = false;

  // for convert data type
  tmp_regs_[0] = (__m256 *)&(registers_[8]);
  tmp_regs_[1] = (__m256 *)&(registers_[9]);
  tmp_regs_[2] = (__m256 *)&(registers_[10]);
  tmp_regs_[3] = (__m256 *)&(registers_[11]);
  tmp_regs_[4] = (__m256 *)&(registers_[12]);
  tmp_regs_[5] = (__m256 *)&(registers_[13]);
  tmp_regs_[6] = (__m256 *)&(registers_[14]);
  tmp_regs_[7] = (__m256 *)&(registers_[15]);
}

GemmCalculatorInt16::~GemmCalculatorInt16() {
  if (NULL != mat_c_block_) {
    free(mat_c_block_);
    mat_c_block_ = NULL;
  }
  if (NULL != mat_b_block_) {
    free(mat_b_block_);
    mat_b_block_ = NULL;
  }
  if (NULL != mat_a_block_) {
    free(mat_a_block_);
    mat_a_block_ = NULL;
  }
}

bool GemmCalculatorInt16::is_ready() const {
  return ready_;
}

int32 GemmCalculatorInt16::init() {
  if (true == ready_) {
    return SUCCESS;
  }

  mat_a_block_ = (int16 *)memalign(32, sizeof(*mat_a_block_) * g_num_elem_of_block_a * g_multiply_latency);
  if (NULL == mat_a_block_) {
    return ERROR_NO_MEM;
  }

  mat_b_block_ = (int16 *)memalign(32, sizeof(*mat_b_block_) * g_num_elem_of_block_b * g_multiply_latency);
  if (NULL == mat_b_block_) {
    return ERROR_NO_MEM;
  }

  mat_c_block_ = (float *)memalign(32, sizeof(*mat_c_block_) * g_num_elem_of_block_c);
  if (NULL == mat_c_block_) {
    return ERROR_NO_MEM;
  }

  ready_ = true;
  return SUCCESS;
}

int32 GemmCalculatorInt16::gemm_int16_AxB(const int16 *mat_a, const int16 *mat_b, float *mat_c,
                                          const int32 m, const int32 n, const int32 k,
                                          const int32 lda, const int32 ldb, const int32 ldc) {
  if (NULL == mat_a || NULL == mat_b || NULL == mat_c
      || m <=0 || n <=0 || k <= 0 || lda <=0 || ldb <= 0 || ldc <= 0) {
    fprintf(stderr, "[ERROR] invalid parameter: mat_a = %p, mat_b = %p, mat_c = %p"
                    "                           m = %d, n = %d, k = %d"
                    "                           lda = %d, ldb = %d, ldc = %d.\n",
                    mat_a, mat_b, mat_c, m, n, k, lda, ldb, ldc);
    return ERROR_INVALID_PARAMETER;
  }
  if (false == ready_) {
    fprintf(stderr, "[ERROR] calculator have not been initialized.\n");
    return ERROR_NO_INITIALIZE;
  }

  int32 nblk_in_row_a = (k + g_num_cols_of_block_a - 1) / g_num_cols_of_block_a;
  int32 nblk_in_col_a = (m + g_num_rows_of_block_a - 1) / g_num_rows_of_block_a;
  int32 nblk_in_row_b = (n + g_num_cols_of_block_b - 1) / g_num_cols_of_block_b;
  int32 nblk_in_col_b = (k + g_num_rows_of_block_b - 1) / g_num_rows_of_block_b;

  // initialize mat_c
  memset(mat_c, 0, sizeof(*mat_c) * m * ldc);

  // loop over all mat_b blocks.
  for (int32 blk_cid_in_b = 0; blk_cid_in_b < nblk_in_row_b; ++blk_cid_in_b) {
    for (int32 blk_rid_in_b = 0; blk_rid_in_b < nblk_in_col_b; blk_rid_in_b += g_multiply_latency) {
      // fetch [g_multiply_latency] blocks from [mat_b] to [mat_b_block_]
      for (int32 i = 0; i < g_multiply_latency; ++i) {
        int16 *buffer_b = mat_b_block_ + i * g_num_elem_of_block_b;
        int32 global_rid_b = g_num_rows_of_block_b * (blk_rid_in_b + i);
        int32 global_cid_b = g_num_cols_of_block_b * blk_cid_in_b;
        fill_block_into_buffer(buffer_b, mat_b, g_num_rows_of_block_b, g_num_cols_of_block_b,
                               k, n, global_rid_b, global_cid_b, ldb, true);
      }

      registers_[4] = _mm256_load_si256((__m256i*)(mat_b_block_));
      registers_[5] = _mm256_load_si256((__m256i*)(mat_b_block_ + 16));
      registers_[6] = _mm256_load_si256((__m256i*)(mat_b_block_ + 32));
      registers_[7] = _mm256_load_si256((__m256i*)(mat_b_block_ + 48));

      for (int32 blk_rid_in_a = 0; blk_rid_in_a < nblk_in_col_a; ++blk_rid_in_a) {
        // fetch [g_multiply_latency] blocks from [mat_a] to [mat_a_block_]
        for (int32 i = 0; i < g_multiply_latency; ++i) {
          int16 *buffer_a = mat_a_block_ + i * g_num_elem_of_block_a;
          int32 global_rid_a = g_num_rows_of_block_a * blk_rid_in_a;
          int32 global_cid_a = g_num_cols_of_block_a * (blk_rid_in_b + i);
          fill_block_into_buffer(buffer_a, mat_a, g_num_rows_of_block_a, g_num_cols_of_block_a,
                                 m, k, global_rid_a, global_cid_a, lda, false);
        }

        registers_[0] = _mm256_load_si256((__m256i*)(mat_a_block_));
        registers_[1] = _mm256_load_si256((__m256i*)(mat_a_block_ + 16));
        registers_[2] = _mm256_load_si256((__m256i*)(mat_a_block_ + 32));
        registers_[3] = _mm256_load_si256((__m256i*)(mat_a_block_ + 48));

        // fetch result block from [mat_c] to [matc_c_block_]
        int32 global_rid_c = blk_rid_in_a * g_num_rows_of_block_c;
        int32 global_cid_c = blk_cid_in_b * g_num_cols_of_block_c;
        fill_block_into_buffer(mat_c_block_, mat_c, g_num_rows_of_block_c, g_num_cols_of_block_c,
                               m, n, global_rid_c, global_cid_c, ldc);

        // [mat_block_] += 危([mat_a_block_[i]] * [mat_b_block_[i]]) (i from 0 to [g_multiply_latency] - 1)
        mul_and_add(mat_c_block_);

        fill_block_into_mat(mat_c_block_, mat_c, g_num_rows_of_block_c, g_num_cols_of_block_c,
                            m, n, global_rid_c, global_cid_c, ldc);
      }
    }
  }
  return SUCCESS;
}

inline void GemmCalculatorInt16::fill_block_into_buffer(int16 *block, const int16 *mat,
                                                        const int32 block_rows, const int32 block_cols,
                                                        const int32 mat_rows, const int32 mat_cols,
                                                        const int32 start_row_id, const int32 start_col_id,
                                                        const int32 ld, const bool transpose) {
  if (false == transpose) {
    for (int32 i = 0; i < block_rows; ++i) {
      for (int32 j = 0; j < block_cols; ++j) {
        if ((start_row_id + i < mat_rows) && (start_col_id + j < mat_cols)) {
          block[i * block_cols + j] = mat[(start_row_id + i) * ld + (start_col_id + j)];
        } else {
          block[i * block_cols + j] = 0;
        }
      }
    }
  } else {
    for (int32 i = 0; i < block_rows; ++i) {
      for (int32 j = 0; j < block_cols; ++j) {
        if ((start_row_id + i < mat_rows) && (start_col_id + j < mat_cols)) {
          block[j * block_rows + i] = mat[(start_row_id + i) * ld + (start_col_id + j)];
        } else {
          block[j * block_rows + i] = 0;
        }
      }
    }
  }
}

inline void GemmCalculatorInt16::fill_block_into_buffer(float *block, const float *mat,
                                                        const int32 block_rows, const int32 block_cols,
                                                        const int32 mat_rows, const int32 mat_cols,
                                                        const int32 start_row_id,const  int32 start_col_id,
                                                        const int32 ld) {
  for (int32 i = 0; i < block_rows; ++i) {
    for (int32 j = 0; j < block_cols; ++j) {
      if ((start_row_id + i < mat_rows) && (start_col_id + j < mat_cols)) {
        block[i * block_cols + j] = mat[(start_row_id + i) * ld + (start_col_id + j)];
      } else {
        block[i * block_cols + j] = 0;
      }
    }
  }
}

inline void GemmCalculatorInt16::fill_block_into_mat(const float *block, float *mat,
                                                     const int32 block_rows, const int32 block_cols,
                                                     const int32 mat_rows, const int32 mat_cols,
                                                     const int32 start_row_id, const int32 start_col_id,
                                                     const int32 ld) {
  for (int32 i = 0; i < block_rows; ++i) {
    for (int32 j = 0; j < block_cols; ++j) {
      if ((start_row_id + i < mat_rows) && (start_col_id + j < mat_cols)) {
        mat[(start_row_id + i) * ld + (start_col_id + j)] = block[i * block_cols + j];
      }
    }
  }
}

inline void GemmCalculatorInt16::mul_and_add(float *blk_c) {
  // 1st round
  {
    // element-wise multiply and add nearby products
    registers_[8] = _mm256_madd_epi16(registers_[0], registers_[4]);
    registers_[9] = _mm256_madd_epi16(registers_[1], registers_[5]);
    registers_[10] = _mm256_madd_epi16(registers_[2], registers_[6]);
    registers_[11] = _mm256_madd_epi16(registers_[3], registers_[7]);

    // convert int32 to float
    *(tmp_regs_[0]) = _mm256_cvtepi32_ps(registers_[8]);
    *(tmp_regs_[1]) = _mm256_cvtepi32_ps(registers_[9]);
    *(tmp_regs_[2]) = _mm256_cvtepi32_ps(registers_[10]);
    *(tmp_regs_[3]) = _mm256_cvtepi32_ps(registers_[11]);
 
    // sum
    *(tmp_regs_[0]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
    *(tmp_regs_[1]) = _mm256_add_ps(*(tmp_regs_[2]), *(tmp_regs_[3]));
 
    *(tmp_regs_[4]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
 
    // shuffle for next round
    registers_[4] = _mm256_permute4x64_epi64(registers_[4], 0x39);
    registers_[5] = _mm256_permute4x64_epi64(registers_[5], 0x39);
    registers_[6] = _mm256_permute4x64_epi64(registers_[6], 0x39);
    registers_[7] = _mm256_permute4x64_epi64(registers_[7], 0x39);
  }
 
  // 2ed round
  {
    // element-wise multiply and add nearby products
    registers_[8] = _mm256_madd_epi16(registers_[0], registers_[4]);
    registers_[9] = _mm256_madd_epi16(registers_[1], registers_[5]);
    registers_[10] = _mm256_madd_epi16(registers_[2], registers_[6]);
    registers_[11] = _mm256_madd_epi16(registers_[3], registers_[7]);
 
    // convert int32 to float
    *(tmp_regs_[0]) = _mm256_cvtepi32_ps(registers_[8]);
    *(tmp_regs_[1]) = _mm256_cvtepi32_ps(registers_[9]);
    *(tmp_regs_[2]) = _mm256_cvtepi32_ps(registers_[10]);
    *(tmp_regs_[3]) = _mm256_cvtepi32_ps(registers_[11]);
 
    // sum
    *(tmp_regs_[0]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
    *(tmp_regs_[1]) = _mm256_add_ps(*(tmp_regs_[2]), *(tmp_regs_[3]));
 
    *(tmp_regs_[5]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
 
    // shuffle for next round
    registers_[4] = _mm256_permute4x64_epi64(registers_[4], 0x39);
    registers_[5] = _mm256_permute4x64_epi64(registers_[5], 0x39);
    registers_[6] = _mm256_permute4x64_epi64(registers_[6], 0x39);
    registers_[7] = _mm256_permute4x64_epi64(registers_[7], 0x39);
  }
 
  // merge 1st round and 2ed round
  *(tmp_regs_[5]) = _mm256_hadd_ps(*(tmp_regs_[5]), *(tmp_regs_[4]));

  // 3rd round
  {
    // element-wise multiply and add nearby products
    registers_[8] = _mm256_madd_epi16(registers_[0], registers_[4]);
    registers_[9] = _mm256_madd_epi16(registers_[1], registers_[5]);
    registers_[10] = _mm256_madd_epi16(registers_[2], registers_[6]);
    registers_[11] = _mm256_madd_epi16(registers_[3], registers_[7]);
 
    // convert int32 to float
    *(tmp_regs_[0]) = _mm256_cvtepi32_ps(registers_[8]);
    *(tmp_regs_[1]) = _mm256_cvtepi32_ps(registers_[9]);
    *(tmp_regs_[2]) = _mm256_cvtepi32_ps(registers_[10]);
    *(tmp_regs_[3]) = _mm256_cvtepi32_ps(registers_[11]);
 
    // sum
    *(tmp_regs_[0]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
    *(tmp_regs_[1]) = _mm256_add_ps(*(tmp_regs_[2]), *(tmp_regs_[3]));
 
    *(tmp_regs_[6]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
 
    // shuffle for next round
    registers_[4] = _mm256_permute4x64_epi64(registers_[4], 0x39);
    registers_[5] = _mm256_permute4x64_epi64(registers_[5], 0x39);
    registers_[6] = _mm256_permute4x64_epi64(registers_[6], 0x39);
    registers_[7] = _mm256_permute4x64_epi64(registers_[7], 0x39);
  }
 
  // 4th round
  {
    // element-wise multiply and add nearby products
    registers_[8] = _mm256_madd_epi16(registers_[0], registers_[4]);
    registers_[9] = _mm256_madd_epi16(registers_[1], registers_[5]);
    registers_[10] = _mm256_madd_epi16(registers_[2], registers_[6]);
    registers_[11] = _mm256_madd_epi16(registers_[3], registers_[7]);
 
    // convert int32 to float
    *(tmp_regs_[0]) = _mm256_cvtepi32_ps(registers_[8]);
    *(tmp_regs_[1]) = _mm256_cvtepi32_ps(registers_[9]);
    *(tmp_regs_[2]) = _mm256_cvtepi32_ps(registers_[10]);
    *(tmp_regs_[3]) = _mm256_cvtepi32_ps(registers_[11]);
 
    // sum
    *(tmp_regs_[0]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
    *(tmp_regs_[1]) = _mm256_add_ps(*(tmp_regs_[2]), *(tmp_regs_[3]));
 
    *(tmp_regs_[7]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[1]));
 
    // shuffle for next round
    registers_[4] = _mm256_permute4x64_epi64(registers_[4], 0x39);
    registers_[5] = _mm256_permute4x64_epi64(registers_[5], 0x39);
    registers_[6] = _mm256_permute4x64_epi64(registers_[6], 0x39);
    registers_[7] = _mm256_permute4x64_epi64(registers_[7], 0x39);
  }
 
  // merge 3rd round and 4th round
  *(tmp_regs_[7]) = _mm256_hadd_ps(*(tmp_regs_[7]), *(tmp_regs_[6]));

  // permute results
  //    [0, 1] [1, 2] [0, 0] [1, 1] [2, 3] [3, 0] [2, 2] [3, 3]
  // => [0, 0] [0, 1] [2, 2] [2, 3] [3, 0] [1, 1] [1, 2] [3, 3]
  //    [0, 3] [1, 0] [0, 2] [1, 3] [2, 1] [3, 2] [2, 0] [3, 1]
  // => [2, 0] [2, 1] [0, 2] [0, 3] [1, 0] [3, 1] [3, 2] [1, 3]
  registers_[8] = _mm256_set_epi32(7, 1, 3, 5, 4, 6, 0, 2);
  registers_[9] = _mm256_set_epi32(3, 5, 7, 1, 0, 2, 4, 6);
  *(tmp_regs_[2]) = _mm256_permutevar8x32_ps(*(tmp_regs_[5]), registers_[8]);
  *(tmp_regs_[3]) = _mm256_permutevar8x32_ps(*(tmp_regs_[7]), registers_[9]);
  
  // blend results
  *(tmp_regs_[0]) = _mm256_blend_ps(*(tmp_regs_[2]), *(tmp_regs_[3]), 0x9C);
  *(tmp_regs_[1]) = _mm256_blend_ps(*(tmp_regs_[3]), *(tmp_regs_[2]), 0x9C);
 
  // load blk_c
  *(tmp_regs_[2]) = _mm256_load_ps(blk_c);
  *(tmp_regs_[3]) = _mm256_load_ps(blk_c + 8);
 
  // add result to blk_c
  *(tmp_regs_[4]) = _mm256_add_ps(*(tmp_regs_[0]), *(tmp_regs_[2]));
  *(tmp_regs_[5]) = _mm256_add_ps(*(tmp_regs_[1]), *(tmp_regs_[3]));

  // write mat_c_block_ to mat_c
  _mm256_store_ps(blk_c, *(tmp_regs_[4]));
  _mm256_store_ps(blk_c + 8, *(tmp_regs_[5]));
}

int32 GemmCalculatorInt16::gemm_int16_AxBT(const int16 *mat_a, const int16 *mat_b, float *mat_c,
                                           const int32 m, const int32 n, const int32 k,
                                           const int32 lda, const int32 ldb, const int32 ldc) {
  if (NULL == mat_a || NULL == mat_b || NULL == mat_c
      || m <=0 || n <=0 || k <= 0 || lda <=0 || ldb <= 0 || ldc <= 0 || m % 2 || n % 4 || k % 16) {
    fprintf(stderr, "[ERROR] invalid parameter: mat_a = %p, mat_b = %p, mat_c = %p"
                    "                           m = %d, n = %d, k = %d"
                    "                           lda = %d, ldb = %d, ldc = %d.\n",
                    mat_a, mat_b, mat_c, m, n, k, lda, ldb, ldc);
    return ERROR_INVALID_PARAMETER;
  }

  long long buffer[4][4];
  for (int32 col_c = 0; col_c < n; col_c += 4) {
    const int16 *base_b0 = mat_b + (col_c + 0) * ldb;
    const int16 *base_b1 = mat_b + (col_c + 1) * ldb;
    const int16 *base_b2 = mat_b + (col_c + 2) * ldb;
    const int16 *base_b3 = mat_b + (col_c + 3) * ldb;

    for (int32 row_c = 0; row_c < m; row_c += 2) {
      const int16 *base_a0 = mat_a + (row_c + 0) * lda;
      const int16 *base_a1 = mat_a + (row_c + 1) * lda;
      int32 base_c = row_c * ldc + col_c;

      // load data from mat_b
      registers_[1] = _mm256_load_si256((__m256i *)(base_b0));
      registers_[2] = _mm256_load_si256((__m256i *)(base_b1));
      registers_[3] = _mm256_load_si256((__m256i *)(base_b2));
      registers_[4] = _mm256_load_si256((__m256i *)(base_b3));
      // load data from mat_a
      registers_[0] = _mm256_load_si256((__m256i *)(base_a0));

      _mm_prefetch(base_b0 + 16, (_mm_hint)1);
      _mm_prefetch(base_b1 + 16, (_mm_hint)1);
      _mm_prefetch(base_b2 + 16, (_mm_hint)1);
      _mm_prefetch(base_b3 + 16, (_mm_hint)1);

      // element-wise multiply and add nearby products
      registers_[5] = _mm256_madd_epi16(registers_[0], registers_[1]);
      registers_[6] = _mm256_madd_epi16(registers_[0], registers_[2]);
      registers_[7] = _mm256_madd_epi16(registers_[0], registers_[3]);
      registers_[0] = _mm256_madd_epi16(registers_[0], registers_[4]);

      // convert int32 to int64
      registers_[12] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 0)),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 1)));
      registers_[13] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 0)),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 1)));
      registers_[14] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 0)),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 1)));
      registers_[15] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 0)),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 1)));

      // load data from mat_a
      registers_[0] = _mm256_load_si256((__m256i *)(base_a1));

      // element-wise multiply and add nearby products
      registers_[5] = _mm256_madd_epi16(registers_[0], registers_[1]);
      registers_[6] = _mm256_madd_epi16(registers_[0], registers_[2]);
      registers_[7] = _mm256_madd_epi16(registers_[0], registers_[3]);
      registers_[0] = _mm256_madd_epi16(registers_[0], registers_[4]);

      // convert int32 to int64
      registers_[8] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 0)),
                                       _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 1)));
      registers_[9] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 0)),
                                       _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 1)));
      registers_[10] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 0)),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 1)));
      registers_[11] = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 0)),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 1)));

      for (int32 i = 16; i < k; i += 16) {
        // load data from mat_b
        registers_[1] = _mm256_load_si256((__m256i *)(base_b0 + i));
        registers_[2] = _mm256_load_si256((__m256i *)(base_b1 + i));
        registers_[3] = _mm256_load_si256((__m256i *)(base_b2 + i));
        registers_[4] = _mm256_load_si256((__m256i *)(base_b3 + i));
        // load data from mat_a
        registers_[0] = _mm256_load_si256((__m256i *)(base_a0 + i));

        _mm_prefetch(base_b0 + i + 16, (_mm_hint)1);
        _mm_prefetch(base_b1 + i + 16, (_mm_hint)1);
        _mm_prefetch(base_b2 + i + 16, (_mm_hint)1);
        _mm_prefetch(base_b3 + i + 16, (_mm_hint)1);

        // element-wise multiply and add nearby products
        registers_[5] = _mm256_madd_epi16(registers_[0], registers_[1]);
        registers_[6] = _mm256_madd_epi16(registers_[0], registers_[2]);
        registers_[7] = _mm256_madd_epi16(registers_[0], registers_[3]);
        registers_[0] = _mm256_madd_epi16(registers_[0], registers_[4]);

        // convert int32 to int64
        registers_[12] = _mm256_add_epi64(registers_[12],
                         _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 0)),
                                          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 1))));
        registers_[13] = _mm256_add_epi64(registers_[13],
                         _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 0)),
                                          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 1))));
        registers_[14] = _mm256_add_epi64(registers_[14],
                         _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 0)),
                                          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 1))));
        registers_[15] = _mm256_add_epi64(registers_[15],
                         _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 0)),
                                          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 1))));

        // load data from mat_a
        registers_[0] = _mm256_load_si256((__m256i *)(base_a1 + i));

        // element-wise multiply and add nearby products
        registers_[5] = _mm256_madd_epi16(registers_[0], registers_[1]);
        registers_[6] = _mm256_madd_epi16(registers_[0], registers_[2]);
        registers_[7] = _mm256_madd_epi16(registers_[0], registers_[3]);
        registers_[0] = _mm256_madd_epi16(registers_[0], registers_[4]);

        // convert int32 to int64
        registers_[8] = _mm256_add_epi64(registers_[8],
                        _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 0)),
                                         _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[5], 1))));
        registers_[9] = _mm256_add_epi64(registers_[9],
                        _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 0)),
                                         _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[6], 1))));
        registers_[10] = _mm256_add_epi64(registers_[10],
                         _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 0)),
                                          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[7], 1))));
        registers_[11] = _mm256_add_epi64(registers_[11],
                         _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 0)),
                                          _mm256_cvtepi32_epi64(_mm256_extracti128_si256(registers_[0], 1))));
      }
      registers_[0] = _mm256_add_epi64(_mm256_permute2x128_si256(registers_[12], registers_[13], 0x20),
                                       _mm256_permute2x128_si256(registers_[12], registers_[13], 0x31));
      registers_[1] = _mm256_add_epi64(_mm256_permute2x128_si256(registers_[14], registers_[15], 0x20),
                                       _mm256_permute2x128_si256(registers_[14], registers_[15], 0x31));
      registers_[2] = _mm256_add_epi64(_mm256_permute2x128_si256(registers_[8], registers_[9], 0x20),
                                       _mm256_permute2x128_si256(registers_[8], registers_[9], 0x31));
      registers_[3] = _mm256_add_epi64(_mm256_permute2x128_si256(registers_[10], registers_[11], 0x20),
                                       _mm256_permute2x128_si256(registers_[10], registers_[11], 0x31));

      _mm256_storeu_si256((__m256i *)buffer[0], registers_[0]);
      _mm256_storeu_si256((__m256i *)buffer[1], registers_[1]);
      _mm256_storeu_si256((__m256i *)buffer[2], registers_[2]);
      _mm256_storeu_si256((__m256i *)buffer[3], registers_[3]);

      _mm_prefetch(base_b3 + ldb, (_mm_hint)2);
      _mm_prefetch(base_b3 + ldb * 2, (_mm_hint)2);
      _mm_prefetch(base_b3 + ldb * 3, (_mm_hint)2);
      _mm_prefetch(base_b3 + ldb * 4, (_mm_hint)2);

      mat_c[base_c] = buffer[0][0] + buffer[0][1];
      mat_c[base_c + 1] = buffer[0][2] + buffer[0][3];
      mat_c[base_c + 2] = buffer[1][0] + buffer[1][1];
      mat_c[base_c + 3] = buffer[1][2] + buffer[1][3];
      mat_c[base_c + ldc] = buffer[2][0] + buffer[2][1];
      mat_c[base_c + ldc + 1] = buffer[2][2] + buffer[2][3];
      mat_c[base_c + ldc + 2] = buffer[3][0] + buffer[3][1];
      mat_c[base_c + ldc + 3] = buffer[3][2] + buffer[3][3];
    }
  }
}

