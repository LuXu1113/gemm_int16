/*
 * Introduction:
 *
 *   This code is used to calculate 16-bit sigend integer matrix multiplication,
 * using AVX and AVX2 instructions, supporting there are 16 vector-registers on CPU.
 * It is *NOT* thread-safe.
 *
 */

#include <immintrin.h>

typedef int   int32;
typedef short int16;
typedef char  int8;
typedef unsigned int   uint32;
typedef unsigned short uint16;
typedef unsigned char  uint8;

// Number 1 ~ 132 reserved for Linux errno.
enum ErrorNumber {
  SUCCESS                 = 0,
  ERROR_NO_MEM            = 150,
  ERROR_INVALID_PARAMETER = 151,
  ERROR_NO_INITIALIZE     = 152,
};

// CPU arguments.
static const int32 g_number_of_registers = 16;
static const int32 g_multiply_latency    = 4;

class GemmCalculatorInt16 {
 public:
  GemmCalculatorInt16();
  ~GemmCalculatorInt16();

  int32 init();
  bool  is_ready() const;

  int32 gemm_int16_AxB(const int16 *mat_a, const int16 *mat_b, float *mat_c,
                       const int32 m, const int32 n, const int32 k,
                       const int32 lda, const int32 ldb, const int32 ldc);
  int32 gemm_int16_AxBT(const int16 *mat_a, const int16 *mat_b, float *mat_c,
                        const int32 m, const int32 n, const int32 k,
                        const int32 lda, const int32 ldb, const int32 ldc);

 private:
  inline void fill_block_into_buffer(int16 *block, const int16 *mat,
                                     const int32 block_rows, const int32 block_cols,
                                     const int32 mat_rows, const int32 mat_cols,
                                     const int32 start_row_id, const int32 start_col_id,
                                     const int32 ld, const bool transpose);
  inline void fill_block_into_buffer(float *block, const float *mat,
                                     const int32 block_rows, const int32 block_cols,
                                     const int32 mat_rows, const int32 mat_cols,
                                     const int32 start_row_id, const int32 start_col_id,
                                     const int32 ld);
  inline void fill_block_into_mat(const float *block, float *mat,
                                     const int32 block_rows, const int32 block_cols,
                                     const int32 mat_rows, const int32 mat_cols,
                                     const int32 start_row_id, const int32 start_col_id,
                                     const int32 ld);
  inline void mul_and_add(float *blk_c);

  // wether instance of calculator can be used.
  bool ready_;

  // hold registers of one cpu core.
  __m256i registers_[g_number_of_registers];
  __m256  *tmp_regs_[g_number_of_registers];

  // hold blocks of matrixs.
  int16 *mat_a_block_;
  int16 *mat_b_block_;
  float *mat_c_block_;
};


