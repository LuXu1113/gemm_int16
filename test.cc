#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include "gemm_calculator_int16.h"

#define MAX_MATRIX_NUM (100)
#define MAX_MATRIX_SIZE (1024)
#define TEST_ROUND_NUM (100)
#define TRANS_B (true)

int32 performence(int16 *m_a[], int16 *m_b[], float *m_c[], const int32 m, const int32 n, const int32 k);
void accuracy(int16 *m_a[], int16 *m_b[], float *m_c[], const int32 m, const int32 n, const int32 k);

int main(int argc, char **argv) {
  int32 ret = SUCCESS;

  // check parameters
  if (4 != argc) {
    fprintf(stderr, "Usage: %s [m] [n] [k]", argv[0]);
    return ERROR_INVALID_PARAMETER;
  }
  int32 m = atoi(argv[1]);
  int32 n = atoi(argv[2]);
  int32 k = atoi(argv[3]);
  if (!(m > 0 && n > 0 && k > 0 && m <= MAX_MATRIX_SIZE && n <= MAX_MATRIX_SIZE && k <= MAX_MATRIX_SIZE)) {
    fprintf(stderr, "[ERROR] invalid parameter, m = %d, n = %d, k = %d.\n", m, n, k);
    return ERROR_INVALID_PARAMETER;
  }

  // initialize matrix
  fprintf(stdout, "initializing ...\n");
  static int16 *m_a[MAX_MATRIX_NUM];
  static int16 *m_b[MAX_MATRIX_NUM];
  static float *m_c[MAX_MATRIX_NUM];
  for (int i = 0; i < MAX_MATRIX_NUM; ++i) {
    // m_a[i] = (int16 *)malloc(MAX_MATRIX_SIZE * MAX_MATRIX_SIZE * sizeof(int16));
    // m_b[i] = (int16 *)malloc(MAX_MATRIX_SIZE * MAX_MATRIX_SIZE * sizeof(int16));
    // m_c[i] = (float *)malloc(MAX_MATRIX_SIZE * MAX_MATRIX_SIZE * sizeof(float));
    m_a[i] = (int16 *)memalign(32, MAX_MATRIX_SIZE * MAX_MATRIX_SIZE * sizeof(int16));
    m_b[i] = (int16 *)memalign(32, MAX_MATRIX_SIZE * MAX_MATRIX_SIZE * sizeof(int16));
    m_c[i] = (float *)memalign(32, MAX_MATRIX_SIZE * MAX_MATRIX_SIZE * sizeof(float));
    if (NULL == m_a[i] || NULL == m_b[i] || NULL == m_c[i]) {
      fprintf(stderr, "[ERROR] error no mem.");
      ret = ERROR_NO_MEM;
      break;
    }

    for (int j = 0; j < MAX_MATRIX_SIZE * MAX_MATRIX_SIZE; ++j) {
      m_a[i][j] = (int16)((float)rand() / (float)(RAND_MAX) * 256);
      m_b[i][j] = (int16)((float)rand() / (float)(RAND_MAX) * 256);
    }

    if ((i + 1) % (MAX_MATRIX_NUM / 10 + 1) == 0) {
      fprintf(stdout, "             ... %5.2f%%\n", (float)(i + 1) / (float)MAX_MATRIX_NUM * 100);
    }
  }

  ret = performence(m_a, m_b, m_c, m, n, k);
  accuracy(m_a, m_b, m_c, m, n, k);

  return ret;
}

int32 performence(int16 *m_a[], int16 *m_b[], float *m_c[], const int32 m, const int32 n, const int32 k) {
  int32  ret = SUCCESS;
  struct timespec ts1 = {0, 0};
  struct timespec ts2 = {0, 0};
  double exec_ns = 0.0;

  // initialize calculator
  GemmCalculatorInt16 calculator;
  ret = calculator.init();
  if (ret != SUCCESS) {
    fprintf(stderr, "[ERROR] initialize calculator fail.\n");
    return ret;
  }

  if (!TRANS_B) {
    // c = a * b
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    for (int i = 0; i < TEST_ROUND_NUM; ++i) {
      for (int i = 0; i < MAX_MATRIX_NUM; ++i) {
        calculator.gemm_int16_AxB(m_a[i], m_b[i], m_c[i], m, n, k, k, n, n);
      }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);
    fprintf(stdout, "[c = a * b] Delay(avg): %.2lf us\n", exec_ns / 1e3 / TEST_ROUND_NUM / MAX_MATRIX_NUM);
    fprintf(stdout, "[c = a * b] Perf(avg): %.2lf GFlopS\n",
            2.0 * m * n * k / (exec_ns / TEST_ROUND_NUM/ MAX_MATRIX_NUM));
  } else {
    // c = a * b_T
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    for (int i = 0; i < TEST_ROUND_NUM; ++i) {
      for (int i = 0; i < MAX_MATRIX_NUM; ++i) {
        calculator.gemm_int16_AxBT(m_a[i], m_b[i], m_c[i], m, n, k, k, k, n);
      }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    exec_ns = 1e9 * (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec);
    fprintf(stdout, "[c = a * b_T] Delay(avg): %.2lf us\n", exec_ns / 1e3 / TEST_ROUND_NUM / MAX_MATRIX_NUM);
    fprintf(stdout, "[c = a * b_T] Perf(avg): %.2lf GFlopS\n",
            2.0 * m * n * k / (exec_ns / TEST_ROUND_NUM/ MAX_MATRIX_NUM));
  }

  return ret;
}

void accuracy(int16 *m_a[], int16 *m_b[], float *m_c[], const int32 m, const int32 n, const int32 k) {
  float *s_c[MAX_MATRIX_NUM];
  for (int32 i = 0; i < MAX_MATRIX_NUM; ++i) {
    s_c[i] = (float *)malloc(MAX_MATRIX_SIZE * MAX_MATRIX_SIZE * sizeof(float));
  }

  if (!TRANS_B) {
    // sample gemm c = a * b
    for (int32 i = 0; i < MAX_MATRIX_NUM; ++i) {
      memset(s_c[i], 0, sizeof(*(s_c[i])) * MAX_MATRIX_SIZE * MAX_MATRIX_SIZE);;
      for (int32 row_c = 0; row_c < m; ++row_c) {
        for (int32 col_c = 0; col_c < n; ++col_c) {
          for (int32 j = 0; j < k; ++j) {
            s_c[i][row_c * n + col_c] += (float)m_a[i][row_c * k + j] * (float)m_b[i][j * n + col_c];
          }
        }
      }
    }
  
    // check result c = a * b
    for (int32 i = 0; i < MAX_MATRIX_NUM; ++i) {
      for (int32 row_c = 0; row_c < m; ++row_c) {
        for (int32 col_c = 0; col_c < n; ++col_c) {
          float t1 = s_c[i][row_c * n + col_c];
          float t2 = m_c[i][row_c * n + col_c];
  
          if (fabs(t1 - t2) > 1e-6) {
            fprintf(stdout, "[c = a * b][%d, %d]: %f -> %f\n", row_c, col_c, t1, t2);
          }
        }
      }
    }
  } else {
    // sample gemm c = a * b_T
    for (int32 i = 0; i < MAX_MATRIX_NUM; ++i) {
      memset(s_c[i], 0, sizeof(*(s_c[i])) * MAX_MATRIX_SIZE * MAX_MATRIX_SIZE);;
      for (int32 row_c = 0; row_c < m; ++row_c) {
        for (int32 col_c = 0; col_c < n; ++col_c) {
          for (int32 j = 0; j < k; ++j) {
            s_c[i][row_c * n + col_c] += (float)m_a[i][row_c * k + j] * (float)m_b[i][col_c * k + j];
          }
        }
      }
    }
  
    // check result c = a * b_T
    for (int32 i = 0; i < MAX_MATRIX_NUM; ++i) {
      for (int32 row_c = 0; row_c < m; ++row_c) {
        for (int32 col_c = 0; col_c < n; ++col_c) {
          float t1 = s_c[i][row_c * n + col_c];
          float t2 = m_c[i][row_c * n + col_c];
  
          if (fabs(t1 - t2) > 1e-6) {
            fprintf(stdout, "[c = a * b_T][%d, %d]: %f -> %f\n", row_c, col_c, t1, t2);
          }
        }
      }
    }
  }

  for (int32 i = MAX_MATRIX_NUM - 1; i >= 0; --i) {
    free(s_c[i]);
  }
}

