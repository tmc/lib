// Package cublas provides low-level Go bindings for cuBLAS (CUDA Basic Linear Algebra Subroutines).
//
// cuBLAS implements BLAS (Basic Linear Algebra Subprograms) on NVIDIA GPUs.
// It provides routines for vector/vector (BLAS1), matrix/vector (BLAS2), and
// matrix/matrix (BLAS3) operations in single, double, and half precision.
//
// The central type is [Handle], which manages a cuBLAS context. All BLAS
// operations are methods on Handle. Create one with [Create] and clean up
// with [Handle.Destroy].
//
// This package maps every function from cublas_api.h to a Go function using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libcublas.
// All functions return errors wrapping cublasStatus_t with descriptive messages
// from cublasGetStatusName.
//
// # BLAS3
//
//   - [Handle.Cgemm3m]: This function performs the complex matrix-matrix multiplication, using Gauss complexity reduction algorithm.
//   - [Handle.Cgemm3mStridedBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.Cgemm3mStridedBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.Cgemm3m64]: This function performs the complex matrix-matrix multiplication, using Gauss complexity reduction algorithm.
//   - [Handle.CgemmBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.CgemmBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.CgemmEx]: This function is an extension of cublasgemm().
//   - [Handle.CgemmEx64]: This function is an extension of cublasgemm().
//   - [Handle.CgemmStridedBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.CgemmStridedBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.Cgemm]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.Cgemm64]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.DgemmBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.DgemmBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.DgemmGroupedBatched]: This function performs the matrix-matrix multiplication on groups of matrices.
//   - [Handle.DgemmGroupedBatched64]: This function performs the matrix-matrix multiplication on groups of matrices.
//   - [Handle.DgemmStridedBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.DgemmStridedBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.Dgemm]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.Dgemm64]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.DtrsmBatched]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.DtrsmBatched64]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.Dtrsm]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.Dtrsm64]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.SgemmBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.SgemmBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.SgemmEx]: This function is an extension of cublasgemm().
//   - [Handle.SgemmEx64]: This function is an extension of cublasgemm().
//   - [Handle.SgemmGroupedBatched]: This function performs the matrix-matrix multiplication on groups of matrices.
//   - [Handle.SgemmGroupedBatched64]: This function performs the matrix-matrix multiplication on groups of matrices.
//   - [Handle.SgemmStridedBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.SgemmStridedBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.Sgemm]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.Sgemm64]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.StrsmBatched]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.StrsmBatched64]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.Strsm]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.Strsm64]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.Zgemm3m]: This function performs the complex matrix-matrix multiplication, using Gauss complexity reduction algorithm.
//   - [Handle.Zgemm3m64]: This function performs the complex matrix-matrix multiplication, using Gauss complexity reduction algorithm.
//   - [Handle.ZgemmBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.ZgemmBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.ZgemmStridedBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.ZgemmStridedBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.Zgemm]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.Zgemm64]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//
// # Lifecycle
//
//   - [Create]: This function initializes the cuBLAS library and creates a handle to an opaque structure holding the cuBLAS library context.
//   - [Handle.Destroy]: This function releases hardware resources used by the cuBLAS library.
//
// # BLAS1
//
//   - [Handle.Daxpy]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Daxpy64]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Dnrm2]: This function computes the Euclidean norm of the vector x.
//   - [Handle.Dnrm264]: This function computes the Euclidean norm of the vector x.
//   - [Handle.Dscal]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Dscal64]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Saxpy]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Saxpy64]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Snrm2]: This function computes the Euclidean norm of the vector x.
//   - [Handle.Snrm264]: This function computes the Euclidean norm of the vector x.
//   - [Handle.Sscal]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Sscal64]: This function scales the vector x by the scalar α and overwrites it with the result.
//
// # BLAS2
//
//   - [Handle.DgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.DgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.DgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.DgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.Dgemv]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//   - [Handle.Dgemv64]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//   - [Handle.SgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.SgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.SgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.SgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.Sgemv]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//   - [Handle.Sgemv64]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//
// # Configuration
//
//   - [Handle.GetMathMode]: This function returns the math mode used by the library routines.
//   - [Handle.GetPointerMode]: This function obtains the pointer mode used by the cuBLAS library.
//   - [Handle.SetMathMode]: The [Handle.SetMathMode] function enables you to choose the compute precision modes as defined by [Math].
//   - [Handle.SetPointerMode]: This function sets the pointer mode used by the cuBLAS library.
//
// # Stream Management
//
//   - [Handle.GetStream]: This function gets the cuBLAS library stream, which is being used to execute all calls to the cuBLAS library functions.
//   - [Handle.SetStream]: This function sets the cuBLAS library stream, which will be used to execute all subsequent calls to the cuBLAS library functions.
//
// # Other
//
//   - [Handle.AxpyEx]: This function is an API generalization of the routine cublasaxpy() where input data, output data and compute type can be specified independently.
//   - [Handle.AxpyEx64]: This function is an API generalization of the routine cublasaxpy() where input data, output data and compute type can be specified independently.
//   - [Handle.Caxpy]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Caxpy64]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Ccopy]: This function copies the vector x into the vector y.
//   - [Handle.Ccopy64]: This function copies the vector x into the vector y.
//   - [Handle.Cdgmm]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Cdgmm64]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Cdotc]: This function computes the dot product of vectors x and y.
//   - [Handle.Cdotc64]: This function computes the dot product of vectors x and y.
//   - [Handle.Cdotu]: This function computes the dot product of vectors x and y.
//   - [Handle.Cdotu64]: This function computes the dot product of vectors x and y.
//   - [Handle.Cgbmv]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Cgbmv64]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Cgeam]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.Cgeam64]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.CgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.CgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.CgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.CgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.CgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.Cgemv]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//   - [Handle.Cgemv64]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//   - [Handle.CgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Cgerc]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Cgerc64]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Cgeru]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Cgeru64]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.CgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.CgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.CgetrsBatched]: This function solves an array of systems of linear equations of the form:
//   - [Handle.Chbmv]: This function performs the Hermitian banded matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Chbmv64]: This function performs the Hermitian banded matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Chemm]: This function performs the Hermitian matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a Hermitian matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Chemm64]: This function performs the Hermitian matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a Hermitian matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Chemv]: This function performs the Hermitian matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in lower or upper mode, x and y are vectors, and α and β are scalars.
//   - [Handle.Chemv64]: This function performs the Hermitian matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in lower or upper mode, x and y are vectors, and α and β are scalars.
//   - [Handle.Cher2]: This function performs the Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Cher264]: This function performs the Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Cher2k]: This function performs the Hermitian rank- 2k update C=αop(A)op(B)H+αˉop(B)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Cher2k64]: This function performs the Hermitian rank- 2k update C=αop(A)op(B)H+αˉop(B)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Cher]: This function performs the Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Cher64]: This function performs the Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Cherk3mEx]: This function is an extension of [Handle.Cherk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64].
//   - [Handle.Cherk3mEx64]: This function is an extension of [Handle.Cherk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64].
//   - [Handle.CherkEx]: This function is an extension of [Handle.Cherk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64] This function performs the Hermitian rank- k update C=αop(A)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.CherkEx64]: This function is an extension of [Handle.Cherk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64] This function performs the Hermitian rank- k update C=αop(A)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Cherk]: This function performs the Hermitian rank- k update C=αop(A)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Cherk64]: This function performs the Hermitian rank- k update C=αop(A)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Cherkx]: This function performs a variation of the Hermitian rank- k update C=αop(A)op(B)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Cherkx64]: This function performs a variation of the Hermitian rank- k update C=αop(A)op(B)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Chpmv]: This function performs the Hermitian packed matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Chpmv64]: This function performs the Hermitian packed matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Chpr2]: This function performs the packed Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α is a scalar.
//   - [Handle.Chpr264]: This function performs the packed Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α is a scalar.
//   - [Handle.Chpr]: This function performs the packed Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Chpr64]: This function performs the packed Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.CmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.Crot]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Crot64]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Crotg]: This function constructs the Givens rotation matrix G=(cs−sc) that zeros out the second entry of a 2×1 vector (a,b)T .
//   - [Handle.Cscal]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Cscal64]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Csrot]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Csrot64]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Csscal]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Csscal64]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Cswap]: This function interchanges the elements of vector x and y.
//   - [Handle.Cswap64]: This function interchanges the elements of vector x and y.
//   - [Handle.Csymm]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Csymm64]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Csymv]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Csymv64]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Csyr2]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Csyr264]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Csyr2k]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Csyr2k64]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Csyr]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Csyr64]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Csyrk3mEx]: This function is an extension of [Handle.Csyrk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64].
//   - [Handle.Csyrk3mEx64]: This function is an extension of [Handle.Csyrk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64].
//   - [Handle.CsyrkEx]: This function is an extension of [Handle.Csyrk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64] This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.CsyrkEx64]: This function is an extension of [Handle.Csyrk] where the input matrix and output matrix can have a lower precision but the computation is still done in the type [Complex64] This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Csyrk]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Csyrk64]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Csyrkx]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Csyrkx64]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Ctbmv]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Ctbmv64]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Ctbsv]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Ctbsv64]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Ctpmv]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Ctpmv64]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Ctpsv]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Ctpsv64]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Ctpttr]: This function performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Ctrmm]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Ctrmm64]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Ctrmv]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.Ctrmv64]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.CtrsmBatched]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.CtrsmBatched64]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.Ctrsm]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.Ctrsm64]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.Ctrsv]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Ctrsv64]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Ctrttp]: This function performs the conversion from the triangular format to the triangular packed format
//   - [Handle.Dasum]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.Dasum64]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.Dcopy]: This function copies the vector x into the vector y.
//   - [Handle.Dcopy64]: This function copies the vector x into the vector y.
//   - [Handle.Ddgmm]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Ddgmm64]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Ddot]: This function computes the dot product of vectors x and y.
//   - [Handle.Ddot64]: This function computes the dot product of vectors x and y.
//   - [Handle.Dgbmv]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Dgbmv64]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Dgeam]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.Dgeam64]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.DgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.DgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Dger]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Dger64]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.DgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.DgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.DgetrsBatched]: This function solves an array of systems of linear equations of the form:
//   - [Handle.DmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.DotEx]: These functions support the 64-bit Integer Interface.
//   - [Handle.DotEx64]: These functions support the 64-bit Integer Interface.
//   - [Handle.DotcEx]: These functions support the 64-bit Integer Interface.
//   - [Handle.DotcEx64]: These functions support the 64-bit Integer Interface.
//   - [Handle.Drot]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Drot64]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Drotg]: This function constructs the Givens rotation matrix G=(cs−sc) that zeros out the second entry of a 2×1 vector (a,b)T .
//   - [Handle.Drotm]: This function applies the modified Givens transformation H=(h11h12h21h22) to vectors x and y.
//   - [Handle.Drotm64]: This function applies the modified Givens transformation H=(h11h12h21h22) to vectors x and y.
//   - [Handle.Drotmg]: This function constructs the modified Givens transformation H=(h11h12h21h22) that zeros out the second entry of a 2×1 vector (d1∗x1,d2∗y1)T .
//   - [Handle.Dsbmv]: This function performs the symmetric banded matrix-vector multiplication y=αAx+βy where A is a n×n symmetric banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Dsbmv64]: This function performs the symmetric banded matrix-vector multiplication y=αAx+βy where A is a n×n symmetric banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Dspmv]: This function performs the symmetric packed matrix-vector multiplication y=αAx+βy where A is a n×n symmetric matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Dspmv64]: This function performs the symmetric packed matrix-vector multiplication y=αAx+βy where A is a n×n symmetric matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Dspr2]: This function performs the packed symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Dspr264]: This function performs the packed symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Dspr]: This function performs the packed symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Dspr64]: This function performs the packed symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Dswap]: This function interchanges the elements of vector x and y.
//   - [Handle.Dswap64]: This function interchanges the elements of vector x and y.
//   - [Handle.Dsymm]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Dsymm64]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Dsymv]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Dsymv64]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Dsyr2]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Dsyr264]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Dsyr2k]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Dsyr2k64]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Dsyr]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Dsyr64]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Dsyrk]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Dsyrk64]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Dsyrkx]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Dsyrkx64]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Dtbmv]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Dtbmv64]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Dtbsv]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Dtbsv64]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Dtpmv]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Dtpmv64]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Dtpsv]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Dtpsv64]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Dtpttr]: This function performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Dtrmm]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Dtrmm64]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Dtrmv]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.Dtrmv64]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.Dtrsv]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Dtrsv64]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Dtrttp]: This function performs the conversion from the triangular format to the triangular packed format
//   - [Handle.Dzasum]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.Dzasum64]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.Dznrm2]: This function computes the Euclidean norm of the vector x.
//   - [Handle.Dznrm264]: This function computes the Euclidean norm of the vector x.
//   - [Handle.GemmBatchedEx]: This function is an extension of cublasgemmBatched() that performs the matrix-matrix multiplication of a batch of matrices and allows the user to individually specify the data types for each of the A, B and C matrix arrays, the precision of computation and the GEMM algorithm to be run.
//   - [Handle.GemmBatchedEx64]: This function is an extension of cublasgemmBatched() that performs the matrix-matrix multiplication of a batch of matrices and allows the user to individually specify the data types for each of the A, B and C matrix arrays, the precision of computation and the GEMM algorithm to be run.
//   - [Handle.GemmEx]: This function is an extension of cublasgemm() that allows the user to individually specify the data types for each of the A, B and C matrices, the precision of computation and the GEMM algorithm to be run.
//   - [Handle.GemmEx64]: This function is an extension of cublasgemm() that allows the user to individually specify the data types for each of the A, B and C matrices, the precision of computation and the GEMM algorithm to be run.
//   - [Handle.GemmGroupedBatchedEx]: This function performs the matrix-matrix multiplication on groups of matrices.
//   - [Handle.GemmGroupedBatchedEx64]: This function performs the matrix-matrix multiplication on groups of matrices.
//   - [Handle.GemmStridedBatchedEx]: This function is an extension of cublasgemmStridedBatched() that performs the matrix-matrix multiplication of a batch of matrices and allows the user to individually specify the data types for each of the A, B and C matrices, the precision of computation and the GEMM algorithm to be run.
//   - [Handle.GemmStridedBatchedEx64]: This function is an extension of cublasgemmStridedBatched() that performs the matrix-matrix multiplication of a batch of matrices and allows the user to individually specify the data types for each of the A, B and C matrices, the precision of computation and the GEMM algorithm to be run.
//   - [Handle.GetAtomicsMode]: This function queries the atomic mode of a specific cuBLAS context.
//   - [Handle.GetEmulationSpecialValuesSupport]: This function obtains the value previously programmed to the library handle.
//   - [Handle.GetEmulationStrategy]: This function obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMantissaBitCountPointer]: This function obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMantissaBitOffset]: This function obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMantissaControl]: This function obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMaxMantissaBitCount]: This function obtains the value previously programmed to the library handle.
//   - [GetLoggerCallback]: This function retrieves function pointer to previously installed custom user defined callback function via [SetLoggerCallback] or zero otherwise.
//   - [GetMatrix]: This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space.
//   - [GetMatrixAsync]: This function has the same functionality as [GetMatrix], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [GetMatrixAsync64]: This function has the same functionality as [GetMatrix], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [GetMatrix64]: This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space.
//   - [GetProperty]: This function returns the value of the requested property in memory pointed to by value.
//   - [Handle.GetSmCountTarget]: This function obtains the value previously programmed to the library handle.
//   - [GetVector]: This function copies n elements from a vector x in GPU memory space to a vector y in host memory space.
//   - [GetVectorAsync]: This function has the same functionality as [GetVector], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [GetVectorAsync64]: This function has the same functionality as [GetVector], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [GetVector64]: This function copies n elements from a vector x in GPU memory space to a vector y in host memory space.
//   - [Handle.GetVersion]: This function returns the version number of the cuBLAS library.
//   - [Handle.HSHgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.HSHgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.HSHgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.HSHgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.HSSgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.HSSgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.HSSgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.HSSgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.Hgemm]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.HgemmBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.HgemmBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.HgemmStridedBatched]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.HgemmStridedBatched64]: This function performs the matrix-matrix multiplication of a batch of matrices.
//   - [Handle.Hgemm64]: This function performs the matrix-matrix multiplication C=αop(A)op(B)+βC where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×k , op(B)k×n and Cm×n , respectively.
//   - [Handle.Icamax]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Icamax64]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Icamin]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [Handle.Icamin64]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [Handle.Idamax]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Idamax64]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Idamin]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [Handle.Idamin64]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [Handle.Isamax]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Isamax64]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Isamin]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [Handle.Isamin64]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [Handle.Izamax]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Izamax64]: This function finds the (smallest) index of the element of the maximum magnitude.
//   - [Handle.Izamin]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [Handle.Izamin64]: This function finds the (smallest) index of the element of the minimum magnitude.
//   - [LoggerConfigure]: This function configures logging during runtime.
//   - [Handle.Nrm2Ex]: This function is an API generalization of the routine cublasnrm2() where input data, output data and compute type can be specified independently.
//   - [Handle.Nrm2Ex64]: This function is an API generalization of the routine cublasnrm2() where input data, output data and compute type can be specified independently.
//   - [Handle.RotEx]: This function is an extension to the routine cublasrot() where input data, output data, cosine/sine type, and compute type can be specified independently.
//   - [Handle.RotEx64]: This function is an extension to the routine cublasrot() where input data, output data, cosine/sine type, and compute type can be specified independently.
//   - [Handle.Sasum]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.Sasum64]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.ScalEx]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.ScalEx64]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Scasum]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.Scasum64]: This function computes the sum of the absolute values of the elements of vector x.
//   - [Handle.Scnrm2]: This function computes the Euclidean norm of the vector x.
//   - [Handle.Scnrm264]: This function computes the Euclidean norm of the vector x.
//   - [Handle.Scopy]: This function copies the vector x into the vector y.
//   - [Handle.Scopy64]: This function copies the vector x into the vector y.
//   - [Handle.Sdgmm]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Sdgmm64]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Sdot]: This function computes the dot product of vectors x and y.
//   - [Handle.Sdot64]: This function computes the dot product of vectors x and y.
//   - [Handle.SetAtomicsMode]: Some routines like cublassymv() and cublashemv() have an alternate implementation that use atomics to cumulate results.
//   - [Handle.SetEmulationSpecialValuesSupport]: This function sets the value previously programmed to the library handle.
//   - [Handle.SetEmulationStrategy]: The [Handle.SetEmulationStrategy] function enables you to select how the library should make use of floating point emulation.
//   - [Handle.SetFixedPointEmulationMantissaBitCountPointer]: This function sets the value previously programmed to the library handle.
//   - [Handle.SetFixedPointEmulationMantissaBitOffset]: This function sets the value previously programmed to the library handle.
//   - [Handle.SetFixedPointEmulationMantissaControl]: This function sets the value previously programmed to the library handle.
//   - [Handle.SetFixedPointEmulationMaxMantissaBitCount]: This function sets the value previously programmed to the library handle.
//   - [SetLoggerCallback]: This function installs a custom user defined callback function via cublas C public API.
//   - [SetMatrix]: This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space.
//   - [SetMatrixAsync]: This function has the same functionality as [SetMatrix], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [SetMatrixAsync64]: This function has the same functionality as [SetMatrix], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [SetMatrix64]: This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space.
//   - [Handle.SetSmCountTarget]: The [Handle.SetSmCountTarget] function allows overriding the number of multiprocessors available to the library during kernels execution.
//   - [SetVector]: This function copies n elements from a vector x in host memory space to a vector y in GPU memory space.
//   - [SetVectorAsync]: This function has the same functionality as [SetVector], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [SetVectorAsync64]: This function has the same functionality as [SetVector], with the exception that the data transfer is done asynchronously (with respect to the host) using the given CUDA™ stream parameter.
//   - [SetVector64]: This function copies n elements from a vector x in host memory space to a vector y in GPU memory space.
//   - [Handle.SetWorkspace]: This function sets the cuBLAS library workspace to a user-owned device buffer, which will be used to execute all subsequent calls to the cuBLAS library functions (on the currently set stream).
//   - [Handle.Sgbmv]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Sgbmv64]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Sgeam]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.Sgeam64]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.SgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.SgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Sger]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Sger64]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.SgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.SgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.SgetrsBatched]: This function solves an array of systems of linear equations of the form:
//   - [Handle.SmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.Srot]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Srot64]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Srotg]: This function constructs the Givens rotation matrix G=(cs−sc) that zeros out the second entry of a 2×1 vector (a,b)T .
//   - [Handle.Srotm]: This function applies the modified Givens transformation H=(h11h12h21h22) to vectors x and y.
//   - [Handle.Srotm64]: This function applies the modified Givens transformation H=(h11h12h21h22) to vectors x and y.
//   - [Handle.Srotmg]: This function constructs the modified Givens transformation H=(h11h12h21h22) that zeros out the second entry of a 2×1 vector (d1∗x1,d2∗y1)T .
//   - [Handle.Ssbmv]: This function performs the symmetric banded matrix-vector multiplication y=αAx+βy where A is a n×n symmetric banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Ssbmv64]: This function performs the symmetric banded matrix-vector multiplication y=αAx+βy where A is a n×n symmetric banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Sspmv]: This function performs the symmetric packed matrix-vector multiplication y=αAx+βy where A is a n×n symmetric matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Sspmv64]: This function performs the symmetric packed matrix-vector multiplication y=αAx+βy where A is a n×n symmetric matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Sspr2]: This function performs the packed symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Sspr264]: This function performs the packed symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Sspr]: This function performs the packed symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Sspr64]: This function performs the packed symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Sswap]: This function interchanges the elements of vector x and y.
//   - [Handle.Sswap64]: This function interchanges the elements of vector x and y.
//   - [Handle.Ssymm]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Ssymm64]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Ssymv]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Ssymv64]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Ssyr2]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Ssyr264]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Ssyr2k]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Ssyr2k64]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Ssyr]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Ssyr64]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Ssyrk]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Ssyrk64]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Ssyrkx]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Ssyrkx64]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Stbmv]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Stbmv64]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Stbsv]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Stbsv64]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Stpmv]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Stpmv64]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Stpsv]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Stpsv64]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Stpttr]: This function performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Strmm]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Strmm64]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Strmv]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.Strmv64]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.Strsv]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Strsv64]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Strttp]: This function performs the conversion from the triangular format to the triangular packed format
//   - [Handle.TSSgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.TSSgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.TSSgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.TSSgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.TSTgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.TSTgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.TSTgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.TSTgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.Zaxpy]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Zaxpy64]: This function multiplies the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
//   - [Handle.Zcopy]: This function copies the vector x into the vector y.
//   - [Handle.Zcopy64]: This function copies the vector x into the vector y.
//   - [Handle.Zdgmm]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Zdgmm64]: This function performs the matrix-matrix multiplication C={A×diag(X)if mode == CUBLAS_SIDE_RIGHTdiag(X)×Aif mode == [SideLeft] where A and C are matrices stored in column-major format with dimensions m×n .
//   - [Handle.Zdotc]: This function computes the dot product of vectors x and y.
//   - [Handle.Zdotc64]: This function computes the dot product of vectors x and y.
//   - [Handle.Zdotu]: This function computes the dot product of vectors x and y.
//   - [Handle.Zdotu64]: This function computes the dot product of vectors x and y.
//   - [Handle.Zdrot]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Zdrot64]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Zdscal]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Zdscal64]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Zgbmv]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Zgbmv64]: This function performs the banded matrix-vector multiplication y=α op(A)x+βy where A is a banded matrix with kl subdiagonals and ku superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Zgeam]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.Zgeam64]: This function performs the matrix-matrix addition/transposition C=αop(A)+βop(B) where α and β are scalars, and A , B and C are matrices stored in column-major format with dimensions op(A)m×n , op(B)m×n and Cm×n , respectively.
//   - [Handle.ZgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.ZgemvBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.ZgemvBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.ZgemvStridedBatched]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.ZgemvStridedBatched64]: This function performs the matrix-vector multiplication of a batch of matrices and vectors.
//   - [Handle.Zgemv]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//   - [Handle.Zgemv64]: This function performs the matrix-vector multiplication y=α op(A)x+βy where A is a m×n matrix stored in column-major format, x and y are vectors, and α and β are scalars.
//   - [Handle.ZgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Zgerc]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Zgerc64]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Zgeru]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Zgeru64]: This function performs the rank-1 update A={αxyT+Aif ger(),geru() is calledαxyH+Aif gerc() is called where A is a m×n matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.ZgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.ZgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.ZgetrsBatched]: This function solves an array of systems of linear equations of the form:
//   - [Handle.Zhbmv]: This function performs the Hermitian banded matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Zhbmv64]: This function performs the Hermitian banded matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian banded matrix with k subdiagonals and superdiagonals, x and y are vectors, and α and β are scalars.
//   - [Handle.Zhemm]: This function performs the Hermitian matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a Hermitian matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Zhemm64]: This function performs the Hermitian matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a Hermitian matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Zhemv]: This function performs the Hermitian matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in lower or upper mode, x and y are vectors, and α and β are scalars.
//   - [Handle.Zhemv64]: This function performs the Hermitian matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in lower or upper mode, x and y are vectors, and α and β are scalars.
//   - [Handle.Zher2]: This function performs the Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Zher264]: This function performs the Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Zher2k]: This function performs the Hermitian rank- 2k update C=αop(A)op(B)H+αˉop(B)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Zher2k64]: This function performs the Hermitian rank- 2k update C=αop(A)op(B)H+αˉop(B)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Zher]: This function performs the Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Zher64]: This function performs the Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Zherk]: This function performs the Hermitian rank- k update C=αop(A)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Zherk64]: This function performs the Hermitian rank- k update C=αop(A)op(A)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Zherkx]: This function performs a variation of the Hermitian rank- k update C=αop(A)op(B)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Zherkx64]: This function performs a variation of the Hermitian rank- k update C=αop(A)op(B)H+βC where α and β are scalars, C is a Hermitian matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Zhpmv]: This function performs the Hermitian packed matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Zhpmv64]: This function performs the Hermitian packed matrix-vector multiplication y=αAx+βy where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α and β are scalars.
//   - [Handle.Zhpr2]: This function performs the packed Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α is a scalar.
//   - [Handle.Zhpr264]: This function performs the packed Hermitian rank-2 update A=αxyH+αˉyxH+A where A is a n×n Hermitian matrix stored in packed format, x and y are vectors, and α is a scalar.
//   - [Handle.Zhpr]: This function performs the packed Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.Zhpr64]: This function performs the packed Hermitian rank-1 update A=αxxH+A where A is a n×n Hermitian matrix stored in packed format, x is a vector, and α is a scalar.
//   - [Handle.ZmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.Zrot]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Zrot64]: This function applies Givens rotation matrix (i.e., rotation in the x,y plane counter-clockwise by angle defined by cos(alpha)=c, sin(alpha)=s): G=(cs−sc) to vectors x and y.
//   - [Handle.Zrotg]: This function constructs the Givens rotation matrix G=(cs−sc) that zeros out the second entry of a 2×1 vector (a,b)T .
//   - [Handle.Zscal]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Zscal64]: This function scales the vector x by the scalar α and overwrites it with the result.
//   - [Handle.Zswap]: This function interchanges the elements of vector x and y.
//   - [Handle.Zswap64]: This function interchanges the elements of vector x and y.
//   - [Handle.Zsymm]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Zsymm64]: This function performs the symmetric matrix-matrix multiplication C={αAB+βCif side == [SideLeft]αBA+βCif side == [SideRight] where A is a symmetric matrix stored in lower or upper mode, B and C are m×n matrices, and α and β are scalars.
//   - [Handle.Zsymv]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Zsymv64]: This function performs the symmetric matrix-vector multiplication.
//   - [Handle.Zsyr2]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Zsyr264]: This function performs the symmetric rank-2 update A=α(xyT+yxT)+A where A is a n×n symmetric matrix stored in column-major format, x and y are vectors, and α is a scalar.
//   - [Handle.Zsyr2k]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Zsyr2k64]: This function performs the symmetric rank- 2k update C=α(op(A)op(B)T+op(B)op(A)T)+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Zsyr]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Zsyr64]: This function performs the symmetric rank-1 update A=αxxT+A where A is a n×n symmetric matrix stored in column-major format, x is a vector, and α is a scalar.
//   - [Handle.Zsyrk]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Zsyrk64]: This function performs the symmetric rank- k update C=αop(A)op(A)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A is a matrix with dimensions op(A)n×k .
//   - [Handle.Zsyrkx]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Zsyrkx64]: This function performs a variation of the symmetric rank- k update C=αop(A)op(B)T+βC where α and β are scalars, C is a symmetric matrix stored in lower or upper mode, and A and B are matrices with dimensions op(A)n×k and op(B)n×k , respectively.
//   - [Handle.Ztbmv]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Ztbmv64]: This function performs the triangular banded matrix-vector multiplication x=op(A)x where A is a triangular banded matrix, and x is a vector.
//   - [Handle.Ztbsv]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Ztbsv64]: This function solves the triangular banded linear system with a single right-hand-side op(A)x=b where A is a triangular banded matrix, and x and b are vectors.
//   - [Handle.Ztpmv]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Ztpmv64]: This function performs the triangular packed matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in packed format, and x is a vector.
//   - [Handle.Ztpsv]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Ztpsv64]: This function solves the packed triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in packed format, and x and b are vectors.
//   - [Handle.Ztpttr]: This function performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Ztrmm]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Ztrmm64]: This function performs the triangular matrix-matrix multiplication C={αop(A)Bif side == [SideLeft]αBop(A)if side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, B and C are m×n matrix, and α is a scalar.
//   - [Handle.Ztrmv]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.Ztrmv64]: This function performs the triangular matrix-vector multiplication x=op(A)x where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x is a vector.
//   - [Handle.ZtrsmBatched]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.ZtrsmBatched64]: This function solves an array of triangular linear systems with multiple right-hand-sides {op(Ai)Xi=αBiif side == CUBLAS_SIDE_LEFTXiop(Ai)=αBiif side == [SideRight] where Ai is a triangular matrix stored in lower or upper mode with or without the main diagonal, Xi and Bi are m×n matrices, and α is a scalar.
//   - [Handle.Ztrsm]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.Ztrsm64]: This function solves the triangular linear system with multiple right-hand-sides {op(A)X=αBif side == CUBLAS_SIDE_LEFTXop(A)=αBif side == [SideRight] where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, X and B are m×n matrices, and α is a scalar.
//   - [Handle.Ztrsv]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Ztrsv64]: This function solves the triangular linear system with a single right-hand-side op(A)x=b where A is a triangular matrix stored in lower or upper mode with or without the main diagonal, and x and b are vectors.
//   - [Handle.Ztrttp]: This function performs the conversion from the triangular format to the triangular packed format
//
// [cuBLAS Documentation]: https://docs.nvidia.com/cuda/cublas/
package cublas

// Code generated by gen-nvidia from cublas_api.h. DO NOT EDIT.
