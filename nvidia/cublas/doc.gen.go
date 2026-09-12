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
// This package provides bindings generated from cublas_api.h using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libcublas.
// Functions returning a C status report errors wrapping cublasStatus_t with descriptive messages
// from cublasGetStatusName.
//
// # BLAS3
//
//   - [Handle.Cgemm3m]: supports the 64-bit Integer Interface.
//   - [Handle.Cgemm3mStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.Cgemm3mStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Cgemm3m64]: supports the 64-bit Integer Interface.
//   - [Handle.CgemmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.CgemmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.CgemmEx]: supports the 64-bit Integer Interface.
//   - [Handle.CgemmEx64]: supports the 64-bit Integer Interface.
//   - [Handle.CgemmStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.CgemmStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Cgemm]: supports the 64-bit Integer Interface.
//   - [Handle.Cgemm64]: supports the 64-bit Integer Interface.
//   - [Handle.DgemmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.DgemmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.DgemmGroupedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.DgemmGroupedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.DgemmStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.DgemmStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Dgemm]: supports the 64-bit Integer Interface.
//   - [Handle.Dgemm64]: supports the 64-bit Integer Interface.
//   - [Handle.DtrsmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.DtrsmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrsm]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrsm64]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmEx]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmEx64]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmGroupedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmGroupedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.SgemmStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Sgemm]: supports the 64-bit Integer Interface.
//   - [Handle.Sgemm64]: supports the 64-bit Integer Interface.
//   - [Handle.StrsmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.StrsmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Strsm]: supports the 64-bit Integer Interface.
//   - [Handle.Strsm64]: supports the 64-bit Integer Interface.
//   - [Handle.Zgemm3m]: supports the 64-bit Integer Interface.
//   - [Handle.Zgemm3m64]: supports the 64-bit Integer Interface.
//   - [Handle.ZgemmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.ZgemmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.ZgemmStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.ZgemmStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Zgemm]: supports the 64-bit Integer Interface.
//   - [Handle.Zgemm64]: supports the 64-bit Integer Interface.
//
// # Lifecycle
//
//   - [Create]: initializes the cuBLAS library and creates a handle to an opaque structure holding the cuBLAS library context.
//   - [Handle.Destroy]: releases hardware resources used by the cuBLAS library.
//
// # BLAS1
//
//   - [Handle.Daxpy]: supports the 64-bit Integer Interface.
//   - [Handle.Daxpy64]: supports the 64-bit Integer Interface.
//   - [Handle.Dnrm2]: supports the 64-bit Integer Interface.
//   - [Handle.Dnrm264]: supports the 64-bit Integer Interface.
//   - [Handle.Dscal]: supports the 64-bit Integer Interface.
//   - [Handle.Dscal64]: supports the 64-bit Integer Interface.
//   - [Handle.Saxpy]: supports the 64-bit Integer Interface.
//   - [Handle.Saxpy64]: supports the 64-bit Integer Interface.
//   - [Handle.Snrm2]: supports the 64-bit Integer Interface.
//   - [Handle.Snrm264]: supports the 64-bit Integer Interface.
//   - [Handle.Sscal]: supports the 64-bit Integer Interface.
//   - [Handle.Sscal64]: supports the 64-bit Integer Interface.
//
// # BLAS2
//
//   - [Handle.DgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.DgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.DgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.DgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Dgemv]: supports the 64-bit Integer Interface.
//   - [Handle.Dgemv64]: supports the 64-bit Integer Interface.
//   - [Handle.SgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.SgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.SgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.SgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Sgemv]: supports the 64-bit Integer Interface.
//   - [Handle.Sgemv64]: supports the 64-bit Integer Interface.
//
// # Configuration
//
//   - [Handle.GetMathMode]: returns the math mode used by the library routines.
//   - [Handle.GetPointerMode]: obtains the pointer mode used by the cuBLAS library.
//   - [Handle.SetMathMode]: The [Handle.SetMathMode] function enables you to choose the compute precision modes as defined by [Math].
//   - [Handle.SetPointerMode]: sets the pointer mode used by the cuBLAS library.
//
// # Stream Management
//
//   - [Handle.GetStream]: gets the cuBLAS library stream, which is being used to execute all calls to the cuBLAS library functions.
//   - [Handle.SetStream]: sets the cuBLAS library stream, which will be used to execute all subsequent calls to the cuBLAS library functions.
//
// # Other
//
//   - [Handle.AxpyEx]: supports the 64-bit Integer Interface.
//   - [Handle.AxpyEx64]: supports the 64-bit Integer Interface.
//   - [Handle.Caxpy]: supports the 64-bit Integer Interface.
//   - [Handle.Caxpy64]: supports the 64-bit Integer Interface.
//   - [Handle.Ccopy]: supports the 64-bit Integer Interface.
//   - [Handle.Ccopy64]: supports the 64-bit Integer Interface.
//   - [Handle.Cdgmm]: supports the 64-bit Integer Interface.
//   - [Handle.Cdgmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Cdotc]: supports the 64-bit Integer Interface.
//   - [Handle.Cdotc64]: supports the 64-bit Integer Interface.
//   - [Handle.Cdotu]: supports the 64-bit Integer Interface.
//   - [Handle.Cdotu64]: supports the 64-bit Integer Interface.
//   - [Handle.Cgbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Cgbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Cgeam]: supports the 64-bit Integer Interface.
//   - [Handle.Cgeam64]: supports the 64-bit Integer Interface.
//   - [Handle.CgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.CgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.CgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.CgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.CgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Cgemv]: supports the 64-bit Integer Interface.
//   - [Handle.Cgemv64]: supports the 64-bit Integer Interface.
//   - [Handle.CgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Cgerc]: supports the 64-bit Integer Interface.
//   - [Handle.Cgerc64]: supports the 64-bit Integer Interface.
//   - [Handle.Cgeru]: supports the 64-bit Integer Interface.
//   - [Handle.Cgeru64]: supports the 64-bit Integer Interface.
//   - [Handle.CgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.CgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.CgetrsBatched]: solves an array of systems of linear equations of the form:
//   - [Handle.Chbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Chbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Chemm]: supports the 64-bit Integer Interface.
//   - [Handle.Chemm64]: supports the 64-bit Integer Interface.
//   - [Handle.Chemv]: supports the 64-bit Integer Interface.
//   - [Handle.Chemv64]: supports the 64-bit Integer Interface.
//   - [Handle.Cher2]: supports the 64-bit Integer Interface.
//   - [Handle.Cher264]: supports the 64-bit Integer Interface.
//   - [Handle.Cher2k]: supports the 64-bit Integer Interface.
//   - [Handle.Cher2k64]: supports the 64-bit Integer Interface.
//   - [Handle.Cher]: supports the 64-bit Integer Interface.
//   - [Handle.Cher64]: supports the 64-bit Integer Interface.
//   - [Handle.Cherk3mEx]: supports the 64-bit Integer Interface.
//   - [Handle.Cherk3mEx64]: supports the 64-bit Integer Interface.
//   - [Handle.CherkEx]: supports the 64-bit Integer Interface.
//   - [Handle.CherkEx64]: supports the 64-bit Integer Interface.
//   - [Handle.Cherk]: supports the 64-bit Integer Interface.
//   - [Handle.Cherk64]: supports the 64-bit Integer Interface.
//   - [Handle.Cherkx]: supports the 64-bit Integer Interface.
//   - [Handle.Cherkx64]: supports the 64-bit Integer Interface.
//   - [Handle.Chpmv]: supports the 64-bit Integer Interface.
//   - [Handle.Chpmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Chpr2]: supports the 64-bit Integer Interface.
//   - [Handle.Chpr264]: supports the 64-bit Integer Interface.
//   - [Handle.Chpr]: supports the 64-bit Integer Interface.
//   - [Handle.Chpr64]: supports the 64-bit Integer Interface.
//   - [Handle.CmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.Crot]: supports the 64-bit Integer Interface.
//   - [Handle.Crot64]: supports the 64-bit Integer Interface.
//   - [Handle.Crotg]: supports the 64-bit Integer Interface.
//   - [Handle.Cscal]: supports the 64-bit Integer Interface.
//   - [Handle.Cscal64]: supports the 64-bit Integer Interface.
//   - [Handle.Csrot]: supports the 64-bit Integer Interface.
//   - [Handle.Csrot64]: supports the 64-bit Integer Interface.
//   - [Handle.Csscal]: supports the 64-bit Integer Interface.
//   - [Handle.Csscal64]: supports the 64-bit Integer Interface.
//   - [Handle.Cswap]: supports the 64-bit Integer Interface.
//   - [Handle.Cswap64]: supports the 64-bit Integer Interface.
//   - [Handle.Csymm]: supports the 64-bit Integer Interface.
//   - [Handle.Csymm64]: supports the 64-bit Integer Interface.
//   - [Handle.Csymv]: supports the 64-bit Integer Interface.
//   - [Handle.Csymv64]: supports the 64-bit Integer Interface.
//   - [Handle.Csyr2]: supports the 64-bit Integer Interface.
//   - [Handle.Csyr264]: supports the 64-bit Integer Interface.
//   - [Handle.Csyr2k]: supports the 64-bit Integer Interface.
//   - [Handle.Csyr2k64]: supports the 64-bit Integer Interface.
//   - [Handle.Csyr]: supports the 64-bit Integer Interface.
//   - [Handle.Csyr64]: supports the 64-bit Integer Interface.
//   - [Handle.Csyrk3mEx]: supports the 64-bit Integer Interface.
//   - [Handle.Csyrk3mEx64]: supports the 64-bit Integer Interface.
//   - [Handle.CsyrkEx]: supports the 64-bit Integer Interface.
//   - [Handle.CsyrkEx64]: supports the 64-bit Integer Interface.
//   - [Handle.Csyrk]: supports the 64-bit Integer Interface.
//   - [Handle.Csyrk64]: supports the 64-bit Integer Interface.
//   - [Handle.Csyrkx]: supports the 64-bit Integer Interface.
//   - [Handle.Csyrkx64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Ctbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctbsv]: supports the 64-bit Integer Interface.
//   - [Handle.Ctbsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctpmv]: supports the 64-bit Integer Interface.
//   - [Handle.Ctpmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctpsv]: supports the 64-bit Integer Interface.
//   - [Handle.Ctpsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctpttr]: performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Ctrmm]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrmv]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrmv64]: supports the 64-bit Integer Interface.
//   - [Handle.CtrsmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.CtrsmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrsm]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrsm64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrsv]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ctrttp]: performs the conversion from the triangular format to the triangular packed format
//   - [Handle.Dasum]: supports the 64-bit Integer Interface.
//   - [Handle.Dasum64]: supports the 64-bit Integer Interface.
//   - [Handle.Dcopy]: supports the 64-bit Integer Interface.
//   - [Handle.Dcopy64]: supports the 64-bit Integer Interface.
//   - [Handle.Ddgmm]: supports the 64-bit Integer Interface.
//   - [Handle.Ddgmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Ddot]: supports the 64-bit Integer Interface.
//   - [Handle.Ddot64]: supports the 64-bit Integer Interface.
//   - [Handle.Dgbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Dgbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dgeam]: supports the 64-bit Integer Interface.
//   - [Handle.Dgeam64]: supports the 64-bit Integer Interface.
//   - [Handle.DgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.DgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Dger]: supports the 64-bit Integer Interface.
//   - [Handle.Dger64]: supports the 64-bit Integer Interface.
//   - [Handle.DgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.DgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.DgetrsBatched]: solves an array of systems of linear equations of the form:
//   - [Handle.DmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.DotEx]: These functions support the 64-bit Integer Interface.
//   - [Handle.DotEx64]: These functions support the 64-bit Integer Interface.
//   - [Handle.DotcEx]: These functions support the 64-bit Integer Interface.
//   - [Handle.DotcEx64]: These functions support the 64-bit Integer Interface.
//   - [Handle.Drot]: supports the 64-bit Integer Interface.
//   - [Handle.Drot64]: supports the 64-bit Integer Interface.
//   - [Handle.Drotg]: supports the 64-bit Integer Interface.
//   - [Handle.Drotm]: supports the 64-bit Integer Interface.
//   - [Handle.Drotm64]: supports the 64-bit Integer Interface.
//   - [Handle.Drotmg]: supports the 64-bit Integer Interface.
//   - [Handle.Dsbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Dsbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dspmv]: supports the 64-bit Integer Interface.
//   - [Handle.Dspmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dspr2]: supports the 64-bit Integer Interface.
//   - [Handle.Dspr264]: supports the 64-bit Integer Interface.
//   - [Handle.Dspr]: supports the 64-bit Integer Interface.
//   - [Handle.Dspr64]: supports the 64-bit Integer Interface.
//   - [Handle.Dswap]: supports the 64-bit Integer Interface.
//   - [Handle.Dswap64]: supports the 64-bit Integer Interface.
//   - [Handle.Dsymm]: supports the 64-bit Integer Interface.
//   - [Handle.Dsymm64]: supports the 64-bit Integer Interface.
//   - [Handle.Dsymv]: supports the 64-bit Integer Interface.
//   - [Handle.Dsymv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyr2]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyr264]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyr2k]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyr2k64]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyr]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyr64]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyrk]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyrk64]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyrkx]: supports the 64-bit Integer Interface.
//   - [Handle.Dsyrkx64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Dtbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtbsv]: supports the 64-bit Integer Interface.
//   - [Handle.Dtbsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtpmv]: supports the 64-bit Integer Interface.
//   - [Handle.Dtpmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtpsv]: supports the 64-bit Integer Interface.
//   - [Handle.Dtpsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtpttr]: performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Dtrmm]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrmv]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrsv]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Dtrttp]: performs the conversion from the triangular format to the triangular packed format
//   - [Handle.Dzasum]: supports the 64-bit Integer Interface.
//   - [Handle.Dzasum64]: supports the 64-bit Integer Interface.
//   - [Handle.Dznrm2]: supports the 64-bit Integer Interface.
//   - [Handle.Dznrm264]: supports the 64-bit Integer Interface.
//   - [Handle.GemmBatchedEx]: supports the 64-bit Integer Interface.
//   - [Handle.GemmBatchedEx64]: supports the 64-bit Integer Interface.
//   - [Handle.GemmEx]: supports the 64-bit Integer Interface.
//   - [Handle.GemmEx64]: supports the 64-bit Integer Interface.
//   - [Handle.GemmGroupedBatchedEx]: supports the 64-bit Integer Interface.
//   - [Handle.GemmGroupedBatchedEx64]: supports the 64-bit Integer Interface.
//   - [Handle.GemmStridedBatchedEx]: supports the 64-bit Integer Interface.
//   - [Handle.GemmStridedBatchedEx64]: supports the 64-bit Integer Interface.
//   - [Handle.GetAtomicsMode]: queries the atomic mode of a specific cuBLAS context.
//   - [Handle.GetEmulationSpecialValuesSupport]: obtains the value previously programmed to the library handle.
//   - [Handle.GetEmulationStrategy]: obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMantissaBitCountPointer]: obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMantissaBitOffset]: obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMantissaControl]: obtains the value previously programmed to the library handle.
//   - [Handle.GetFixedPointEmulationMaxMantissaBitCount]: obtains the value previously programmed to the library handle.
//   - [GetLoggerCallback]: retrieves function pointer to previously installed custom user defined callback function via [SetLoggerCallback] or zero otherwise.
//   - [GetMatrix]: supports the 64-bit Integer Interface.
//   - [GetMatrixAsync]: supports the 64-bit Integer Interface.
//   - [GetMatrixAsync64]: supports the 64-bit Integer Interface.
//   - [GetMatrix64]: supports the 64-bit Integer Interface.
//   - [GetProperty]: returns the value of the requested property in memory pointed to by value.
//   - [Handle.GetSmCountTarget]: obtains the value previously programmed to the library handle.
//   - [GetVector]: supports the 64-bit Integer Interface.
//   - [GetVectorAsync]: supports the 64-bit Integer Interface.
//   - [GetVectorAsync64]: supports the 64-bit Integer Interface.
//   - [GetVector64]: supports the 64-bit Integer Interface.
//   - [Handle.GetVersion]: returns the version number of the cuBLAS library.
//   - [Handle.HSHgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.HSHgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.HSHgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.HSHgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.HSSgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.HSSgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.HSSgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.HSSgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Hgemm]: supports the 64-bit Integer Interface.
//   - [Handle.HgemmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.HgemmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.HgemmStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.HgemmStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Hgemm64]: supports the 64-bit Integer Interface.
//   - [Handle.Icamax]: supports the 64-bit Integer Interface.
//   - [Handle.Icamax64]: supports the 64-bit Integer Interface.
//   - [Handle.Icamin]: supports the 64-bit Integer Interface.
//   - [Handle.Icamin64]: supports the 64-bit Integer Interface.
//   - [Handle.Idamax]: supports the 64-bit Integer Interface.
//   - [Handle.Idamax64]: supports the 64-bit Integer Interface.
//   - [Handle.Idamin]: supports the 64-bit Integer Interface.
//   - [Handle.Idamin64]: supports the 64-bit Integer Interface.
//   - [Handle.Isamax]: supports the 64-bit Integer Interface.
//   - [Handle.Isamax64]: supports the 64-bit Integer Interface.
//   - [Handle.Isamin]: supports the 64-bit Integer Interface.
//   - [Handle.Isamin64]: supports the 64-bit Integer Interface.
//   - [Handle.Izamax]: supports the 64-bit Integer Interface.
//   - [Handle.Izamax64]: supports the 64-bit Integer Interface.
//   - [Handle.Izamin]: supports the 64-bit Integer Interface.
//   - [Handle.Izamin64]: supports the 64-bit Integer Interface.
//   - [LoggerConfigure]: configures logging during runtime.
//   - [Handle.Nrm2Ex]: supports the 64-bit Integer Interface.
//   - [Handle.Nrm2Ex64]: supports the 64-bit Integer Interface.
//   - [Handle.RotEx]: supports the 64-bit Integer Interface.
//   - [Handle.RotEx64]: supports the 64-bit Integer Interface.
//   - [Handle.Sasum]: supports the 64-bit Integer Interface.
//   - [Handle.Sasum64]: supports the 64-bit Integer Interface.
//   - [Handle.ScalEx]: supports the 64-bit Integer Interface.
//   - [Handle.ScalEx64]: supports the 64-bit Integer Interface.
//   - [Handle.Scasum]: supports the 64-bit Integer Interface.
//   - [Handle.Scasum64]: supports the 64-bit Integer Interface.
//   - [Handle.Scnrm2]: supports the 64-bit Integer Interface.
//   - [Handle.Scnrm264]: supports the 64-bit Integer Interface.
//   - [Handle.Scopy]: supports the 64-bit Integer Interface.
//   - [Handle.Scopy64]: supports the 64-bit Integer Interface.
//   - [Handle.Sdgmm]: supports the 64-bit Integer Interface.
//   - [Handle.Sdgmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Sdot]: supports the 64-bit Integer Interface.
//   - [Handle.Sdot64]: supports the 64-bit Integer Interface.
//   - [Handle.SetAtomicsMode]: Some routines like cublassymv() and cublashemv() have an alternate implementation that use atomics to cumulate results.
//   - [Handle.SetEmulationSpecialValuesSupport]: sets the value previously programmed to the library handle.
//   - [Handle.SetEmulationStrategy]: The [Handle.SetEmulationStrategy] function enables you to select how the library should make use of floating point emulation.
//   - [Handle.SetFixedPointEmulationMantissaBitCountPointer]: sets the value previously programmed to the library handle.
//   - [Handle.SetFixedPointEmulationMantissaBitOffset]: sets the value previously programmed to the library handle.
//   - [Handle.SetFixedPointEmulationMantissaControl]: sets the value previously programmed to the library handle.
//   - [Handle.SetFixedPointEmulationMaxMantissaBitCount]: sets the value previously programmed to the library handle.
//   - [SetLoggerCallback]: installs a custom user defined callback function via cublas C public API.
//   - [SetMatrix]: supports the 64-bit Integer Interface.
//   - [SetMatrixAsync]: supports the 64-bit Integer Interface.
//   - [SetMatrixAsync64]: supports the 64-bit Integer Interface.
//   - [SetMatrix64]: supports the 64-bit Integer Interface.
//   - [Handle.SetSmCountTarget]: The [Handle.SetSmCountTarget] function allows overriding the number of multiprocessors available to the library during kernels execution.
//   - [SetVector]: supports the 64-bit Integer Interface.
//   - [SetVectorAsync]: supports the 64-bit Integer Interface.
//   - [SetVectorAsync64]: supports the 64-bit Integer Interface.
//   - [SetVector64]: supports the 64-bit Integer Interface.
//   - [Handle.SetWorkspace]: sets the cuBLAS library workspace to a user-owned device buffer, which will be used to execute all subsequent calls to the cuBLAS library functions (on the currently set stream).
//   - [Handle.Sgbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Sgbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Sgeam]: supports the 64-bit Integer Interface.
//   - [Handle.Sgeam64]: supports the 64-bit Integer Interface.
//   - [Handle.SgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.SgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Sger]: supports the 64-bit Integer Interface.
//   - [Handle.Sger64]: supports the 64-bit Integer Interface.
//   - [Handle.SgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.SgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.SgetrsBatched]: solves an array of systems of linear equations of the form:
//   - [Handle.SmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.Srot]: supports the 64-bit Integer Interface.
//   - [Handle.Srot64]: supports the 64-bit Integer Interface.
//   - [Handle.Srotg]: supports the 64-bit Integer Interface.
//   - [Handle.Srotm]: supports the 64-bit Integer Interface.
//   - [Handle.Srotm64]: supports the 64-bit Integer Interface.
//   - [Handle.Srotmg]: supports the 64-bit Integer Interface.
//   - [Handle.Ssbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Ssbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Sspmv]: supports the 64-bit Integer Interface.
//   - [Handle.Sspmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Sspr2]: supports the 64-bit Integer Interface.
//   - [Handle.Sspr264]: supports the 64-bit Integer Interface.
//   - [Handle.Sspr]: supports the 64-bit Integer Interface.
//   - [Handle.Sspr64]: supports the 64-bit Integer Interface.
//   - [Handle.Sswap]: supports the 64-bit Integer Interface.
//   - [Handle.Sswap64]: supports the 64-bit Integer Interface.
//   - [Handle.Ssymm]: supports the 64-bit Integer Interface.
//   - [Handle.Ssymm64]: supports the 64-bit Integer Interface.
//   - [Handle.Ssymv]: supports the 64-bit Integer Interface.
//   - [Handle.Ssymv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyr2]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyr264]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyr2k]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyr2k64]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyr]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyr64]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyrk]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyrk64]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyrkx]: supports the 64-bit Integer Interface.
//   - [Handle.Ssyrkx64]: supports the 64-bit Integer Interface.
//   - [Handle.Stbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Stbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Stbsv]: supports the 64-bit Integer Interface.
//   - [Handle.Stbsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Stpmv]: supports the 64-bit Integer Interface.
//   - [Handle.Stpmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Stpsv]: supports the 64-bit Integer Interface.
//   - [Handle.Stpsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Stpttr]: performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Strmm]: supports the 64-bit Integer Interface.
//   - [Handle.Strmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Strmv]: supports the 64-bit Integer Interface.
//   - [Handle.Strmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Strsv]: supports the 64-bit Integer Interface.
//   - [Handle.Strsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Strttp]: performs the conversion from the triangular format to the triangular packed format
//   - [Handle.TSSgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.TSSgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.TSSgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.TSSgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.TSTgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.TSTgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.TSTgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.TSTgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Zaxpy]: supports the 64-bit Integer Interface.
//   - [Handle.Zaxpy64]: supports the 64-bit Integer Interface.
//   - [Handle.Zcopy]: supports the 64-bit Integer Interface.
//   - [Handle.Zcopy64]: supports the 64-bit Integer Interface.
//   - [Handle.Zdgmm]: supports the 64-bit Integer Interface.
//   - [Handle.Zdgmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Zdotc]: supports the 64-bit Integer Interface.
//   - [Handle.Zdotc64]: supports the 64-bit Integer Interface.
//   - [Handle.Zdotu]: supports the 64-bit Integer Interface.
//   - [Handle.Zdotu64]: supports the 64-bit Integer Interface.
//   - [Handle.Zdrot]: supports the 64-bit Integer Interface.
//   - [Handle.Zdrot64]: supports the 64-bit Integer Interface.
//   - [Handle.Zdscal]: supports the 64-bit Integer Interface.
//   - [Handle.Zdscal64]: supports the 64-bit Integer Interface.
//   - [Handle.Zgbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Zgbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Zgeam]: supports the 64-bit Integer Interface.
//   - [Handle.Zgeam64]: supports the 64-bit Integer Interface.
//   - [Handle.ZgelsBatched]: Aarray is an array of pointers to matrices stored in column-major format.
//   - [Handle.ZgemvBatched]: supports the 64-bit Integer Interface.
//   - [Handle.ZgemvBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.ZgemvStridedBatched]: supports the 64-bit Integer Interface.
//   - [Handle.ZgemvStridedBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Zgemv]: supports the 64-bit Integer Interface.
//   - [Handle.Zgemv64]: supports the 64-bit Integer Interface.
//   - [Handle.ZgeqrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions m x n and leading dimension lda.
//   - [Handle.Zgerc]: supports the 64-bit Integer Interface.
//   - [Handle.Zgerc64]: supports the 64-bit Integer Interface.
//   - [Handle.Zgeru]: supports the 64-bit Integer Interface.
//   - [Handle.Zgeru64]: supports the 64-bit Integer Interface.
//   - [Handle.ZgetrfBatched]: Aarray is an array of pointers to matrices stored in column-major format with dimensions nxn and leading dimension lda.
//   - [Handle.ZgetriBatched]: Aarray and Carray are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and ldc respectively.
//   - [Handle.ZgetrsBatched]: solves an array of systems of linear equations of the form:
//   - [Handle.Zhbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Zhbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Zhemm]: supports the 64-bit Integer Interface.
//   - [Handle.Zhemm64]: supports the 64-bit Integer Interface.
//   - [Handle.Zhemv]: supports the 64-bit Integer Interface.
//   - [Handle.Zhemv64]: supports the 64-bit Integer Interface.
//   - [Handle.Zher2]: supports the 64-bit Integer Interface.
//   - [Handle.Zher264]: supports the 64-bit Integer Interface.
//   - [Handle.Zher2k]: supports the 64-bit Integer Interface.
//   - [Handle.Zher2k64]: supports the 64-bit Integer Interface.
//   - [Handle.Zher]: supports the 64-bit Integer Interface.
//   - [Handle.Zher64]: supports the 64-bit Integer Interface.
//   - [Handle.Zherk]: supports the 64-bit Integer Interface.
//   - [Handle.Zherk64]: supports the 64-bit Integer Interface.
//   - [Handle.Zherkx]: supports the 64-bit Integer Interface.
//   - [Handle.Zherkx64]: supports the 64-bit Integer Interface.
//   - [Handle.Zhpmv]: supports the 64-bit Integer Interface.
//   - [Handle.Zhpmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Zhpr2]: supports the 64-bit Integer Interface.
//   - [Handle.Zhpr264]: supports the 64-bit Integer Interface.
//   - [Handle.Zhpr]: supports the 64-bit Integer Interface.
//   - [Handle.Zhpr64]: supports the 64-bit Integer Interface.
//   - [Handle.ZmatinvBatched]: A and Ainv are arrays of pointers to matrices stored in column-major format with dimensions n*n and leading dimension lda and lda_inv respectively.
//   - [Handle.Zrot]: supports the 64-bit Integer Interface.
//   - [Handle.Zrot64]: supports the 64-bit Integer Interface.
//   - [Handle.Zrotg]: supports the 64-bit Integer Interface.
//   - [Handle.Zscal]: supports the 64-bit Integer Interface.
//   - [Handle.Zscal64]: supports the 64-bit Integer Interface.
//   - [Handle.Zswap]: supports the 64-bit Integer Interface.
//   - [Handle.Zswap64]: supports the 64-bit Integer Interface.
//   - [Handle.Zsymm]: supports the 64-bit Integer Interface.
//   - [Handle.Zsymm64]: supports the 64-bit Integer Interface.
//   - [Handle.Zsymv]: supports the 64-bit Integer Interface.
//   - [Handle.Zsymv64]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyr2]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyr264]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyr2k]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyr2k64]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyr]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyr64]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyrk]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyrk64]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyrkx]: supports the 64-bit Integer Interface.
//   - [Handle.Zsyrkx64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztbmv]: supports the 64-bit Integer Interface.
//   - [Handle.Ztbmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztbsv]: supports the 64-bit Integer Interface.
//   - [Handle.Ztbsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztpmv]: supports the 64-bit Integer Interface.
//   - [Handle.Ztpmv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztpsv]: supports the 64-bit Integer Interface.
//   - [Handle.Ztpsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztpttr]: performs the conversion from the triangular packed format to the triangular format
//   - [Handle.Ztrmm]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrmm64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrmv]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrmv64]: supports the 64-bit Integer Interface.
//   - [Handle.ZtrsmBatched]: supports the 64-bit Integer Interface.
//   - [Handle.ZtrsmBatched64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrsm]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrsm64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrsv]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrsv64]: supports the 64-bit Integer Interface.
//   - [Handle.Ztrttp]: performs the conversion from the triangular format to the triangular packed format
//
// [cuBLAS Documentation]: https://docs.nvidia.com/cuda/cublas/
package cublas

// Code generated by gen-nvidia from cublas_api.h. DO NOT EDIT.
