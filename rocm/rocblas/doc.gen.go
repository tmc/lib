// Package rocblas provides Go bindings for rocBLAS.
//
// rocBLAS implements BLAS (Basic Linear Algebra Subprograms) optimized for AMD GPUs.
// These bindings use purego for dynamic loading without cgo.
//
// # Other
//
//   - [Handle.AxpyBatchedEx]: BLAS EX API
//   - [Handle.AxpyEx]: BLAS EX API
//   - [Handle.AxpyStridedBatchedEx]: BLAS EX API
//   - [Handle.Chbmv]: hbmv performs the matrix-vector operations:
//   - [Handle.ChbmvBatched]: hbmv_batched performs one of the matrix-vector operations:
//   - [Handle.ChbmvStridedBatched]: hbmv_strided_batched performs one of the matrix-vector operations:
//   - [Handle.Chemm]: hemm performs one of the matrix-matrix operations:
//   - [Handle.ChemmBatched]: hemm_batched performs a batch of the matrix-matrix operations:
//   - [Handle.ChemmStridedBatched]: hemm_strided_batched performs a batch of the matrix-matrix operations:
//   - [Handle.Chemv]: hemv performs one of the matrix-vector operations:
//   - [Handle.ChemvBatched]: hemv_batched performs one of the matrix-vector operations:
//   - [Handle.ChemvStridedBatched]: hemv_strided_batched performs one of the matrix-vector operations:
//   - [Handle.Cher]: her performs the matrix-vector operations:
//   - [Handle.Cher2]: her2 performs the matrix-vector operations:
//   - [Handle.Cher2Batched]: her2_batched performs the matrix-vector operations:
//   - [Handle.Cher2StridedBatched]: her2_strided_batched performs the matrix-vector operations:
//   - [Handle.Cher2k]: her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update:
//   - [Handle.Cher2kBatched]: her2k_batched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update:
//   - [Handle.Cher2kStridedBatched]: her2k_strided_batched performs a batch of the matrix-matrix operations for a Hermitian rank-2k...
//   - [Handle.CherBatched]: her_batched performs the matrix-vector operations:
//   - [Handle.CherStridedBatched]: her_strided_batched performs the matrix-vector operations:
//   - [Handle.Cherk]: herk performs one of the matrix-matrix operations for a Hermitian rank-k update:
//   - [Handle.CherkBatched]: herk_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k update:
//   - [Handle.CherkStridedBatched]: herk_strided_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k update:
//   - [Handle.Cherkx]: herkx performs one of the matrix-matrix operations for a Hermitian rank-k update:
//   - [Handle.CherkxBatched]: herkx_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k update:
//   - [Handle.CherkxStridedBatched]: herkx_strided_batched performs a batch of the matrix-matrix operations for a Hermitian rank-k...
//   - [Handle.Chpmv]: hpmv performs the matrix-vector operation:
//   - [Handle.ChpmvBatched]: hpmv_batched performs the matrix-vector operation:
//   - [Handle.ChpmvStridedBatched]: hpmv_strided_batched performs the matrix-vector operation:
//   - [Handle.Chpr]: hpr performs the matrix-vector operations:
//   - [Handle.Chpr2]: hpr2 performs the matrix-vector operations:
//   - [Handle.Chpr2Batched]: hpr2_batched performs the matrix-vector operations:
//   - [Handle.Chpr2StridedBatched]: hpr2_strided_batched performs the matrix-vector operations:
//   - [Handle.ChprBatched]: hpr_batched performs the matrix-vector operations:
//   - [Handle.ChprStridedBatched]: hpr_strided_batched performs the matrix-vector operations:
//   - [Handle.Destroy]: Destroy handle
//   - [Handle.DotBatchedEx]: BLAS EX API
//   - [Handle.DotEx]: BLAS EX API
//   - [Handle.DotStridedBatchedEx]: BLAS EX API
//   - [Handle.GeamEx]: BLAS EX API
//   - [Handle.GemmBatchedEx]: BLAS EX API
//   - [Handle.GemmEx]: BLAS EX API
//   - [Handle.GemmStridedBatchedEx]: BLAS EX API
//   - [Handle.Haxpy]: axpy computes constant alpha multiplied by vector x, plus vector y:
//   - [Handle.HaxpyBatched]: axpy_batched compute y := alpha * x + y over a set of batched vectors.
//   - [Handle.HaxpyStridedBatched]: axpy_strided_batched compute y := alpha * x + y over a set of strided batched vectors.
//   - [Handle.Isamax]: amax finds the first index of the element of maximum magnitude of a vector x.
//   - [Handle.IsamaxBatched]: amax_batched finds the first index of the element of maximum magnitude of each vector x_i in a...
//   - [Handle.IsamaxStridedBatched]: amax_strided_batched finds the first index of the element of maximum magnitude of each vector x_i...
//   - [Handle.Isamin]: amin finds the first index of the element of minimum magnitude of a vector x.
//   - [Handle.IsaminBatched]: amin_batched finds the first index of the element of minimum magnitude of each vector x_i in a...
//   - [Handle.IsaminStridedBatched]: amin_strided_batched finds the first index of the element of minimum magnitude of each vector x_i...
//   - [Handle.Nrm2BatchedEx]: BLAS_EX API
//   - [Handle.Nrm2Ex]: BLAS_EX API
//   - [Handle.Nrm2StridedBatchedEx]: BLAS_EX API
//   - [Handle.RotBatchedEx]: BLAS EX API
//   - [Handle.RotEx]: BLAS EX API
//   - [Handle.RotStridedBatchedEx]: rot_strided_batched_ex applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha)...
//   - [Handle.Sasum]: asum computes the sum of the magnitudes of elements of a real vector x,
//   - [Handle.SasumBatched]: asum_batched computes the sum of the magnitudes of the elements in a batch of real vectors x_i,
//   - [Handle.SasumStridedBatched]: asum_strided_batched computes the sum of the magnitudes of elements of a real vectors x_i,
//   - [Handle.ScalBatchedEx]: BLAS EX API
//   - [Handle.ScalEx]: BLAS EX API
//   - [Handle.ScalStridedBatchedEx]: BLAS EX API
//   - [Handle.Scopy]: copy copies each element x [i] into y [i], for i = 1 , ...
//   - [Handle.ScopyBatched]: copy_batched copies each element x_i [j] into y_i [j], for j = 1 , ...
//   - [Handle.ScopyStridedBatched]: copy_strided_batched copies each element x_i [j] into y_i [j], for j = 1 , ...
//   - [Handle.Sdgmm]: dgmm performs one of the matrix-matrix operations:
//   - [Handle.SdgmmBatched]: dgmm_batched performs one of the batched matrix-matrix operations:
//   - [Handle.SdgmmStridedBatched]: dgmm_strided_batched performs one of the batched matrix-matrix operations:
//   - [Handle.Sgbmv]: gbmv performs one of the matrix-vector operations:
//   - [Handle.SgbmvBatched]: gbmv_batched performs one of the matrix-vector operations:
//   - [Handle.SgbmvStridedBatched]: gbmv_strided_batched performs one of the matrix-vector operations:
//   - [Handle.Sgeam]: geam performs one of the matrix-matrix operations:
//   - [Handle.SgeamBatched]: geam_batched performs one of the batched matrix-matrix operations:
//   - [Handle.SgeamStridedBatched]: geam_strided_batched performs one of the batched matrix-matrix operations:
//   - [Handle.Sgemv]: gemv performs one of the matrix-vector operations:
//   - [Handle.SgemvBatched]: gemv_batched performs a batch of matrix-vector operations:
//   - [Handle.SgemvStridedBatched]: gemv_strided_batched performs a batch of matrix-vector operations:
//   - [Handle.Sger]: ger,geru,gerc performs the matrix-vector operations:
//   - [Handle.SgerBatched]: ger_batched,geru_batched,gerc_batched perform a batch of the matrix-vector operations:
//   - [Handle.SgerStridedBatched]: ger_strided_batched,geru_strided_batched,gerc_strided_batched performs the matrix-vector operations:
//   - [Handle.Srot]: rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
//   - [Handle.SrotBatched]: rot_batched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched...
//   - [Handle.SrotStridedBatched]: rot_strided_batched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to...
//   - [Handle.Srotg]: rotg creates the Givens rotation matrix for the vector (a b).
//   - [Handle.SrotgBatched]: rotg_batched creates the Givens rotation matrix for the batched vectors (a_i b_i), for i = 1, ...,...
//   - [Handle.SrotgStridedBatched]: rotg_strided_batched creates the Givens rotation matrix for the strided batched vectors (a_i b_i),...
//   - [Handle.Srotm]: rotm applies the modified Givens rotation matrix defined by param to vectors x and y.
//   - [Handle.SrotmBatched]: rotm_batched applies the modified Givens rotation matrix defined by param_i to batched vectors x_i...
//   - [Handle.SrotmStridedBatched]: rotm_strided_batched applies the modified Givens rotation matrix defined by param_i to strided...
//   - [Handle.Srotmg]: rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
//   - [Handle.SrotmgBatched]: rotmg_batched creates the modified Givens rotation matrix for the batched vectors (d1_i * x1_i,...
//   - [Handle.SrotmgStridedBatched]: rotmg_strided_batched creates the modified Givens rotation matrix for the strided batched vectors...
//   - [Handle.Ssbmv]: sbmv performs the matrix-vector operation:
//   - [Handle.SsbmvBatched]: sbmv_batched performs the matrix-vector operation:
//   - [Handle.SsbmvStridedBatched]: sbmv_strided_batched performs the matrix-vector operation:
//   - [Handle.Sspmv]: spmv performs the matrix-vector operation:
//   - [Handle.SspmvBatched]: spmv_batched performs the matrix-vector operation:
//   - [Handle.SspmvStridedBatched]: spmv_strided_batched performs the matrix-vector operation:
//   - [Handle.Sspr]: spr performs the matrix-vector operations:
//   - [Handle.Sspr2]: spr2 performs the matrix-vector operation:
//   - [Handle.Sspr2Batched]: spr2_batched performs the matrix-vector operation:
//   - [Handle.Sspr2StridedBatched]: spr2_strided_batched performs the matrix-vector operation:
//   - [Handle.SsprBatched]: spr_batched performs the matrix-vector operations:
//   - [Handle.SsprStridedBatched]: spr_strided_batched performs the matrix-vector operations:
//   - [Handle.Sswap]: swap interchanges vectors x and y:
//   - [Handle.SswapBatched]: swap_batched interchanges vectors x_i and y_i, for i = 1 , ...
//   - [Handle.SswapStridedBatched]: swap_strided_batched interchanges vectors x_i and y_i, for i = 1 , ...
//   - [Handle.Ssymm]: symm performs one of the matrix-matrix operations:
//   - [Handle.SsymmBatched]: symm_batched performs a batch of the matrix-matrix operations:
//   - [Handle.SsymmStridedBatched]: symm_strided_batched performs a batch of the matrix-matrix operations:
//   - [Handle.Ssymv]: symv performs the matrix-vector operation:
//   - [Handle.SsymvBatched]: symv_batched performs the matrix-vector operation:
//   - [Handle.SsymvStridedBatched]: symv_strided_batched performs the matrix-vector operation:
//   - [Handle.Ssyr]: syr performs the matrix-vector operations:
//   - [Handle.Ssyr2]: syr2 performs the matrix-vector operations:
//   - [Handle.Ssyr2Batched]: syr2_batched performs a batch of matrix-vector operations:
//   - [Handle.Ssyr2StridedBatched]: syr2_strided_batched the matrix-vector operations:
//   - [Handle.Ssyr2k]: syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update:
//   - [Handle.Ssyr2kBatched]: syr2k_batched performs a batch of the matrix-matrix operations for a symmetric rank-2k update:
//   - [Handle.Ssyr2kStridedBatched]: syr2k_strided_batched performs a batch of the matrix-matrix operations for a symmetric rank-2k...
//   - [Handle.SsyrBatched]: syr_batched performs a batch of matrix-vector operations:
//   - [Handle.SsyrStridedBatched]: syr_strided_batched performs the matrix-vector operations:
//   - [Handle.Ssyrk]: syrk performs one of the matrix-matrix operations for a symmetric rank-k update:
//   - [Handle.SsyrkBatched]: syrk_batched performs a batch of the matrix-matrix operations for a symmetric rank-k update:
//   - [Handle.SsyrkStridedBatched]: syrk_strided_batched performs a batch of the matrix-matrix operations for a symmetric rank-k update:
//   - [Handle.Ssyrkx]: syrkx performs one of the matrix-matrix operations for a symmetric rank-k update:
//   - [Handle.SsyrkxBatched]: syrkx_batched performs a batch of the matrix-matrix operations for a symmetric rank-k update:
//   - [Handle.SsyrkxStridedBatched]: syrkx_strided_batched performs a batch of the matrix-matrix operations for a symmetric rank-k...
//   - [Handle.Stbmv]: tbmv performs one of the matrix-vector operations:
//   - [Handle.StbmvBatched]: tbmv_batched performs one of the matrix-vector operations:
//   - [Handle.StbmvStridedBatched]: tbmv_strided_batched performs one of the matrix-vector operations:
//   - [Handle.Stbsv]: tbsv solves:
//   - [Handle.StbsvBatched]: tbsv_batched solves:
//   - [Handle.StbsvStridedBatched]: tbsv_strided_batched solves:
//   - [Handle.Stpmv]: tpmv performs one of the matrix-vector operations:
//   - [Handle.StpmvBatched]: tpmv_batched performs one of the matrix-vector operations:
//   - [Handle.StpmvStridedBatched]: tpmv_strided_batched performs one of the matrix-vector operations:
//   - [Handle.Stpsv]: tpsv solves:
//   - [Handle.StpsvBatched]: tpsv_batched solves:
//   - [Handle.StpsvStridedBatched]: tpsv_strided_batched solves:
//   - [Handle.Strmm]: trmm performs one of the matrix-matrix operations:
//   - [Handle.StrmmBatched]: trmm_batched performs one of the matrix-matrix operations:
//   - [Handle.StrmmStridedBatched]: trmm_strided_batched performs one of the matrix-matrix operations:
//   - [Handle.Strmv]: trmv performs one of the matrix-vector operations:
//   - [Handle.StrmvBatched]: trmv_batched performs one of the matrix-vector operations:
//   - [Handle.StrmvStridedBatched]: trmv_strided_batched performs one of the matrix-vector operations:
//   - [Handle.Strsm]: trsm solves:
//   - [Handle.StrsmBatched]: trsm_batched performs the following batched operation:
//   - [Handle.StrsmStridedBatched]: trsm_srided_batched performs the following strided batched operation:
//   - [Handle.Strsv]: trsv solves:
//   - [Handle.StrsvBatched]: trsv_batched solves:
//   - [Handle.StrsvStridedBatched]: trsv_strided_batched solves:
//   - [Handle.Strtri]: trtri compute the inverse of a matrix A, namely, invA
//   - [Handle.StrtriBatched]: trtri_batched compute the inverse of A_i and write into invA_i where
//   - [Handle.StrtriStridedBatched]: trtri_strided_batched compute the inverse of A_i and write into invA_i where
//   - [Handle.TrsmBatchedEx]: BLAS EX API
//   - [Handle.TrsmEx]: BLAS EX API
//   - [Handle.TrsmStridedBatchedEx]: BLAS EX API
//
// # Handle Management
//
//   - [NewHandle]: Create handle
//
// # Queries
//
//   - [Handle.GetAtomicsMode]: Get [AtomicsMode]
//   - [Handle.GetMathMode]: Get [MathMode]
//   - [GetMatrix]: Copy matrix from device to host
//   - [Stream.GetMatrixAsync]: asynchronously copy matrix from device to host [Stream.GetMatrixAsync] copies a matrix from device...
//   - [Handle.GetPerformanceMetric]: returns the performance metric being used for solution selection Returns the performance metric...
//   - [Handle.GetPointerMode]: Get [PointerMode]
//   - [Handle.GetStream]: Get stream [0] from handle
//   - [GetVector]: Copy vector from device to host
//   - [Stream.GetVectorAsync]: Asynchronously copy vector from device to host [Stream.GetVectorAsync] copies a vector from pinned...
//   - [GetVersionString]: Loads char* buf with the rocblas library version.
//   - [GetVersionStringSize]: Queries the minimum buffer size for a successful call to [[GetVersionString]]
//
// # Dot Product Operations
//
//   - [Handle.Sdot]: dot(u) performs the dot product of vectors x and y:
//   - [Handle.SdotBatched]: dot_batched(u) performs a batch of dot products of vectors x and y:
//   - [Handle.SdotStridedBatched]: dot_strided_batched(u) performs a batch of dot products of vectors x and y:
//
// # Configuration
//
//   - [Handle.SetMathMode]: Set [MathMode]
//   - [SetMatrix]: Copy matrix from host to device
//   - [Stream.SetMatrixAsync]: Asynchronously copy matrix from host to device [Stream.SetMatrixAsync] copies a matrix from pinned...
//   - [Handle.SetPerformanceMetric]: specifies the performance metric that solution selection uses Determines which performance metric...
//   - [Handle.SetPointerMode]: Set [PointerMode]
//   - [Handle.SetStream]: Set stream for handle
//   - [SetVector]: Copy vector from host to device
//   - [Stream.SetVectorAsync]: Asynchronously copy vector from host to device [Stream.SetVectorAsync] copies a vector from pinned...
//
// # GEMM Operations
//
//   - [Handle.Sgemm]: gemm performs one of the matrix-matrix operations:
//   - [Handle.SgemmBatched]: gemm_batched performs one of the batched matrix-matrix operations:
//   - [Handle.SgemmStridedBatched]: gemm_strided_batched performs one of the strided batched matrix-matrix operations:
//   - [Handle.Sgemmt]: gemmt performs matrix-matrix operations and updates the upper or lower triangular part of the...
//   - [Handle.SgemmtBatched]: gemmt_batched performs matrix-matrix operations and updates the upper or lower triangular part of...
//   - [Handle.SgemmtStridedBatched]: gemmt_strided_batched performs matrix-matrix operations and updates the upper or lower triangular...
//
// # Norm Operations
//
//   - [Handle.Snrm2]: nrm2 computes the euclidean norm of a real or complex vector:
//   - [Handle.Snrm2Batched]: nrm2_batched computes the euclidean norm over a batch of real or complex vectors:
//   - [Handle.Snrm2StridedBatched]: nrm2_strided_batched computes the euclidean norm over a batch of real or complex vectors:
//
// # Scale Operations
//
//   - [Handle.Sscal]: scal scales each element of vector x with scalar alpha:
//   - [Handle.SscalBatched]: scal_batched scales each element of vector x_i with scalar alpha, for i = 1, ...
//   - [Handle.SscalStridedBatched]: scal_strided_batched scales each element of vector x_i with scalar alpha, for i = 1, ...
//
// [rocBLAS Documentation]: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/
package rocblas

// Code generated by gen-rocm from rocblas-functions.h. DO NOT EDIT.
