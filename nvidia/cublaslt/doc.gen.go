// Package cublaslt provides low-level Go bindings for cuBLASLt (CUDA Linear Algebra Subroutines Lightweight).
//
// cuBLASLt is a lightweight matrix multiplication library designed for modern
// GPU architectures (Volta and newer), providing flexible matrix data layouts,
// explicit epilogue fusions, and algorithm heuristic search capabilities.
//
// Most operations require a [Handle], created with [Create]. Use [NewMatmulDesc],
// [NewMatrixLayout], and [NewMatmulPreference] to configure GEMM operations.
//
// # Other
//
//   - [Create]: initializes the cuBLASLt library and creates a handle to an opaque structure holding the cuBLASLt library context.
//   - [Handle.Destroy]: releases hardware resources used by the cuBLASLt library.
//   - [DisableCpuInstructionsSetMask]: Instructs cuBLASLt library to not use CPU instructions specified by the flags in the mask.
//   - [NewEmulationDesc]: creates a new emulation descriptor.
//   - [EmulationDesc.Destroy]: destroys a previously created emulation descriptor.
//   - [EmulationDesc.GetAttribute]: returns the value of the queried attribute belonging to a previously created emulation descriptor.
//   - [EmulationDesc.SetAttribute]: sets the value of the specified attribute belonging to a previously created emulation descriptor.
//   - [GetCudartVersion]: returns the version number of the CUDA Runtime library.
//   - [GetProperty]: returns the value of the requested property by writing it to the memory location pointed to by the value parameter.
//   - [GetVersion]: returns the version number of cuBLASLt library.
//   - [NewMatrixLayout]: creates a grouped matrix layout descriptor by allocating the memory needed to hold its opaque structure.
//   - [HeuristicsCacheGetCapacity]: Returns the Heuristics Cache capacity.
//   - [HeuristicsCacheSetCapacity]: Sets the Heuristics Cache capacity.
//   - [LoggerForceDisable]: disables logging for the entire run.
//   - [LoggerOpenFile]: opens and sets the logging output file in the given path.
//   - [LoggerSetCallback]: sets the logging callback function.
//   - [LoggerSetFile]: sets the logging output file.
//   - [LoggerSetLevel]: sets the value of the logging level.
//   - [LoggerSetMask]: sets the value of the logging mask.
//   - [Handle.Matmul]: computes the matrix multiplication of matrices A and B to produce the output matrix D, according to the following operation:
//   - [MatmulAlgoCapGetAttribute]: returns the value of the queried capability attribute for an initialized [MatmulAlgo] descriptor structure.
//   - [Handle.MatmulAlgoCheck]: performs the correctness check on the matrix multiply algorithm descriptor for the matrix multiply operation [Handle.Matmul] function with the given input matrices A, B and C, and the output matrix D.
//   - [MatmulAlgoConfigGetAttribute]: returns the value of the queried configuration attribute for an initialized [MatmulAlgo] descriptor.
//   - [MatmulAlgoConfigSetAttribute]: sets the value of the specified configuration attribute for an initialized [MatmulAlgo] descriptor.
//   - [Handle.MatmulAlgoGetHeuristic]: retrieves the possible algorithms for the matrix multiply operation [Handle.Matmul] function with the given input matrices A, B and C, and the output matrix D.
//   - [Handle.MatmulAlgoGetIds]: retrieves the IDs of all the matrix multiply algorithms that are valid, and can potentially be run by the [Handle.Matmul] function, for given types of the input matrices A, B and C, and of the output matrix D.
//   - [Handle.MatmulAlgoInit]: initializes the matrix multiply algorithm structure for the [Handle.Matmul] , for a specified matrix multiply algorithm and input matrices A, B and C, and the output matrix D.
//   - [NewMatmulDesc]: creates a matrix multiply descriptor by allocating the memory needed to hold its opaque structure.
//   - [MatmulDesc.Destroy]: destroys a previously created matrix multiply descriptor object.
//   - [MatmulDesc.GetAttribute]: returns the value of the queried attribute belonging to a previously created matrix multiply descriptor.
//   - [MatmulDesc.SetAttribute]: sets the value of the specified attribute belonging to a previously created matrix multiply descriptor.
//   - [NewMatmulPreference]: creates a matrix multiply heuristic search preferences descriptor by allocating the memory needed to hold its opaque structure.
//   - [MatmulPreference.Destroy]: destroys a previously created matrix multiply preferences descriptor object.
//   - [MatmulPreference.GetAttribute]: returns the value of the queried attribute belonging to a previously created matrix multiply heuristic search preferences descriptor.
//   - [MatmulPreference.SetAttribute]: sets the value of the specified attribute belonging to a previously created matrix multiply preferences descriptor.
//   - [MatrixLayout.Destroy]: destroys a previously created matrix layout descriptor object.
//   - [MatrixLayout.GetAttribute]: returns the value of the queried attribute belonging to the specified matrix layout descriptor.
//   - [MatrixLayout.SetAttribute]: sets the value of the specified attribute belonging to a previously created matrix layout descriptor.
//   - [Handle.MatrixTransform]: computes the matrix transformation operation on the input matrices A and B, to produce the output matrix C, according to the below operation:
//   - [NewMatrixTransformDesc]: creates a matrix transform descriptor by allocating the memory needed to hold its opaque structure.
//   - [MatrixTransformDesc.Destroy]: destroys a previously created matrix transform descriptor object.
//   - [MatrixTransformDesc.GetAttribute]: returns the value of the queried attribute belonging to a previously created matrix transform descriptor.
//   - [MatrixTransformDesc.SetAttribute]: sets the value of the specified attribute belonging to a previously created matrix transform descriptor.
//
// [cuBLASLt Documentation]: https://docs.nvidia.com/cuda/cublas/#cublaslt-library
package cublaslt

// Code generated by gen-nvidia from cublasLt.h. DO NOT EDIT.
