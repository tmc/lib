//
// # Other
//
//   - [Create]: This function initializes the cuBLASLt library and creates a handle to an opaque structure holding the cuBLASLt library context.
//   - [Handle.Destroy]: This function releases hardware resources used by the cuBLASLt library.
//   - [NewEmulationDesc]: This function creates a new emulation descriptor.
//   - [EmulationDesc.Destroy]: This function destroys a previously created emulation descriptor.
//   - [EmulationDesc.GetAttribute]: This function returns the value of the queried attribute belonging to a previously created emulation descriptor.
//   - [EmulationDesc.SetAttribute]: This function sets the value of the specified attribute belonging to a previously created emulation descriptor.
//   - [GetProperty]: This function returns the value of the requested property by writing it to the memory location pointed to by the value parameter.
//   - [GroupedMatrixLayoutCreate]: Experimental: This function creates a grouped matrix layout descriptor by allocating the memory needed to hold its opaque structure.
//   - [HeuristicsCacheGetCapacity]: Returns the Heuristics Cache capacity.
//   - [HeuristicsCacheSetCapacity]: Sets the Heuristics Cache capacity.
//   - [LoggerForceDisable]: Experimental: This function disables logging for the entire run.
//   - [LoggerOpenFile]: Experimental: This function opens and sets the logging output file in the given path.
//   - [LoggerSetCallback]: Experimental: This function sets the logging callback function.
//   - [LoggerSetFile]: Experimental: This function sets the logging output file.
//   - [LoggerSetLevel]: Experimental: This function sets the value of the logging level.
//   - [LoggerSetMask]: Experimental: This function sets the value of the logging mask.
//   - [Handle.Matmul]: This function computes the matrix multiplication of matrices A and B to produce the output matrix D, according to the following operation:
//   - [MatmulAlgoCapGetAttribute]: This function returns the value of the queried capability attribute for an initialized [MatmulAlgo] descriptor structure.
//   - [Handle.MatmulAlgoCheck]: This function performs the correctness check on the matrix multiply algorithm descriptor for the matrix multiply operation [Handle.Matmul] function with the given input matrices A, B and C, and the output matrix D.
//   - [MatmulAlgoConfigGetAttribute]: This function returns the value of the queried configuration attribute for an initialized [MatmulAlgo] descriptor.
//   - [MatmulAlgoConfigSetAttribute]: This function sets the value of the specified configuration attribute for an initialized [MatmulAlgo] descriptor.
//   - [Handle.MatmulAlgoGetHeuristic]: This function retrieves the possible algorithms for the matrix multiply operation [Handle.Matmul] function with the given input matrices A, B and C, and the output matrix D.
//   - [Handle.MatmulAlgoGetIds]: This function retrieves the IDs of all the matrix multiply algorithms that are valid, and can potentially be run by the [Handle.Matmul] function, for given types of the input matrices A, B and C, and of the output matrix D.
//   - [Handle.MatmulAlgoInit]: This function initializes the matrix multiply algorithm structure for the [Handle.Matmul] , for a specified matrix multiply algorithm and input matrices A, B and C, and the output matrix D.
//   - [NewMatmulDesc]: This function creates a matrix multiply descriptor by allocating the memory needed to hold its opaque structure.
//   - [MatmulDesc.Destroy]: This function destroys a previously created matrix multiply descriptor object.
//   - [MatmulDesc.GetAttribute]: This function returns the value of the queried attribute belonging to a previously created matrix multiply descriptor.
//   - [MatmulDesc.SetAttribute]: This function sets the value of the specified attribute belonging to a previously created matrix multiply descriptor.
//   - [NewMatmulPreference]: This function creates a matrix multiply heuristic search preferences descriptor by allocating the memory needed to hold its opaque structure.
//   - [MatmulPreference.Destroy]: This function destroys a previously created matrix multiply preferences descriptor object.
//   - [MatmulPreference.GetAttribute]: This function returns the value of the queried attribute belonging to a previously created matrix multiply heuristic search preferences descriptor.
//   - [MatmulPreference.SetAttribute]: This function sets the value of the specified attribute belonging to a previously created matrix multiply preferences descriptor.
//   - [NewMatrixLayout]: This function creates a matrix layout descriptor by allocating the memory needed to hold its opaque structure.
//   - [MatrixLayout.Destroy]: This function destroys a previously created matrix layout descriptor object.
//   - [MatrixLayout.GetAttribute]: This function returns the value of the queried attribute belonging to the specified matrix layout descriptor.
//   - [MatrixLayout.SetAttribute]: This function sets the value of the specified attribute belonging to a previously created matrix layout descriptor.
//   - [Handle.MatrixTransform]: This function computes the matrix transformation operation on the input matrices A and B, to produce the output matrix C, according to the below operation:
//   - [NewMatrixTransformDesc]: This function creates a matrix transform descriptor by allocating the memory needed to hold its opaque structure.
//   - [MatrixTransformDesc.Destroy]: This function destroys a previously created matrix transform descriptor object.
//   - [MatrixTransformDesc.GetAttribute]: This function returns the value of the queried attribute belonging to a previously created matrix transform descriptor.
//   - [MatrixTransformDesc.SetAttribute]: This function sets the value of the specified attribute belonging to a previously created matrix transform descriptor.
package cublaslt

// Code generated by gen-nvidia from cublasLt.h. DO NOT EDIT.
