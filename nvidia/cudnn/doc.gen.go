// Package cudnn provides low-level Go bindings for cuDNN (CUDA Deep Neural Network library).
//
// cuDNN provides GPU-accelerated primitives for deep neural networks including
// convolution, pooling, softmax, batch normalization, activation, RNN, attention,
// and CTC loss operations.
//
// The central type is [Handle], which manages a cuDNN context. Most operations
// require tensor descriptors ([TensorDescriptor]) and other descriptors to
// describe the data layout. Create handles and descriptors with the Create*
// functions (e.g., [Create], [CreateTensorDescriptor]) and clean up with
// the corresponding Destroy methods.
//
// This package maps every function from cudnn.h to a Go function using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libcudnn.
// All functions return errors wrapping cudnnStatus_t with descriptive messages
// from cudnnGetErrorString.
//
// # Backend API
//
//   - [NewBackendDescriptorCreateDescriptor]: This function allocates memory in the descriptor for a given descriptor type and at the location pointed by the descriptor.
//   - [BackendDescriptor.DestroyDescriptor]: This function destroys instances of [BackendDescriptor] that were previously created using [NewBackendDescriptorCreateDescriptor].
//   - [BackendDescriptor.Execute]: This function executes the given Engine Configuration Plan on the VariantPack and the finalized ExecutionPlan on the data.
//   - [BackendDescriptor.Finalize]: This function finalizes the memory pointed to by the descriptor.
//   - [BackendDescriptor.GetAttribute]: This function retrieves the values of an attribute of a descriptor.
//   - [BackendDescriptor.SetAttribute]: This function sets an attribute of a descriptor to values provided as a pointer.
//
// # CTC Loss
//
//   - [Handle.CTCLoss]: This function returns the CTC costs and gradients, given the probabilities and labels.
//   - [Handle.CTCLossV8]: This function returns the CTC costs and gradients, given the probabilities and labels.
//
// # Lifecycle
//
//   - [Create]: This function initializes the cuDNN library and creates a handle to an opaque structure holding the cuDNN library context.
//   - [CreateCTCLossDescriptor]: This function creates a CTC loss function descriptor.
//   - [CreateDropoutDescriptor]: This function creates a generic dropout descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateLRNDescriptor]: This function allocates the memory needed to hold the data needed for LRN and DivisiveNormalization layers operation and returns a descriptor used with subsequent layer forward and backward calls.
//   - [CreateRNNDataDescriptor]: This function creates a RNN data descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateRNNDescriptor]: This function creates a generic RNN descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateSpatialTransformerDescriptor]: This function creates a generic spatial transformer descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateTensorDescriptor]: This function creates a generic tensor descriptor object by allocating the memory needed to hold its opaque structure.
//   - [Handle.Destroy]: This function releases the resources used by the cuDNN handle.
//   - [CTCLossDescriptor.Destroy]: This function destroys a CTC loss function descriptor object.
//   - [DropoutDescriptor.Destroy]: This function destroys a previously created dropout descriptor object.
//   - [LRNDescriptor.Destroy]: This function destroys a previously created LRN descriptor object.
//   - [RNNDataDescriptor.Destroy]: This function destroys a previously created RNN data descriptor object.
//   - [RNNDescriptor.Destroy]: This function destroys a previously created RNN descriptor object.
//   - [SpatialTransformerDescriptor.Destroy]: This function destroys a previously created spatial transformer descriptor object.
//   - [TensorDescriptor.Destroy]: This function destroys a previously created tensor descriptor object.
//
// # Dropout
//
//   - [Handle.DropoutBackward]: This function performs backward dropout operation over dy returning results in dx.
//   - [Handle.DropoutForward]: This function performs forward dropout operation over x returning results in y.
//   - [TensorDescriptor.DropoutGetReserveSpaceSize]: This function is used to query the amount of reserve needed to run dropout with the input dimensions given by xDesc.
//   - [Handle.DropoutGetStatesSize]: This function is used to query the amount of space required to store the states of the random number generators used by the [Handle.DropoutForward] function.
//
// # Stream Management
//
//   - [Handle.GetStream]: This function retrieves the user CUDA stream programmed in the cuDNN handle.
//   - [Handle.SetStream]: This function sets the user’s CUDA stream in the cuDNN handle.
//
// # LRN
//
//   - [Handle.LRNCrossChannelBackward]: This function performs the backward LRN layer computation.
//   - [Handle.LRNCrossChannelForward]: This function performs the forward LRN layer computation.
//
// # RNN
//
//   - [Handle.RNNBackwardData]: This function computes exact, first-order derivatives of the RNN model with respect to its inputs: x, hx and for the LSTM cell type also cx.
//   - [Handle.RNNBackwardWeights]: This function computes exact, first-order derivatives of the RNN model with respect to all trainable parameters: weights and biases.
//   - [Handle.RNNForward]: This routine computes the forward response of the recurrent neural network described by rnnDesc with inputs in x, hx, cx, and weights/biases in the weightSpace buffer.
//   - [RNNDescriptor.RNNGetClip]: Retrieves the current LSTM cell clipping parameters, and stores them in the arguments provided.
//   - [RNNDescriptor.RNNSetClip]: Sets the LSTM cell clipping mode.
//
// # Softmax
//
//   - [Handle.SoftmaxBackward]: This routine computes the gradient of the softmax function.
//   - [Handle.SoftmaxForward]: This routine computes the softmax function.
//
// # Other
//
//   - [AdvVersionCheck]: Cross-library version checker.
//   - [Handle.BuildRNNDynamic]: This function compiles the RNN persistent code using CUDA runtime compilation library (NVRTC) when the [RnnAlgoPersistDynamic] algo is selected.
//   - [CnnVersionCheck]: Cross-library version checker.
//   - [Handle.DivisiveNormalizationBackward]: This function performs the backward DivisiveNormalization layer computation.
//   - [Handle.DivisiveNormalizationForward]: This function performs the forward spatial DivisiveNormalization layer computation.
//   - [CTCLossDescriptor.Get]: This function returns the configuration of the passed CTC loss function descriptor.
//   - [Handle.GetCTCLossWorkspaceSize]: This function returns the amount of GPU memory workspace the user needs to allocate to be able to call [Handle.CTCLoss] with the specified algorithm.
//   - [Handle.GetCTCLossWorkspaceSizeV8]: This function returns the amount of GPU memory workspace the user needs to allocate to be able to call [Handle.CTCLossV8] with the specified algorithm.
//   - [GetCallback]: This function queries the internal states of cuDNN error reporting functionality.
//   - [DropoutDescriptor.Get]: This function queries the fields of a previously initialized dropout descriptor.
//   - [LRNDescriptor.Get]: This function retrieves values stored in the previously initialized LRN descriptor object.
//   - [GetProperty]: This function writes a specific part of the cuDNN library version number into the provided host storage.
//   - [RNNDataDescriptor.Get]: This function retrieves a previously created RNN data descriptor object.
//   - [RNNDescriptor.Get]: This function retrieves RNN network parameters that were configured by [RNNDescriptor.Set].
//   - [Handle.GetRNNTempSpaceSizes]: This function computes the work and reserve space buffer sizes based on the RNN network geometry stored in rnnDesc, designated usage (inference or training) defined by the fMode argument, and the current RNN data dimensions ( maxSeqLength, batchSize) retrieved from xDesc.
//   - [Handle.GetRNNWeightParams]: This function is used to obtain the start address and shape of every RNN weight matrix and bias vector in each pseudo-layer within the recurrent neural network model.
//   - [Handle.GetRNNWeightSpaceSize]: This function reports the required size of the weight space buffer in bytes.
//   - [TensorDescriptor.GetTensor4dDescriptor]: This function queries the parameters of the previously initialized Tensor4d descriptor object.
//   - [TensorDescriptor.GetTensorNdDescriptor]: This function retrieves values stored in a previously initialized TensorNd descriptor object.
//   - [TensorDescriptor.GetTensorSizeInBytes]: This function returns the size of the tensor in memory in respect to the given descriptor.
//   - [GraphVersionCheck]: Cross-library version checker.
//   - [OpsVersionCheck]: Cross-library version checker.
//   - [DropoutDescriptor.Restore]: This function restores a dropout descriptor to a previously saved-off state.
//   - [CTCLossDescriptor.Set]: This function sets a CTC loss function descriptor.
//   - [SetCallback]: This function sets the internal states of cuDNN error reporting functionality.
//   - [DropoutDescriptor.Set]: This function initializes a previously created dropout descriptor object.
//   - [LRNDescriptor.Set]: This function initializes a previously created LRN descriptor object.
//   - [RNNDataDescriptor.Set]: This function initializes a previously created RNN data descriptor object.
//   - [RNNDescriptor.Set]: This function initializes a previously created RNN descriptor object.
//   - [SpatialTransformerDescriptor.SetSpatialTransformerNdDescriptor]: This function initializes a previously created generic spatial transformer descriptor object.
//   - [Handle.SetTensor]: This function sets all the elements of a tensor to a given value.
//   - [TensorDescriptor.SetTensor4dDescriptor]: This function initializes a previously created generic tensor descriptor object into a 4D tensor.
//   - [TensorDescriptor.SetTensor4dDescriptorEx]: This function initializes a previously created generic tensor descriptor object into a 4D tensor, similarly to [TensorDescriptor.SetTensor4dDescriptor] but with the strides explicitly passed as parameters.
//   - [TensorDescriptor.SetTensorNdDescriptor]: This function initializes a previously created generic tensor descriptor object.
//   - [TensorDescriptor.SetTensorNdDescriptorEx]: This function initializes an Nd tensor descriptor.
//   - [Handle.SpatialTfGridGeneratorBackward]: This function computes the gradient of a grid generation operation.
//   - [Handle.SpatialTfGridGeneratorForward]: This function generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
//   - [Handle.SpatialTfSamplerBackward]: This function computes the gradient of a sampling operation.
//   - [Handle.SpatialTfSamplerForward]: This function performs a sampler operation and generates the output tensor using the grid given by the grid generator.
//
// [cuDNN Documentation]: https://docs.nvidia.com/deeplearning/cudnn/
package cudnn

// Code generated by gen-nvidia from cudnn.h. DO NOT EDIT.
