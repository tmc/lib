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
// This package maps selected host functions from cudnn.h to a Go function using
// [github.com/ebitengine/purego] for cgo-free dynamic loading of libcudnn.
// Status-returning functions wrap cudnnStatus_t errors with descriptive messages
// from cudnnGetErrorString.
//
// # Activation
//
//   - [Handle.ActivationBackward]: This routine computes the gradient of a neuron activation function.
//   - [Handle.ActivationForward]: This routine applies a specified neuron activation function element-wise over each input value.
//
// # Backend API
//
//   - [NewBackendDescriptorCreateDescriptor]: allocates memory in the descriptor for a given descriptor type and at the location pointed by the descriptor.
//   - [BackendDescriptor.DestroyDescriptor]: destroys instances of [BackendDescriptor] that were previously created using [NewBackendDescriptorCreateDescriptor].
//   - [BackendDescriptor.Execute]: executes the given Engine Configuration Plan on the VariantPack and the finalized ExecutionPlan on the data.
//   - [BackendDescriptor.Finalize]: finalizes the memory pointed to by the descriptor.
//   - [BackendDescriptor.GetAttribute]: retrieves the values of an attribute of a descriptor.
//   - [BackendDescriptor.Initialize]: repurposes a pre-allocated memory pointed to by a descriptor of size sizeInByte to a backend descriptor of type descriptorType.
//   - [BackendDescriptor.SetAttribute]: sets an attribute of a descriptor to values provided as a pointer.
//
// # Batch Normalization
//
//   - [Handle.BatchNormalizationBackward]: performs the backward batch normalization layer computation.
//   - [Handle.BatchNormalizationBackwardEx]: is an extension of the [Handle.BatchNormalizationBackward] for performing the backward batch normalization layer computation with a fast NHWC semi-persistent kernel.
//   - [Handle.BatchNormalizationForwardInference]: performs the forward batch normalization layer computation for the inference phase.
//   - [Handle.BatchNormalizationForwardTraining]: performs the forward batch normalization layer computation for the training phase.
//   - [Handle.BatchNormalizationForwardTrainingEx]: is an extension of the [Handle.BatchNormalizationForwardTraining] for performing the forward batch normalization layer computation.
//
// # CTC Loss
//
//   - [Handle.CTCLoss]: returns the CTC costs and gradients, given the probabilities and labels.
//   - [Handle.CTCLossV8]: returns the CTC costs and gradients, given the probabilities and labels.
//
// # Convolution
//
//   - [Handle.ConvolutionBackwardBias]: computes the convolution function gradient with respect to the bias, which is the sum of every element belonging to the same feature map across all of the images of the input tensor.
//   - [Handle.ConvolutionBackwardData]: computes the convolution data gradient of the tensor dy, where y is the output of the forward convolution in [Handle.ConvolutionForward].
//   - [Handle.ConvolutionBackwardFilter]: computes the convolution weight (filter) gradient of the tensor dy, where y is the output of the forward convolution in [Handle.ConvolutionForward].
//   - [Handle.ConvolutionBiasActivationForward]: applies a bias and then an activation to the convolutions or cross-correlations of [Handle.ConvolutionForward], returning results in y.
//   - [Handle.ConvolutionForward]: executes convolutions or cross-correlations over x using filters specified with w, returning results in y.
//
// # Lifecycle
//
//   - [Create]: initializes the cuDNN library and creates a handle to an opaque structure holding the cuDNN library context.
//   - [CreateActivationDescriptor]: creates an activation descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateAttnDescriptor]: creates one instance of an opaque attention descriptor object by allocating the host memory for it and initializing all descriptor fields.
//   - [CreateCTCLossDescriptor]: creates a CTC loss function descriptor.
//   - [CreateConvolutionDescriptor]: creates a convolution descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateDropoutDescriptor]: creates a generic dropout descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateFilterDescriptor]: creates a filter descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateFusedOpsConstParamPack]: creates an opaque structure to store the various problem size information, such as the shape, layout and the type of tensors, and the descriptors for convolution and activation, for the selected sequence of cudnnFusedOps computations.
//   - [CreateFusedOpsPlan]: creates the plan descriptor for the cudnnFusedOps computation.
//   - [CreateFusedOpsVariantParamPack]: creates the variant pack descriptor for the cudnnFusedOps computation.
//   - [CreateLRNDescriptor]: allocates the memory needed to hold the data needed for LRN and DivisiveNormalization layers operation and returns a descriptor used with subsequent layer forward and backward calls.
//   - [CreateOpTensorDescriptor]: creates a tensor pointwise math descriptor.
//   - [CreatePoolingDescriptor]: creates a pooling descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateRNNDataDescriptor]: creates a RNN data descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateRNNDescriptor]: creates a generic RNN descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateReduceTensorDescriptor]: creates a reduced tensor descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateSeqDataDescriptor]: creates one instance of an opaque sequence data descriptor object by allocating the host memory for it and initializing all descriptor fields.
//   - [CreateSpatialTransformerDescriptor]: creates a generic spatial transformer descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateTensorDescriptor]: creates a generic tensor descriptor object by allocating the memory needed to hold its opaque structure.
//   - [CreateTensorTransformDescriptor]: creates a tensor transform descriptor object by allocating the memory needed to hold its opaque structure.
//   - [Handle.Destroy]: releases the resources used by the cuDNN handle.
//   - [ActivationDescriptor.Destroy]: destroys a previously created activation descriptor object.
//   - [AttnDescriptor.Destroy]: destroys the attention descriptor object and releases its memory.
//   - [CTCLossDescriptor.Destroy]: destroys a CTC loss function descriptor object.
//   - [ConvolutionDescriptor.Destroy]: destroys a previously created convolution descriptor object.
//   - [DropoutDescriptor.Destroy]: destroys a previously created dropout descriptor object.
//   - [FilterDescriptor.Destroy]: destroys a filter object.
//   - [FusedOpsConstParamPack.Destroy]: destroys a previously-created [FusedOpsConstParamPack] structure.
//   - [FusedOpsPlan.Destroy]: destroys the plan descriptor provided.
//   - [FusedOpsVariantParamPack.Destroy]: destroys a previously-created descriptor for cudnnFusedOps constant parameters.
//   - [LRNDescriptor.Destroy]: destroys a previously created LRN descriptor object.
//   - [OpTensorDescriptor.Destroy]: deletes a tensor pointwise math descriptor object.
//   - [PoolingDescriptor.Destroy]: destroys a previously created pooling descriptor object.
//   - [RNNDataDescriptor.Destroy]: destroys a previously created RNN data descriptor object.
//   - [RNNDescriptor.Destroy]: destroys a previously created RNN descriptor object.
//   - [ReduceTensorDescriptor.Destroy]: destroys a previously created reduce tensor descriptor object.
//   - [SeqDataDescriptor.Destroy]: destroys the sequence data descriptor object and releases its memory.
//   - [SpatialTransformerDescriptor.Destroy]: destroys a previously created spatial transformer descriptor object.
//   - [TensorDescriptor.Destroy]: destroys a previously created tensor descriptor object.
//   - [TensorTransformDescriptor.Destroy]: Destroys a previously created tensor transform descriptor.
//
// # Dropout
//
//   - [Handle.DropoutBackward]: performs backward dropout operation over dy returning results in dx.
//   - [Handle.DropoutForward]: performs forward dropout operation over x returning results in y.
//   - [TensorDescriptor.DropoutGetReserveSpaceSize]: is used to query the amount of reserve needed to run dropout with the input dimensions given by xDesc.
//   - [Handle.DropoutGetStatesSize]: is used to query the amount of space required to store the states of the random number generators used by the [Handle.DropoutForward] function.
//
// # Version Queries
//
//   - [GetCudartVersion]: The same version of a given cuDNN library can be compiled against different CUDA toolkit versions.
//   - [GetMaxDeviceVersion]: returns the maximum SM version that the cuDNN library is aware of and supports natively.
//   - [GetVersion]: returns the version number of the cuDNN library.
//
// # Stream Management
//
//   - [Handle.GetStream]: retrieves the user CUDA stream programmed in the cuDNN handle.
//   - [Handle.SetStream]: sets the user’s CUDA stream in the cuDNN handle.
//
// # LRN
//
//   - [Handle.LRNCrossChannelBackward]: performs the backward LRN layer computation.
//   - [Handle.LRNCrossChannelForward]: performs the forward LRN layer computation.
//
// # Normalization
//
//   - [Handle.NormalizationBackward]: performs backward normalization layer computation that is specified by mode.
//   - [Handle.NormalizationForwardInference]: performs the forward normalization layer computation for the inference phase.
//   - [Handle.NormalizationForwardTraining]: performs the forward normalization layer computation for the training phase.
//
// # Tensor Operations
//
//   - [Handle.OpTensor]: [TensorNchwVectC] is not supported as input tensor format.
//   - [Handle.ReduceTensor]: reduces tensor A by implementing the equation C = alpha * reduce op ( A ) + beta * C, given tensors A and C and scaling factors alpha and beta.
//
// # Pooling
//
//   - [Handle.PoolingBackward]: computes the gradient of a pooling operation.
//   - [Handle.PoolingForward]: computes pooling of input values (meaning, the maximum or average of several adjacent values) to produce an output with smaller height and/or width.
//
// # RNN
//
//   - [Handle.RNNBackwardData]: computes exact, first-order derivatives of the RNN model with respect to its inputs: x, hx and for the LSTM cell type also cx.
//   - [Handle.RNNBackwardWeights]: computes exact, first-order derivatives of the RNN model with respect to all trainable parameters: weights and biases.
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
//   - [Handle.AddTensor]: adds the scaled values of a bias tensor to another tensor.
//   - [AdvVersionCheck]: Cross-library version checker.
//   - [Handle.BuildRNNDynamic]: compiles the RNN persistent code using CUDA runtime compilation library (NVRTC) when the [RnnAlgoPersistDynamic] algo is selected.
//   - [CnnVersionCheck]: Cross-library version checker.
//   - [TensorDescriptor.DeriveBN]: derives a secondary tensor descriptor for the batch normalization scale, invVariance, bnBias, and bnScale subtensors from the layer’s x data descriptor.
//   - [TensorDescriptor.DeriveNorm]: derives tensor descriptors for the normalization mean, invariance, normBias, and normScale subtensors from the layer’s x data descriptor and norm mode.
//   - [Handle.DivisiveNormalizationBackward]: performs the backward DivisiveNormalization layer computation.
//   - [Handle.DivisiveNormalizationForward]: performs the forward spatial DivisiveNormalization layer computation.
//   - [Handle.FindConvolutionBackwardDataAlgorithm]: attempts all algorithms available for [Handle.ConvolutionBackwardData].
//   - [Handle.FindConvolutionBackwardDataAlgorithmEx]: attempts all algorithms available for [Handle.ConvolutionBackwardData].
//   - [Handle.FindConvolutionBackwardFilterAlgorithm]: attempts all algorithms available for [Handle.ConvolutionBackwardFilter].
//   - [Handle.FindConvolutionBackwardFilterAlgorithmEx]: attempts all algorithms available for [Handle.ConvolutionBackwardFilter].
//   - [Handle.FindConvolutionForwardAlgorithm]: attempts all algorithms available for [Handle.ConvolutionForward].
//   - [Handle.FindConvolutionForwardAlgorithmEx]: attempts all algorithms available for [Handle.ConvolutionForward].
//   - [Handle.FusedOpsExecute]: executes the sequence of cudnnFusedOps operations.
//   - [ActivationDescriptor.Get]: queries a previously initialized generic activation descriptor object.
//   - [ActivationDescriptor.GetActivationDescriptorSwishBeta]: queries the current beta parameter set for SWISH activation.
//   - [AttnDescriptor.Get]: retrieves settings from the previously created attention descriptor.
//   - [Handle.GetBatchNormalizationBackwardExWorkspaceSize]: returns the amount of GPU memory workspace the user should allocate to be able to call [Handle.GetBatchNormalizationBackwardExWorkspaceSize] function for the specified bnOps input setting.
//   - [Handle.GetBatchNormalizationForwardTrainingExWorkspaceSize]: returns the amount of GPU memory workspace the user should allocate to be able to call [Handle.GetBatchNormalizationForwardTrainingExWorkspaceSize] function for the specified bnOps input setting.
//   - [Handle.GetBatchNormalizationTrainingExReserveSpaceSize]: returns the amount of reserve GPU memory workspace the user should allocate for the batch normalization operation, for the specified bnOps input setting.
//   - [CTCLossDescriptor.GetCTCLossDescriptorEx]: returns the configuration of the passed CTC loss function descriptor.
//   - [CTCLossDescriptor.Get]: returns the configuration of the passed CTC loss function descriptor.
//   - [Handle.GetCTCLossWorkspaceSize]: returns the amount of GPU memory workspace the user needs to allocate to be able to call [Handle.CTCLoss] with the specified algorithm.
//   - [Handle.GetCTCLossWorkspaceSizeV8]: returns the amount of GPU memory workspace the user needs to allocate to be able to call [Handle.CTCLossV8] with the specified algorithm.
//   - [GetCallback]: queries the internal states of cuDNN error reporting functionality.
//   - [ConvolutionDescriptor.GetConvolution2dDescriptor]: queries a previously initialized 2D convolution descriptor object.
//   - [ConvolutionDescriptor.GetConvolution2dForwardOutputDim]: returns the dimensions of the resulting 4D tensor of a 2D convolution, given the convolution descriptor, the input tensor descriptor and the filter descriptor.
//   - [Handle.GetConvolutionBackwardDataAlgorithmMaxCount]: returns the maximum number of algorithms which can be returned from [Handle.FindConvolutionBackwardDataAlgorithm] and [Handle.GetConvolutionForwardAlgorithm].
//   - [Handle.GetConvolutionBackwardDataAlgorithm]: serves as a heuristic for obtaining the best suited algorithm for [Handle.ConvolutionBackwardData] for the given layer specifications.
//   - [Handle.GetConvolutionBackwardDataWorkspaceSize]: returns the amount of GPU memory workspace the user needs to allocate to be able to call [Handle.ConvolutionBackwardData] with the specified algorithm.
//   - [Handle.GetConvolutionBackwardFilterAlgorithmMaxCount]: returns the maximum number of algorithms which can be returned from [Handle.FindConvolutionBackwardFilterAlgorithm] and [Handle.GetConvolutionForwardAlgorithm].
//   - [Handle.GetConvolutionBackwardFilterAlgorithm]: serves as a heuristic for obtaining the best suited algorithm for [Handle.ConvolutionBackwardFilter] for the given layer specifications.
//   - [Handle.GetConvolutionBackwardFilterWorkspaceSize]: returns the amount of GPU memory workspace the user needs to allocate to be able to call [Handle.ConvolutionBackwardFilter] with the specified algorithm.
//   - [Handle.GetConvolutionForwardAlgorithmMaxCount]: returns the maximum number of algorithms which can be returned from [Handle.FindConvolutionForwardAlgorithm] and [Handle.GetConvolutionForwardAlgorithm].
//   - [Handle.GetConvolutionForwardAlgorithm]: serves as a heuristic for obtaining the best suited algorithm for [Handle.ConvolutionForward] for the given layer specifications.
//   - [Handle.GetConvolutionForwardWorkspaceSize]: returns the amount of GPU memory workspace the user needs to allocate to be able to call [Handle.ConvolutionForward] with the specified algorithm.
//   - [ConvolutionDescriptor.GetConvolutionGroupCount]: returns the group count specified in the given convolution descriptor.
//   - [ConvolutionDescriptor.GetConvolutionMathType]: returns the math type specified in a given convolution descriptor.
//   - [ConvolutionDescriptor.GetConvolutionNdDescriptor]: queries a previously initialized convolution descriptor object.
//   - [ConvolutionDescriptor.GetConvolutionNdForwardOutputDim]: returns the dimensions of the resulting Nd tensor of a nbDims-2-D convolution, given the convolution descriptor, the input tensor descriptor and the filter descriptor This function can help to setup the output tensor and allocate the proper amount of memory prior to launch the actual convolution.
//   - [ConvolutionDescriptor.GetConvolutionReorderType]: retrieves the convolution reorder type from the given convolution descriptor.
//   - [DropoutDescriptor.Get]: queries the fields of a previously initialized dropout descriptor.
//   - [FilterDescriptor.GetFilter4dDescriptor]: queries the parameters of the previously initialized Filter4d descriptor object.
//   - [FilterDescriptor.GetFilterNdDescriptor]: queries a previously initialized FilterNd descriptor object.
//   - [FilterDescriptor.GetFilterSizeInBytes]: returns the size of the filter tensor in memory with respect to the given descriptor.
//   - [Handle.GetFoldedConvBackwardDataDescriptors]: calculates folding descriptors for backward data gradients.
//   - [FusedOpsConstParamPack.GetFusedOpsConstParamPackAttribute]: retrieves the values of the descriptor pointed to by the param pointer input.
//   - [FusedOpsVariantParamPack.GetFusedOpsVariantParamPackAttribute]: retrieves the settings of the variable parameter pack descriptor.
//   - [LRNDescriptor.Get]: retrieves values stored in the previously initialized LRN descriptor object.
//   - [GetLastErrorString]: retrieves the last encountered cuDNN error message in the current thread to a NULL terminated (ASCIIZ) string.
//   - [Handle.GetMultiHeadAttnBuffers]: computes weight, work, and reserve space buffer sizes used by the following functions: Assigning NULL to the reserveSpaceSizeInBytes argument indicates that the user does not plan to invoke multihead attention gradient functions: [Handle.MultiHeadAttnBackwardData] and [Handle.MultiHeadAttnBackwardWeights].
//   - [Handle.GetMultiHeadAttnWeights]: obtains the shape of the weight or bias tensor.
//   - [Handle.GetNormalizationBackwardWorkspaceSize]: returns the amount of GPU memory workspace the user should allocate to be able to call [Handle.NormalizationBackward] function for the specified normOps and algo input setting.
//   - [Handle.GetNormalizationForwardTrainingWorkspaceSize]: returns the amount of GPU memory workspace the user should allocate to be able to call [Handle.NormalizationForwardTraining] function for the specified normOps and algo input setting.
//   - [Handle.GetNormalizationTrainingReserveSpaceSize]: returns the amount of reserve GPU memory workspace the user should allocate for the normalization operation, for the specified normOps input setting.
//   - [OpTensorDescriptor.Get]: returns the configuration of the passed tensor pointwise math descriptor.
//   - [PoolingDescriptor.GetPooling2dDescriptor]: queries a previously created Pooling2d descriptor object.
//   - [PoolingDescriptor.GetPooling2dForwardOutputDim]: provides the output dimensions of a tensor after Pooling2d has been applied.
//   - [PoolingDescriptor.GetPoolingNdDescriptor]: queries a previously initialized generic PoolingNd descriptor object.
//   - [PoolingDescriptor.GetPoolingNdForwardOutputDim]: provides the output dimensions of a tensor after PoolingNd has been applied.
//   - [GetProperty]: writes a specific part of the cuDNN library version number into the provided host storage.
//   - [RNNDataDescriptor.Get]: retrieves a previously created RNN data descriptor object.
//   - [RNNDescriptor.Get]: retrieves RNN network parameters that were configured by [RNNDescriptor.Set].
//   - [Handle.GetRNNTempSpaceSizes]: computes the work and reserve space buffer sizes based on the RNN network geometry stored in rnnDesc, designated usage (inference or training) defined by the fMode argument, and the current RNN data dimensions ( maxSeqLength, batchSize) retrieved from xDesc.
//   - [Handle.GetRNNWeightParams]: is used to obtain the start address and shape of every RNN weight matrix and bias vector in each pseudo-layer within the recurrent neural network model.
//   - [Handle.GetRNNWeightSpaceSize]: reports the required size of the weight space buffer in bytes.
//   - [ReduceTensorDescriptor.Get]: queries a previously initialized reduce tensor descriptor object.
//   - [Handle.GetReductionIndicesSize]: This is a helper function to return the minimum size of the index space to be passed to the reduction given the input and output tensors.
//   - [Handle.GetReductionWorkspaceSize]: This is a helper function to return the minimum size of the workspace to be passed to the reduction given the input and output tensors.
//   - [SeqDataDescriptor.Get]: retrieves settings from a previously created sequence data descriptor.
//   - [TensorDescriptor.GetTensor4dDescriptor]: queries the parameters of the previously initialized Tensor4d descriptor object.
//   - [TensorDescriptor.GetTensorNdDescriptor]: retrieves values stored in a previously initialized TensorNd descriptor object.
//   - [TensorDescriptor.GetTensorSizeInBytes]: returns the size of the tensor in memory in respect to the given descriptor.
//   - [TensorTransformDescriptor.Get]: returns the values stored in a previously initialized tensor transform descriptor.
//   - [GraphVersionCheck]: Cross-library version checker.
//   - [Handle.Im2Col]: constructs the A matrix necessary to perform a forward pass of GEMM convolution.
//   - [TensorTransformDescriptor.InitTransformDest]: initializes and returns a destination tensor descriptor destDesc for tensor transform operations.
//   - [Handle.MakeFusedOpsPlan]: determines the optimum kernel to execute, and the workspace size the user should allocate, prior to the actual execution of the fused operations by [Handle.FusedOpsExecute].
//   - [Handle.MultiHeadAttnBackwardData]: computes exact, first-order derivatives of the multihead attention block with respect to its inputs: Q, K, V.
//   - [Handle.MultiHeadAttnBackwardWeights]: computes exact, first-order derivatives of the multihead attention block with respect to its trainable parameters: projection weights and projection biases.
//   - [Handle.MultiHeadAttnForward]: The [Handle.MultiHeadAttnForward] function computes the forward responses of the multihead attention layer.
//   - [OpsVersionCheck]: Cross-library version checker.
//   - [Handle.QueryRuntimeError]: cuDNN library functions perform extensive input argument checking before launching GPU kernels.
//   - [Handle.ReorderFilterAndBias]: reorders the filter and bias values for tensors with data type CUDNN_DATA_INT8x32 and tensor format [TensorNchwVectC].
//   - [DropoutDescriptor.Restore]: restores a dropout descriptor to a previously saved-off state.
//   - [Handle.ScaleTensor]: scales all the elements of a tensor by a given factor.
//   - [ActivationDescriptor.Set]: initializes a previously created generic activation descriptor object.
//   - [ActivationDescriptor.SetActivationDescriptorSwishBeta]: sets the beta parameter of the SWISH activation function to swish_beta.
//   - [AttnDescriptor.Set]: configures a multihead attention descriptor that was previously created using the [CreateAttnDescriptor] function.
//   - [CTCLossDescriptor.SetCTCLossDescriptorEx]: is an extension of [CTCLossDescriptor.Set].
//   - [CTCLossDescriptor.Set]: sets a CTC loss function descriptor.
//   - [SetCallback]: sets the internal states of cuDNN error reporting functionality.
//   - [ConvolutionDescriptor.SetConvolution2dDescriptor]: initializes a previously created convolution descriptor object into a 2D correlation.
//   - [ConvolutionDescriptor.SetConvolutionGroupCount]: allows the user to specify the number of groups to be used in the associated convolution.
//   - [ConvolutionDescriptor.SetConvolutionMathType]: allows the user to specify whether or not the use of tensor op is permitted in the library routines associated with a given convolution descriptor.
//   - [ConvolutionDescriptor.SetConvolutionNdDescriptor]: initializes a previously created generic convolution descriptor object into a Nd correlation.
//   - [ConvolutionDescriptor.SetConvolutionReorderType]: sets the convolution reorder type for the given convolution descriptor.
//   - [DropoutDescriptor.Set]: initializes a previously created dropout descriptor object.
//   - [FilterDescriptor.SetFilter4dDescriptor]: initializes a previously created filter descriptor object into a 4D filter.
//   - [FilterDescriptor.SetFilterNdDescriptor]: initializes a previously created filter descriptor object.
//   - [FusedOpsConstParamPack.SetFusedOpsConstParamPackAttribute]: sets the descriptor pointed to by the param pointer input.
//   - [FusedOpsVariantParamPack.SetFusedOpsVariantParamPackAttribute]: sets the variable parameter pack descriptor.
//   - [LRNDescriptor.Set]: initializes a previously created LRN descriptor object.
//   - [OpTensorDescriptor.Set]: initializes a tensor pointwise math descriptor.
//   - [PoolingDescriptor.SetPooling2dDescriptor]: initializes a previously created generic pooling descriptor object into a 2D description.
//   - [PoolingDescriptor.SetPoolingNdDescriptor]: initializes a previously created generic pooling descriptor object.
//   - [RNNDataDescriptor.Set]: initializes a previously created RNN data descriptor object.
//   - [RNNDescriptor.Set]: initializes a previously created RNN descriptor object.
//   - [ReduceTensorDescriptor.Set]: initializes a previously created reduce tensor descriptor object.
//   - [SeqDataDescriptor.Set]: initializes a previously created sequence data descriptor object.
//   - [SpatialTransformerDescriptor.SetSpatialTransformerNdDescriptor]: initializes a previously created generic spatial transformer descriptor object.
//   - [Handle.SetTensor]: sets all the elements of a tensor to a given value.
//   - [TensorDescriptor.SetTensor4dDescriptor]: initializes a previously created generic tensor descriptor object into a 4D tensor.
//   - [TensorDescriptor.SetTensor4dDescriptorEx]: initializes a previously created generic tensor descriptor object into a 4D tensor, similarly to [TensorDescriptor.SetTensor4dDescriptor] but with the strides explicitly passed as parameters.
//   - [TensorDescriptor.SetTensorNdDescriptor]: initializes a previously created generic tensor descriptor object.
//   - [TensorDescriptor.SetTensorNdDescriptorEx]: initializes an Nd tensor descriptor.
//   - [TensorTransformDescriptor.Set]: initializes a tensor transform descriptor that was previously created using the [CreateTensorTransformDescriptor] function.
//   - [Handle.SpatialTfGridGeneratorBackward]: computes the gradient of a grid generation operation.
//   - [Handle.SpatialTfGridGeneratorForward]: generates a grid of coordinates in the input tensor corresponding to each pixel from the output tensor.
//   - [Handle.SpatialTfSamplerBackward]: computes the gradient of a sampling operation.
//   - [Handle.SpatialTfSamplerForward]: performs a sampler operation and generates the output tensor using the grid given by the grid generator.
//   - [Handle.TransformFilter]: converts the filter between different formats, data types, or dimensions based on the described transformation.
//   - [Handle.TransformTensor]: copies the scaled data from one tensor to another tensor with a different layout.
//   - [Handle.TransformTensorEx]: converts the tensor layouts between different formats.
//
// [cuDNN Documentation]: https://docs.nvidia.com/deeplearning/cudnn/
package cudnn

// Code generated by gen-nvidia from cudnn.h. DO NOT EDIT.
