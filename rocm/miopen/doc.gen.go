// Package miopen provides Go bindings for the MIOpen deep learning library.
//
// MIOpen is AMD's open-source deep learning primitives library, providing optimized
// implementations of convolutions, pooling, normalization, and other operations for
// AMD GPUs. These bindings use purego for dynamic loading without cgo.
//
// # Activation Functions
//
//   - [Handle.ActivationBackward]: Execute a activation backwards layer
//   - [Handle.ActivationForward]: Execute an activation forward layer
//
// # Other
//
//   - [Handle.AddLayerNormForward]: Execute a add and layernorm forward layer
//   - [BackendDescriptor.DestroyDescriptor]: Destroys an instance of [BackendDescriptor]
//   - [Handle.BackendExecute]: Executes a graph
//   - [BackendDescriptor.Finalize]: Finalizes a backend descriptor
//   - [BackendDescriptor.GetAttribute]: Retrieves backend descriptor's attribute
//   - [BackendDescriptor.Initialize]: Repurposes an instance of [BackendDescriptor]
//   - [BackendDescriptor.SetAttribute]: Sets an attribute of a descriptor
//   - [Handle.CTCLoss]: Execute forward inference for [Handle.CTCLoss] layer
//   - [Handle.CatForward]: Execute a cat forward layer
//   - [Handle.CompileFusionPlan]: Compiles the fusion plan
//   - [TensorDescriptor.DeriveBN]: Derive tensor for gamma and beta from input tensor descriptor
//   - [Handle.Destroy]: Destroys the MIOpen handle.
//   - [ActivationDescriptor.Destroy]: Destroys the activation descriptor object
//   - [CTCLossDescriptor.Destroy]: Destroys a CTC loss function descriptor object
//   - [ConvolutionDescriptor.Destroy]: Destroys the tensor descriptor object
//   - [DropoutDescriptor.Destroy]: Destroys the dropout descriptor object
//   - [FindOptions.Destroy]: Destroys [FindOptions] object.
//   - [FusionPlanDescriptor.DestroyFusionPlan]: [Handle.Destroy] the fusion plan descriptor object
//   - [LRNDescriptor.Destroy]: Destroys the LRN descriptor object
//   - [OperatorArgs.Destroy]: Destroys an operator argument object
//   - [PoolingDescriptor.Destroy]: Destroys the pooling descriptor object
//   - [Problem.Destroy]: Destroys a problem object.
//   - [RNNDescriptor.Destroy]: Destroys the tensor descriptor object
//   - [ReduceTensorDescriptor.Destroy]: [Handle.Destroy] the [Handle.ReduceTensor] descriptor object
//   - [SeqTensorDescriptor.Destroy]: Destroys the sequence data tensor descriptor
//   - [Solution.Destroy]: Destroys solution object.
//   - [TensorDescriptor.Destroy]: Destroys the tensor descriptor
//   - [Handle.EnableProfiling]: Enable profiling to retrieve kernel time
//   - [Handle.ExecuteFusionPlan]: Executes the fusion plan
//   - [Problem.FuseProblems]: Fuse two problems into a single one.
//   - [Handle.FusedAdam]: Perform Fused Adam optimization for a single tensor (Adaptive Moment Estimation).
//   - [Handle.FusedAdamWithOutput]: Execute single tensor Adam optimization and receive the result in a separate output tensor.
//   - [FusionPlanDescriptor.PlanConvolutionGetAlgo]: Returns the supported algorithms for the convolution operator in the Fusion Plan
//   - [FusionPlanDescriptor.PlanConvolutionSetAlgo]: Requests the fusion runtime to choose a particular algorithm for the added convolution operation
//   - [FusionPlanDescriptor.PlanGetOp]: Allows access to the operators in a fusion plan This api call does bounds checking on the supplied...
//   - [Handle.FusionPlanGetWorkSpaceSize]: Query the workspace size required for the fusion plan
//   - [Handle.GLUBackward]: Execute a GLU backward layer
//   - [Handle.GLUForward]: Execute a GLU forward layer
//   - [Handle.GroupNormForward]: Execute a groupnorm forward layer
//   - [ConvolutionDescriptor.Init]: Creates a 2-D convolution layer descriptor
//   - [ConvolutionDescriptor.InitConvolutionNdDescriptor]: Creates a N-dimensional convolution layer descriptor
//   - [Handle.KthvalueForward]: Execute a Kthvalue forward layer
//   - [Handle.LRNBackward]: Execute a LRN backward layer
//   - [Handle.LRNForward]: Execute a LRN forward layer
//   - [TensorDescriptor.LRNGetWorkSpaceSize]: Determine the workspace requirements.
//   - [Handle.LayerNormForward]: Execute a layernorm forward layer
//   - [NewSolution]: Loads solution object from binary data.
//   - [Handle.OpTensor]: Execute element-wise tensor operations
//   - [Handle.PReLUBackward]: Execute a PReLU backward layer
//   - [Handle.ReduceCalculationForward]: Execute a reducecalculation forward layer
//   - [Handle.ReduceExtremeForward]: Find the the extreme (minimum, maximum) value and index of a tensor across Dimension.
//   - [Handle.ReduceTensor]: TensorReduce function doing reduction on tensor A by implementing C = alpha * reduceOp(A) + beta * C
//   - [DropoutDescriptor.Restore]: Restore the dropout descriptor to a saved state
//   - [Handle.RoPEBackward]: Execute a rope backward layer
//   - [Handle.RoPEForward]: Execute a rope forward layer
//   - [Handle.RunSolution]: Runs the solution using the passed in buffers.
//   - [Solution.Save]: Saves a solution object as binary data.
//   - [Handle.ScaleTensor]: Scales all elements in a tensor by a single value.
//   - [Handle.SoftMarginLossBackward]: Execute a SoftMarginLoss backward layer
//   - [Handle.SoftMarginLossForward]: Execute a SoftMarginLoss forward layer
//   - [Handle.T5LayerNormBackward]: Execute a T5layernorm backward layer
//   - [Handle.T5LayerNormForward]: Execute a T5layernorm forward layer
//   - [Handle.TransformTensor]: Copies one tensor to another tensor with a different layout/scale.
//   - [Handle.TransformersAdamW]: Implements Adam algorithm with weight decay fix as introduced in Decoupled Weight Decay...
//   - [Handle.TransformersAdamWWithOutput]: Execute single tensor Adam optimization and receive the result in a separate output tensor.
//
// # Batch Normalization
//
//   - [Handle.BatchNormalizationBackward]: Execute backwards propagation layer for batch normalization
//   - [Handle.BatchNormalizationBackward_V2]: Execute backwards propagation layer for batch normalization
//   - [Handle.BatchNormalizationForwardInference]: Execute forward inference layer for batch normalization
//   - [Handle.BatchNormalizationForwardInference_V2]: Execute forward inference layer for batch normalization
//   - [Handle.BatchNormalizationForwardTraining]: Execute forward training layer for batch normalization
//   - [Handle.BatchNormalizationForwardTraining_V2]: Execute forward training layer for batch normalization
//
// # Convolution Operations
//
//   - [Handle.ConvolutionBackwardBias]: Calculates the gradient with respect to the bias.
//   - [Handle.ConvolutionBackwardData]: Execute a backward data convolution layer
//   - [Handle.ConvolutionBackwardDataCompileSolution]: Compiles the solution provided by the user, this solution may be acquired by the...
//   - [Handle.ConvolutionBackwardDataGetSolution]: Query the applicable solutions for a backward convolution w-r-t data as described by input, output...
//   - [Handle.ConvolutionBackwardDataGetSolutionCount]: Query the maximum number of solutions applicable for the given input/output and weights tensor...
//   - [Handle.ConvolutionBackwardDataGetSolutionWorkspaceSize]: Returns the workspace size required for a particular solution id.
//   - [Handle.ConvolutionBackwardDataGetWorkSpaceSize]: Query the workspace size required for a backward data convolution algorithm.
//   - [Handle.ConvolutionBackwardDataImmediate]: Executes the Backward convolution w-r-t data operation based on the provided solution ID.
//   - [Handle.ConvolutionBackwardWeights]: Execute a backward weights convolution layer
//   - [Handle.ConvolutionBackwardWeightsCompileSolution]: Compiles the solution provided by the user, this solution may be acquired by the...
//   - [Handle.ConvolutionBackwardWeightsGetSolution]: Query the applicable solutions for a backward convolution w-r-t weights as described by input,...
//   - [Handle.ConvolutionBackwardWeightsGetSolutionCount]: Query the maximum number of solutions applicable for the given input/output and weights tensor...
//   - [Handle.ConvolutionBackwardWeightsGetSolutionWorkspaceSize]: Returns the workspace size required for a particular solution id.
//   - [Handle.ConvolutionBackwardWeightsGetWorkSpaceSize]: Get the GPU memory required for the backward weights convolution algorithm.
//   - [Handle.ConvolutionBackwardWeightsImmediate]: Executes the Backward convolution w-r-t weights operation based on the provided solution ID.
//   - [Handle.ConvolutionBiasActivationForward]: Prepares and executes the Convlution+Bias+Activation Fusion.
//   - [Handle.ConvolutionForward]: Execute a forward convolution layer
//   - [Handle.ConvolutionForwardBias]: Calculate element-wise scale and shift of a tensor via a bias tensor
//   - [Handle.ConvolutionForwardCompileSolution]: Compiles the solution provided by the user, this solution may be acquired by the...
//   - [Handle.ConvolutionForwardGetSolution]: Query the applicable solutions for a convolution configuration described by input, output and...
//   - [Handle.ConvolutionForwardGetSolutionCount]: Query the maximum number of solutions applicable for the given input/output and weights tensor...
//   - [Handle.ConvolutionForwardGetSolutionWorkspaceSize]: Returns the workspace size required for a particular solution id.
//   - [Handle.ConvolutionForwardGetWorkSpaceSize]: Query the workspace size required for a forward convolution algorithm.
//   - [Handle.ConvolutionForwardImmediate]: Executes the Forward convolution operation based on the provided solution ID.
//
// # Handle Creation
//
//   - [NewHandle]: Method to create the MIOpen handle object.
//   - [NewActivationDescriptor]: Creates the Activation descriptor object
//   - [ActivationDescriptor.CreateActivationProblem]: Initializes a problem object describing an activation operation.
//   - [NewProblemBatchnorm]: Initializes a problem object describing an activation operation.
//   - [NewProblemBias]: Initializes a problem object describing an bias operation.
//   - [NewCTCLossDescriptor]: [NewHandle] a CTC loss function Descriptor
//   - [ConvolutionDescriptor.CreateConvProblem]: Initializes a problem object describing a convolution operation.
//   - [NewConvolutionDescriptor]: Creates a convolution layer descriptor
//   - [NewDropoutDescriptor]: Creates the dropout descriptor object
//   - [NewFindOptions]: Initializes [FindOptions] object.
//   - [TensorDescriptor.CreateFusionPlan]: Creates the kenrel fusion plan descriptor object
//   - [NewLRNDescriptor]: Creates a local response normalization (LRN) layer descriptor
//   - [NewMhaDescriptor]: Creates the mha descriptor object
//   - [FusionPlanDescriptor.CreateOpActivationBackward]: Creates a backward activation operator.
//   - [FusionPlanDescriptor.CreateOpActivationForward]: Creates a forward activation operator.
//   - [FusionPlanDescriptor.CreateOpBatchNormBackward]: Creates a back propagation batch normalization operator.
//   - [FusionPlanDescriptor.CreateOpBatchNormForward]: Creates a forward training batch normalization operator.
//   - [FusionPlanDescriptor.CreateOpBatchNormInference]: Creates a forward inference batch normalization operator.
//   - [FusionPlanDescriptor.CreateOpBiasForward]: Creates a forward bias operator.
//   - [FusionPlanDescriptor.CreateOpConvForward]: Creates forward convolution operator.
//   - [NewOperatorArgs]: Creates an operator argument object
//   - [NewPoolingDescriptor]: Creates a pooling layer descriptor
//   - [NewRNNDescriptor]: [NewHandle] a RNN layer Descriptor
//   - [NewReduceTensorDescriptor]: Creates the [Handle.ReduceTensor] descriptor object
//   - [NewSeqTensorDescriptor]: [NewHandle] a Tensor Descriptor for sequence data
//   - [NewSoftmaxDescriptor]: Creates the Softmax descriptor object
//   - [SoftmaxDescriptor.CreateSoftmaxProblem]: Initializes a problem object describing a softmax operation.
//   - [NewTensorDescriptor]: [NewHandle] a Tensor Descriptor
//   - [NewHandleWithStream]: [NewHandle] a MIOpen handle with an accelerator stream.
//
// # Dropout
//
//   - [Handle.DropoutBackward]: Execute backward dropout operation
//   - [Handle.DropoutForward]: Execute forward dropout operation
//   - [TensorDescriptor.DropoutGetReserveSpaceSize]: Query the amount of memory required to run dropout
//   - [Handle.DropoutGetStatesSize]: Query the amount of memory required to store the states of the random number generators
//
// # Algorithm Search
//
//   - [Handle.FindConvolutionBackwardDataAlgorithm]: Search and run the backwards data convolution algorithms and return a list of kernel times.
//   - [Handle.FindConvolutionBackwardWeightsAlgorithm]: Search and run the backwards weights convolutional algorithms and return a list of kernel times.
//   - [Handle.FindConvolutionForwardAlgorithm]: Search and run the forward convolutional algorithms and return a list of kernel times.
//   - [Handle.FindSolutions]: Finds solutions to a problem by running different applicable solutions.
//
// # Queries
//
//   - [PoolingDescriptor.Get2d]: Gets a 2-D pooling layer descriptor details
//   - [TensorDescriptor.Get4d]: Get the details of the tensor descriptor
//   - [ActivationDescriptor.Get]: Gets the activation layer descriptor details
//   - [CTCLossDescriptor.Get]: Retrieves a CTC loss function descriptor's details
//   - [Handle.GetCTCLossWorkspaceSize]: Query the amount of memory required to execute [Handle.CTCLoss]
//   - [ConvolutionDescriptor.GetConvolutionAttribute]: Get the attribute of the convolution descriptor
//   - [ConvolutionDescriptor.Get]: Retrieves a 2-D convolution layer descriptor's details
//   - [ConvolutionDescriptor.GetConvolutionForwardOutputDim]: Get the shape of a resulting 4-D tensor from a 2-D convolution
//   - [ConvolutionDescriptor.GetConvolutionGroupCount]: Get the number of groups to be used in Group/Depthwise convolution
//   - [ConvolutionDescriptor.GetConvolutionNdDescriptor]: Retrieves a N-dimensional convolution layer descriptor's details
//   - [ConvolutionDescriptor.GetConvolutionNdForwardOutputDim]: Get the shape of a resulting N-dimensional tensor from a (N-2)-dimensional convolution
//   - [ConvolutionDescriptor.GetConvolutionSpatialDim]: Retrieves the spatial dimension of a convolution layer descriptor
//   - [DropoutDescriptor.Get]: Get the details of the dropout descriptor
//   - [Handle.GetGetitemWorkspaceSize]: Helper function to query the minimum workspace size required by the getitem call
//   - [Handle.GetKernelTime]: Get time for last kernel launched
//   - [LRNDescriptor.Get]: Gets a LRN layer descriptor details
//   - [MhaDescriptor.Get]: Gets the Mha descriptor details
//   - [PoolingDescriptor.GetNd]: Get details of a N-D pooling layer descriptor
//   - [Handle.GetPReLUBackwardWorkspaceSize]: Helper function to query the minimum workspace size required by the PReLU backward call
//   - [PoolingDescriptor.GetPoolingForwardOutputDim]: Gets the shape of the output tensor for 2-D pooling
//   - [PoolingDescriptor.GetPoolingIndexType]: Get the index data type for pooling layer.
//   - [PoolingDescriptor.GetPoolingNdForwardOutputDim]: Gets the shape of the output tensor for N-D pooling
//   - [PoolingDescriptor.GetPoolingWorkSpaceIndexMode]: Get workspace index mode for pooling layer.
//   - [SeqTensorDescriptor.GetRNNData]: Get shape of RNN seqData tensor
//   - [RNNDescriptor.Get]: Retrieves a RNN layer descriptor's details
//   - [RNNDescriptor.GetRNNDescriptor_V2]: Retrieves a RNN layer descriptor's details version 2.
//   - [Handle.GetRNNHiddenTensorSize]: Obtain a the size in bytes of the RNN hidden tensor
//   - [Handle.GetRNNInputTensorSize]: Obtain a the size in bytes of the RNN input tensor
//   - [Handle.GetRNNLayerBias]: Gets a bias for a specific layer in an RNN stack
//   - [RNNDescriptor.GetRNNLayerBiasOffset]: Gets a bias index offset for a specific layer in an RNN stack
//   - [Handle.GetRNNLayerBiasSize]: Gets the number of bytes of a bias
//   - [Handle.GetRNNLayerParam]: Gets a weight matrix for a specific layer in an RNN stack
//   - [RNNDescriptor.GetRNNLayerParamOffset]: Gets an index offset for a specific weight matrix for a layer in the RNN stack
//   - [Handle.GetRNNLayerParamSize]: Gets the number of bytes of a parameter matrix
//   - [RNNDescriptor.GetRNNPaddingMode]: This function retrieves the RNN padding mode from the RNN descriptor.
//   - [Handle.GetRNNParamsDescriptor]: Obtain a weight tensor descriptor for RNNs
//   - [Handle.GetRNNParamsSize]: Query the amount of parameter memory required for RNN training
//   - [Handle.GetRNNTempSpaceSizes]: Query the amount of additional memory required for this RNN layer execution.
//   - [Handle.GetRNNTrainingReserveSize]: Query the amount of memory required for RNN training
//   - [Handle.GetRNNWorkspaceSize]: Query the amount of memory required to execute the RNN layer
//   - [Handle.GetReduceCalculationWorkspaceSize]: Helper function to query the minimum workspace size required by the [Handle.ReduceTensor] call
//   - [ReduceTensorDescriptor.Get]: Query a [Handle.ReduceTensor] descriptor object
//   - [Handle.GetReductionIndicesSize]: Helper function to query the minimum index space size required by the [Handle.ReduceTensor] call
//   - [Handle.GetReductionWorkspaceSize]: Helper function to query the minimum workspace size required by the [Handle.ReduceTensor] call
//   - [SoftmaxDescriptor.Get]: Gets the softmax layer descriptor details
//   - [Solution.GetSolutionSize]: Reads the expected size of a solution.
//   - [Solution.GetSolutionSolverId]: Reads id of the solver referred by the solution.
//   - [Solution.GetSolutionTime]: Reads the time spent to execute the solution the last it was run.
//   - [Solution.GetSolutionWorkspaceSize]: Reads the amount of workspace required to exectute the solution.
//   - [GetSolverIdConvAlgorithm]: Gets the convolution algorithm implemented by a solver.
//   - [Handle.GetStream]: Get the previously created accelerator command queue
//   - [Handle.GetT5LayerNormBackwardWorkspaceSize]: Helper function to query the minimum workspace size required by the T5layernorm backward call
//   - [TensorDescriptor.Get]: Get the details of the N-dimensional tensor descriptor.
//   - [TensorDescriptor.GetTensorDescriptorSize]: Set shape of N-dimensional tensor
//   - [TensorDescriptor.GetTensorNumBytes]: Returns number of bytes associated with tensor descriptor
//   - [GetVersion]: Method to return version of MIOpen
//   - [Handle.GetitemBackward]: Execute a getitem backward layer
//
// # Pooling Operations
//
//   - [Handle.PoolingBackward]: Execute a backward pooling layer
//   - [Handle.PoolingForward]: Execute a forward pooling layer
//   - [TensorDescriptor.PoolingGetWorkSpaceSize]: Get the amount of GPU memory required for pooling
//   - [PoolingDescriptor.GetWorkSpaceSizeV2]: Get the amount of GPU memory required for pooling
//
// # Recurrent Neural Networks
//
//   - [Handle.RNNBackwardData]: Execute backward data for recurrent layer
//   - [Handle.RNNBackwardWeights]: Execute backward weights for recurrent layer
//   - [Handle.RNNForward]: Execute forward training for recurrent layer.
//   - [Handle.RNNForwardInference]: Execute forward inference for RNN layer
//   - [Handle.RNNForwardTraining]: Execute forward training for recurrent layer
//
// # Descriptor Configuration
//
//   - [PoolingDescriptor.Set2d]: Sets a 2-D pooling layer descriptor details.
//   - [TensorDescriptor.Set4d]: Set shape of 4D tensor
//   - [TensorDescriptor.Set4dTensorDescriptorEx]: Set shape and stride of 4D tensor
//   - [ActivationDescriptor.Set]: Sets the activation layer descriptor details
//   - [Handle.SetAllocator]: Set allocator for previously created [Handle]
//   - [CTCLossDescriptor.Set]: Set the details of a CTC loss function descriptor
//   - [ConvolutionDescriptor.SetConvolutionAttribute]: Set the attribute of the convolution descriptor
//   - [ConvolutionDescriptor.SetConvolutionGroupCount]: Set the number of groups to be used in Group/Depthwise convolution
//   - [DropoutDescriptor.Set]: Initialize the dropout descriptor
//   - [FindOptions.SetFindOptionAttachBinaries]: Forces library to attach kernel binaries to solutions for later saving.
//   - [FindOptions.SetFindOptionPreallocatedTensor]: Attaches a preallocated tensor to find options.
//   - [FindOptions.SetFindOptionPreallocatedWorkspace]: Attaches the preallocated workspace to find options.
//   - [FindOptions.SetFindOptionResultsOrder]: Sets the results order find option.
//   - [FindOptions.SetFindOptionTuning]: Sets the tuning find option.
//   - [FindOptions.SetFindOptionWorkspaceLimit]: Sets the workspace limit find option.
//   - [LRNDescriptor.Set]: Sets a LRN layer descriptor details
//   - [MhaDescriptor.Set]: Sets the Mha descriptor details
//   - [PoolingDescriptor.SetNd]: Set details of a N-D pooling layer descriptor
//   - [TensorDescriptor.SetNdTensorDescriptorWithLayout]: Set shape of ND tensor with specific layout
//   - [OperatorArgs.SetOpArgsActivBackward]: Sets the arguments for backward activation op
//   - [OperatorArgs.SetOpArgsActivForward]: Sets the arguments for forward activation op
//   - [OperatorArgs.SetOpArgsBatchNormBackward]: Sets the arguments for backward batch normalization op
//   - [OperatorArgs.SetOpArgsBatchNormForward]: Sets the arguments for forward batch normalization op
//   - [OperatorArgs.SetOpArgsBatchNormInference]: Sets the arguments for inference batch normalization op
//   - [OperatorArgs.SetOpArgsBiasForward]: Sets the arguments for forward bias op
//   - [OperatorArgs.SetOpArgsConvForward]: Sets the arguments for forward convolution op
//   - [PoolingDescriptor.SetPoolingIndexType]: Set index data type for pooling layer.
//   - [PoolingDescriptor.SetPoolingWorkSpaceIndexMode]: Set workspace index mode for pooling layer.
//   - [Problem.SetProblemTensorDescriptor]: Sets a tensor descriptor for the specified argument.
//   - [SeqTensorDescriptor.SetRNNData]: Set shape of RNN seqData tensor
//   - [RNNDescriptor.Set]: Set the details of the RNN descriptor
//   - [RNNDescriptor.SetRNNDescriptor_V2]: Set the details of the RNN descriptor version 2.
//   - [Handle.SetRNNLayerBias]: Sets a bias for a specific layer in an RNN stack
//   - [Handle.SetRNNLayerParam]: Sets a weight matrix for a specific layer in an RNN stack
//   - [RNNDescriptor.SetRNNPaddingMode]: Sets a bias for a specific layer in an RNN stack
//   - [ReduceTensorDescriptor.Set]: Initialize a [Handle.ReduceTensor] descriptor object
//   - [SoftmaxDescriptor.Set]: Sets the softmax descriptor details
//   - [Handle.SetStream]: Set accelerator command queue previously created
//   - [Handle.SetTensor]: Fills a tensor with a single value.
//   - [TensorDescriptor.SetTensorCastType]: Set the tensor cast type
//   - [TensorDescriptor.Set]: Set shape of N-dimensional tensor
//   - [ConvolutionDescriptor.SetTransposeConvNdOutputPadding]: Set the output padding to be used in N-dimensional Transpose convolution
//   - [ConvolutionDescriptor.SetTransposeConvOutputPadding]: Set the output padding to be used in 2-D Transpose convolution
//
// # Softmax Operations
//
//   - [Handle.SoftmaxBackward]: Execute a softmax backwards layer
//   - [Handle.SoftmaxBackward_V2]: Execute a softmax backwards layer with expanded modes and algorithms
//   - [Handle.SoftmaxForward]: Execute a softmax forward layer
//   - [Handle.SoftmaxForward_V2]: Execute a softmax forward layer with expanded modes and algorithms
//
// [MIOpen Documentation]: https://rocm.docs.amd.com/projects/MIOpen/en/latest/
package miopen

// Code generated by gen-rocm from miopen.h. DO NOT EDIT.
