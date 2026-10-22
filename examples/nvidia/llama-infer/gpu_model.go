package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// GPULayerWeights holds GPU-resident parameter slices for one transformer layer.
// Matrices are stored as the model's weight dtype; norms and biases are float32.
type GPULayerWeights struct {
	InputNorm cudamem.Slice[float32]
	PostNorm  cudamem.Slice[float32]
	WQ        deviceWeight
	WK        deviceWeight
	WV        deviceWeight
	QBias     cudamem.Slice[float32]
	KBias     cudamem.Slice[float32]
	VBias     cudamem.Slice[float32]
	WO        deviceWeight
	WGate     deviceWeight
	WUp       deviceWeight
	WDown     deviceWeight
}

// GPUModelWeights holds all model parameters in device GPU memory. A tied
// LMHead borrows EmbedTokens' memory.
type GPUModelWeights struct {
	DType       weightDType
	EmbedTokens deviceWeight
	Layers      []GPULayerWeights
	Norm        cudamem.Slice[float32]
	LMHead      deviceWeight
}

// GPUScratchpad holds intermediate GPU-resident activation and logit buffers.
type GPUScratchpad struct {
	X            cudamem.Slice[float32]
	NormX        cudamem.Slice[float32]
	Q            cudamem.Slice[float32]
	K            cudamem.Slice[float32]
	V            cudamem.Slice[float32]
	AttnOut      cudamem.Slice[float32]
	Gate         cudamem.Slice[float32]
	Up           cudamem.Slice[float32]
	FFNAct       cudamem.Slice[float32]
	FFNOut       cudamem.Slice[float32]
	Logits       cudamem.Slice[float32]
	TokenIn      cudamem.Slice[int32]
	TokenOut     cudamem.Slice[int32]
	HostTokenIn  []int32
	HostTokenOut []int32
}

// NewGPUScratchpad allocates all intermediate scratchpad activation buffers on dev.
// On error it frees whatever it allocated. If that free fails too, it returns
// the scratchpad holding the buffers it could not free, with both errors.
func NewGPUScratchpad(dev *cuda.Device, cfg *ModelConfig) (*GPUScratchpad, error) {
	return newGPUScratchpadWith(cfg,
		func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) },
		func(n int) (cudamem.Slice[int32], error) { return cudamem.Alloc[int32](dev, n) },
		freeOwned[cudamem.Slice[float32]], freeOwned[cudamem.Slice[int32]])
}

// newGPUScratchpadWith is NewGPUScratchpad with its device operations as
// arguments: allocF32 and allocI32 allocate n elements, and freeF32 and
// freeI32 free a buffer, which they must zero only when the free
// succeeds, as freeOwned does. A buffer an allocation returns with its
// error is kept, and freed with the rest.
func newGPUScratchpadWith(cfg *ModelConfig,
	allocF32 func(n int) (cudamem.Slice[float32], error),
	allocI32 func(n int) (cudamem.Slice[int32], error),
	freeF32 func(*cudamem.Slice[float32]) error,
	freeI32 func(*cudamem.Slice[int32]) error) (res *GPUScratchpad, err error) {
	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	sp := &GPUScratchpad{
		HostTokenIn:  make([]int32, 1),
		HostTokenOut: make([]int32, 1),
	}
	defer func() {
		if err != nil {
			if ferr := sp.release(freeF32, freeI32); ferr != nil {
				err = errors.Join(err, ferr)
				res = sp
			}
		}
	}()
	for _, b := range []struct {
		dst *cudamem.Slice[float32]
		n   int
	}{
		{&sp.X, cfg.HiddenSize},
		{&sp.NormX, cfg.HiddenSize},
		{&sp.Q, qDim},
		{&sp.K, kvDim},
		{&sp.V, kvDim},
		{&sp.AttnOut, qDim},
		{&sp.Gate, cfg.IntermediateSize},
		{&sp.Up, cfg.IntermediateSize},
		{&sp.FFNAct, cfg.IntermediateSize},
		{&sp.FFNOut, cfg.HiddenSize},
		{&sp.Logits, cfg.VocabSize},
	} {
		if *b.dst, err = allocF32(b.n); err != nil {
			return nil, fmt.Errorf("alloc scratch: %w", err)
		}
	}
	if sp.TokenIn, err = allocI32(1); err != nil {
		return nil, fmt.Errorf("alloc token in: %w", err)
	}
	if sp.TokenOut, err = allocI32(1); err != nil {
		return nil, fmt.Errorf("alloc token out: %w", err)
	}
	return sp, nil
}

// freeOwned frees *s and then zeroes it. If the free fails, *s is kept
// unchanged, so that the caller still owns it and can retry, and the error
// is returned as is. A zero cudamem.Slice frees as a no-op.
func freeOwned[S interface{ Free() error }](s *S) error {
	if err := (*s).Free(); err != nil {
		return err
	}
	var zero S
	*s = zero
	return nil
}

// Free releases all GPU scratchpad memory, each buffer once. A buffer
// whose free fails is kept, so that Free can be called again; the errors are
// joined and keep their identity for errors.Is. After a successful Free,
// calling it again does nothing.
func (sp *GPUScratchpad) Free() error {
	return sp.release(freeOwned[cudamem.Slice[float32]], freeOwned[cudamem.Slice[int32]])
}

// release frees every scratchpad buffer with the given functions, which
// must zero a buffer only when its free succeeds, as freeOwned does.
func (sp *GPUScratchpad) release(f32 func(*cudamem.Slice[float32]) error, i32 func(*cudamem.Slice[int32]) error) error {
	var errs []error
	for _, s := range []*cudamem.Slice[float32]{
		&sp.X, &sp.NormX, &sp.Q, &sp.K, &sp.V, &sp.AttnOut,
		&sp.Gate, &sp.Up, &sp.FFNAct, &sp.FFNOut, &sp.Logits,
	} {
		errs = append(errs, f32(s))
	}
	errs = append(errs, i32(&sp.TokenIn), i32(&sp.TokenOut))
	return errors.Join(errs...)
}

// matrices returns pointers to every matrix weight in w, in upload order.
func (w *GPUModelWeights) matrices() []*deviceWeight {
	m := []*deviceWeight{&w.EmbedTokens, &w.LMHead}
	for i := range w.Layers {
		l := &w.Layers[i]
		m = append(m, &l.WQ, &l.WK, &l.WV, &l.WO, &l.WGate, &l.WUp, &l.WDown)
	}
	return m
}

// vectors returns pointers to every norm and bias in w, in upload order.
func (w *GPUModelWeights) vectors() []*cudamem.Slice[float32] {
	v := []*cudamem.Slice[float32]{&w.Norm}
	for i := range w.Layers {
		l := &w.Layers[i]
		v = append(v, &l.InputNorm, &l.PostNorm, &l.QBias, &l.KBias, &l.VBias)
	}
	return v
}

// Bytes returns the size in bytes of the device buffer w refers to, or 0 for
// no weight. A borrowed weight reports the full size of the buffer it views.
func (w deviceWeight) Bytes() int64 {
	if w.buf == nil {
		return 0
	}
	switch b := w.buf.(type) {
	case cudamem.Slice[float32]:
		return int64(b.Len()) * 4
	case cudamem.Slice[uint16]:
		return int64(b.Len()) * 2
	case interface{ Bytes() int64 }:
		return b.Bytes()
	case interface{ Len() int }:
		switch w.dtype {
		case weightF32:
			return int64(b.Len()) * 4
		case weightBF16:
			return int64(b.Len()) * 2
		}
	}
	return 0
}

// StreamedBytesPerToken returns the total bytes of model weights and norm/bias
// vectors streamed across one single-token decode step. EmbedTokens is a
// single-row lookup and is excluded, whereas LMHead is streamed across the
// full vocabulary matrix even when tied to EmbedTokens.
func (w *GPUModelWeights) StreamedBytesPerToken() int64 {
	if w == nil {
		return 0
	}
	total := w.LMHead.Bytes() + int64(w.Norm.Len())*4
	for i := range w.Layers {
		l := &w.Layers[i]
		total += l.WQ.Bytes() + l.WK.Bytes() + l.WV.Bytes() + l.WO.Bytes() +
			l.WGate.Bytes() + l.WUp.Bytes() + l.WDown.Bytes()
		total += int64(l.InputNorm.Len()+l.PostNorm.Len()+l.QBias.Len()+l.KBias.Len()+l.VBias.Len()) * 4
	}
	return total
}

// Free releases all GPU weight memory, each allocation once: a tied LMHead
// is a borrowed view and never frees the embedding table. A weight whose
// free fails is kept, so that Free can be called again; the errors are
// joined and keep their identity for errors.Is. After a successful Free,
// calling it again does nothing.
func (w *GPUModelWeights) Free() error {
	return w.free(freeOwned[cudamem.Slice[float32]])
}

// free releases every matrix and frees every vector with vec, which must
// zero a vector only when its free succeeds, as freeOwned does.
func (w *GPUModelWeights) free(vec func(*cudamem.Slice[float32]) error) error {
	var errs []error
	for _, m := range w.matrices() {
		errs = append(errs, m.release())
	}
	for _, v := range w.vectors() {
		errs = append(errs, vec(v))
	}
	return errors.Join(errs...)
}

// UploadModelToGPU transfers all host model weights into GPU memory with
// float32 matrices.
func UploadModelToGPU(dev *cuda.Device, w *ModelWeights, cfg *ModelConfig) (*GPUModelWeights, error) {
	return uploadModel(dev, w, cfg, weightF32)
}

// tiedHead reports whether w's lm_head is the embedding table itself, as
// LoadModel makes it when the checkpoint has no lm_head.
func tiedHead(w *ModelWeights) bool {
	return len(w.LMHead) > 0 && len(w.LMHead) == len(w.EmbedTokens) && &w.LMHead[0] == &w.EmbedTokens[0]
}

// uploadModel transfers all host model weights into GPU memory, storing the
// matrices as dtype. A tied lm_head is uploaded once and borrowed. On error
// it frees whatever it uploaded. If that free fails too, it returns the
// weights it could not free, with both errors, so the caller can retry Free.
func uploadModel(dev *cuda.Device, w *ModelWeights, cfg *ModelConfig, dtype weightDType) (*GPUModelWeights, error) {
	return uploadModelWith(w, cfg, dtype,
		func(host []float32, dtype weightDType) (deviceWeight, error) { return uploadWeight(dev, host, dtype) },
		func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) },
		(*cudamem.Slice[float32]).CopyFromHost,
		freeOwned[cudamem.Slice[float32]])
}

// uploadModelWith is uploadModel with its device operations as arguments:
// upload uploads a matrix as dtype, alloc allocates a vector of n elements,
// fill copies host values into a vector, and free frees a vector, which it
// must zero only when the free succeeds, as freeOwned does.
func uploadModelWith(w *ModelWeights, cfg *ModelConfig, dtype weightDType,
	upload func(host []float32, dtype weightDType) (deviceWeight, error),
	alloc func(n int) (cudamem.Slice[float32], error),
	fill func(dst *cudamem.Slice[float32], host []float32) error,
	free func(*cudamem.Slice[float32]) error) (res *GPUModelWeights, err error) {
	if len(w.Layers) != cfg.NumHiddenLayers {
		return nil, fmt.Errorf("model has %d layers, config says %d", len(w.Layers), cfg.NumHiddenLayers)
	}
	gw := &GPUModelWeights{DType: dtype, Layers: make([]GPULayerWeights, cfg.NumHiddenLayers)}
	defer func() {
		if err != nil {
			if ferr := gw.free(free); ferr != nil {
				err = errors.Join(err, ferr)
				res = gw
			}
		}
	}()

	mats := gw.matrices()
	msrc := [][]float32{w.EmbedTokens, w.LMHead}
	for i := range w.Layers {
		l := &w.Layers[i]
		msrc = append(msrc, l.WQ, l.WK, l.WV, l.WO, l.WGate, l.WUp, l.WDown)
	}
	for i, host := range msrc {
		if i == 1 && tiedHead(w) {
			gw.LMHead = gw.EmbedTokens.borrow()
			continue
		}
		if len(host) == 0 {
			return nil, fmt.Errorf("upload matrix weight %d: empty", i)
		}
		if *mats[i], err = upload(host, dtype); err != nil {
			return nil, fmt.Errorf("upload matrix weight %d: %w", i, err)
		}
	}

	vecs := gw.vectors()
	vsrc := [][]float32{w.Norm}
	for i := range w.Layers {
		l := &w.Layers[i]
		vsrc = append(vsrc, l.InputNorm, l.PostNorm, l.QBias, l.KBias, l.VBias)
	}
	for i, host := range vsrc {
		if len(host) == 0 {
			continue // absent optional tensor, such as a bias
		}
		if *vecs[i], err = alloc(len(host)); err != nil {
			return nil, fmt.Errorf("upload vector weight %d: %w", i, err)
		}
		if err = fill(vecs[i], host); err != nil {
			return nil, fmt.Errorf("upload vector weight %d: %w", i, err)
		}
	}
	return gw, nil
}

// fusedKernelsQualified reports whether FusedRMSNormQKV, FusedSwiGLUDown,
// FusedRMSNormQKVBF16 and FusedSwiGLUDownBF16 have passed parity on a device.
// When false, -fuse-egraph is refused rather than silently running unqualified
// kernels.
var fusedKernelsQualified = true

// attnThreads is the decode attention block width. CausalFlashAttentionDecode
// accepts any width from 1 to 1024; shared memory follows from it through
// cudaattn.DecodeSharedMemoryBytes.
const attnThreads = 128

// GPUForwardPass executes the complete transformer forward pass 100% on device GPU memory.
func GPUForwardPass(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenID int32,
	pos int,
) (int32, error) {
	return GPUForwardPassStream(gpuW, cfg, kv, sp, gm, attnFn, sampleFn, tokenID, pos, nil)
}

// GPUForwardPassStream executes one full transformer decoder forward step on the specified stream.
func GPUForwardPassStream(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenID int32,
	pos int,
	stream *cuda.Stream,
) (int32, error) {
	token, err := enqueueGPUForwardPass(gpuW, cfg, kv, sp, gm, attnFn, sampleFn, tokenID, pos, stream)
	if stream == nil {
		return token, err
	}
	// Even an enqueue error may leave work using the scratch buffers. Drain
	// before returning ownership to the caller, and report a failed drain
	// after any enqueue error.
	if waitErr := stream.Synchronize(context.Background()); waitErr != nil {
		err = errors.Join(err, waitErr)
	}
	if err != nil {
		return 0, err
	}
	return sp.HostTokenOut[0], nil
}

// checkToken reports a token id outside a vocabulary of vocab tokens. The
// embedding kernels index the table with it unchecked.
func checkToken(id int32, vocab int) error {
	if id < 0 || int64(id) >= int64(vocab) {
		return fmt.Errorf("token %d outside vocabulary of %d", id, vocab)
	}
	return nil
}

// checkPos reports a position outside a KV cache of maxSeq positions.
// Attention reads positions [0, pos], so pos must index the cache.
func checkPos(pos, maxSeq int) error {
	if pos < 0 || pos >= maxSeq {
		return fmt.Errorf("position %d outside the kv cache of %d", pos, maxSeq)
	}
	return nil
}

// enqueueGPUForwardPass leaves stream work pending when stream is non-nil.
// The caller must drain it before reading HostTokenOut or releasing buffers.
// With a nil stream it performs the synchronous default-stream operation.
func enqueueGPUForwardPass(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	kv *cudakv.Cache,
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenID int32,
	pos int,
	stream *cuda.Stream,
) (int32, error) {

	if *fuseEGraphFlag && !fusedKernelsQualified {
		return 0, fmt.Errorf("-fuse-egraph: fused kernels are not qualified on a device; run without it")
	}
	if *fuseDecodeFlag && !fusedKernelsQualified {
		return 0, fmt.Errorf("-fuse-decode: fused kernels are not qualified on a device; run without it")
	}
	if err := gpuW.checkDTypes(); err != nil {
		return 0, err
	}
	if err := checkToken(tokenID, cfg.VocabSize); err != nil {
		return 0, err
	}
	if err := checkPos(pos, cfg.MaxSeqLen); err != nil {
		return 0, err
	}
	seqLen := pos + 1
	attnShared, err := cudaattn.DecodeSharedMemoryBytes(seqLen, attnThreads)
	if err != nil {
		return 0, fmt.Errorf("attention shared memory for %d tokens: %w", seqLen, err)
	}

	// Copy input token ID (4 bytes) to device
	sp.HostTokenIn[0] = tokenID
	if stream != nil {
		if err := sp.TokenIn.CopyFromHostAsync(sp.HostTokenIn, stream); err != nil {
			return 0, err
		}
	} else {
		if err := sp.TokenIn.CopyFromHost(sp.HostTokenIn); err != nil {
			return 0, err
		}
	}

	// The attention kernel's dynamic shared memory grows with the position.
	// Check it against the device limit and raise the kernel attribute once per
	// step; an oversized request fails here instead of at launch.
	attnPrep := cudalaunch.Kernel(attnFn).SharedMem(uint32(attnShared))
	if stream != nil {
		attnPrep = attnPrep.On(stream)
	}
	if err := attnPrep.PrepareSharedMem(sp.X.Device()); err != nil {
		return 0, fmt.Errorf("attention launch at position %d: %w", pos, err)
	}

	if err := emitForwardStep(gpuW, cfg, kv.Layer, sp, gm, attnFn, sampleFn, sp.TokenIn, pos, uint32(attnShared), func(l *cudalaunch.Launcher, args ...any) error {
		if stream != nil {
			l = l.On(stream)
		}
		return l.Launch(args...)
	}); err != nil {
		return 0, err
	}

	if stream != nil {
		if err := sp.TokenOut.CopyToHostAsync(sp.HostTokenOut, stream); err != nil {
			return 0, err
		}
		return 0, nil
	}

	// Synchronize GPU stream and read back only the 1 single token ID
	if err := driver.CtxSynchronize(); err != nil {
		return 0, err
	}

	if err := sp.TokenOut.CopyToHost(sp.HostTokenOut); err != nil {
		return 0, err
	}

	return sp.HostTokenOut[0], nil
}

// emitForwardStep emits the per-token decoder kernel sequence (15L+4 unfused
// or 11L+4 fused) in launch order, reading the input token ID from tokenSrc
// and passing each configured *cudalaunch.Launcher and its arguments to emit.
// Callers must prepare dynamic shared memory on attnFn for at least attnShared
// bytes before executing the emitted launches.
func emitForwardStep(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	layerKV func(int) (cudakv.LayerCache, error),
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenSrc cudamem.Slice[int32],
	pos int,
	attnShared uint32,
	emit func(l *cudalaunch.Launcher, args ...any) error,
) error {
	if *fuseEGraphFlag && !fusedKernelsQualified {
		return fmt.Errorf("-fuse-egraph: fused kernels are not qualified on a device; run without it")
	}
	if *fuseDecodeFlag && !fusedKernelsQualified {
		return fmt.Errorf("-fuse-decode: fused kernels are not qualified on a device; run without it")
	}
	if err := gpuW.checkDTypes(); err != nil {
		return err
	}
	if err := checkPos(pos, cfg.MaxSeqLen); err != nil {
		return err
	}
	gemv, embed := gm.gemvFn, gm.embedFn
	fusedQKV, fusedSwiGLU := gm.fusedQKVFn, gm.fusedSwiGLUFn
	fusedQKVRoPEStore, fusedOutProjRes := gm.fusedQKVRoPEStoreFn, gm.fusedOutProjResFn
	fusedGateUpSwiGLU, fusedDownProjRes := gm.fusedGateUpSwiGLUFn, gm.fusedDownProjResFn
	if gpuW.DType == weightBF16 {
		gemv, embed = gm.gemvBF16Fn, gm.embedBF16Fn
		fusedQKV, fusedSwiGLU = gm.fusedQKVBF16Fn, gm.fusedSwiGLUBF16Fn
		fusedQKVRoPEStore, fusedOutProjRes = gm.fusedQKVRoPEStoreBF16Fn, gm.fusedOutProjResBF16Fn
		fusedGateUpSwiGLU, fusedDownProjRes = gm.fusedGateUpSwiGLUBF16Fn, gm.fusedDownProjResBF16Fn
	}
	headDim := cfg.HeadDim()
	qHeads := cfg.NumAttentionHeads
	kvHeads := cfg.NumKeyValueHeads
	hiddenSize := cfg.HiddenSize
	interSize := cfg.IntermediateSize
	vocabSize := cfg.VocabSize
	ropeFactor, ropeLow, ropeHigh := float32(0), float32(0), float32(0)
	ropeOrigMax := int32(0)
	if cfg.RoPEScaling.hasLlama3Params() {
		rs := cfg.RoPEScaling
		ropeFactor, ropeLow, ropeHigh = rs.Factor, rs.LowFreqFactor, rs.HighFreqFactor
		ropeOrigMax = int32(rs.OriginalMaxPositionEmbeddings)
	}

	ptrOrNull := func(s cudamem.Slice[float32]) driver.DevicePtr {
		if s.Len() == 0 {
			return driver.DevicePtr(0)
		}
		return s.Ptr()
	}
	nullPtr := driver.DevicePtr(0)
	launcher := cudalaunch.Kernel

	// 1. Embedding lookup
	if err := emit(launcher(embed).Grid(1, 1, 1).Block(256, 1, 1),
		gpuW.EmbedTokens, tokenSrc, sp.X, int32(hiddenSize)); err != nil {
		return err
	}

	// 2. Transformer Decoder Layers
	attnCfg := cudaattn.DefaultConfig(qHeads, kvHeads, headDim)
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerW := &gpuW.Layers[i]
		lkv, err := layerKV(i)
		if err != nil {
			return err
		}

		if *fuseDecodeFlag && fusedQKVRoPEStore != 0 && fusedOutProjRes != 0 && fusedGateUpSwiGLU != 0 && fusedDownProjRes != 0 {
			if err := emit(launcher(fusedQKVRoPEStore).Grid(uint32((qHeads+kvHeads)*(headDim/2)), 1, 1).Block(128, 1, 1),
				sp.X, layerW.InputNorm, layerW.WQ, layerW.WK, layerW.WV,
				ptrOrNull(layerW.QBias), ptrOrNull(layerW.KBias), ptrOrNull(layerW.VBias),
				sp.Q, lkv.K, lkv.V,
				int32(hiddenSize), int32(qHeads), int32(kvHeads), int32(headDim), int32(pos),
				cfg.RmsNormEps, cfg.RopeTheta, ropeFactor, ropeLow, ropeHigh, ropeOrigMax); err != nil {
				return err
			}
			if err := emit(launcher(attnFn).Grid(uint32(qHeads), 1, 1).Block(attnThreads, 1, 1).SharedMem(attnShared),
				sp.Q, lkv.K, lkv.V, sp.AttnOut, int32(pos), int32(cfg.MaxSeqLen), int32(qHeads), int32(kvHeads), int32(headDim), attnCfg.Scale); err != nil {
				return err
			}
			if err := emit(launcher(fusedOutProjRes).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1),
				sp.AttnOut, sp.X, layerW.WO, sp.X, int32(qHeads*headDim), int32(hiddenSize)); err != nil {
				return err
			}
			if err := emit(launcher(fusedGateUpSwiGLU).Grid(uint32(interSize), 1, 1).Block(128, 1, 1),
				sp.X, layerW.PostNorm, layerW.WGate, layerW.WUp, sp.FFNAct, int32(hiddenSize), int32(interSize), cfg.RmsNormEps); err != nil {
				return err
			}
			if err := emit(launcher(fusedDownProjRes).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1),
				sp.FFNAct, sp.X, layerW.WDown, sp.X, int32(interSize), int32(hiddenSize)); err != nil {
				return err
			}
			continue
		}

		// Step 2a & 2b: Input RMSNorm + QKV Linear Projection
		if *fuseEGraphFlag && fusedQKV != 0 {
			// Fused RMSNorm + QKV projection: one block per Q, K and V row.
			if err := emit(launcher(fusedQKV).Grid(uint32(qHeads*headDim+2*kvHeads*headDim), 1, 1).Block(128, 1, 1),
				sp.X, layerW.InputNorm, layerW.WQ, layerW.WK, layerW.WV,
				ptrOrNull(layerW.QBias), ptrOrNull(layerW.KBias), ptrOrNull(layerW.VBias),
				sp.Q, sp.K, sp.V,
				int32(hiddenSize), int32(qHeads*headDim), int32(kvHeads*headDim), cfg.RmsNormEps); err != nil {
				return err
			}
		} else {
			// Unfused fallback path
			if err := emit(launcher(gm.normFn).Grid(1, 1, 1).Block(256, 1, 1),
				sp.X, layerW.InputNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
				return err
			}
			if err := emit(launcher(gemv).Grid(uint32(qHeads*headDim), 1, 1).Block(128, 1, 1),
				sp.NormX, layerW.WQ, ptrOrNull(layerW.QBias), sp.Q, int32(hiddenSize), int32(qHeads*headDim)); err != nil {
				return err
			}
			if err := emit(launcher(gemv).Grid(uint32(kvHeads*headDim), 1, 1).Block(128, 1, 1),
				sp.NormX, layerW.WK, ptrOrNull(layerW.KBias), sp.K, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
				return err
			}
			if err := emit(launcher(gemv).Grid(uint32(kvHeads*headDim), 1, 1).Block(128, 1, 1),
				sp.NormX, layerW.WV, ptrOrNull(layerW.VBias), sp.V, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
				return err
			}
		}

		// Step 2c: RoPE
		if cfg.RoPEScaling.hasLlama3Params() && gm.ropeScaledFn != 0 {
			rs := cfg.RoPEScaling
			if err := emit(cudalaunch.Kernel(gm.ropeScaledFn).Grid(uint32(qHeads), 1, 1).Block(uint32(headDim/2), 1, 1),
				sp.Q, sp.K, int32(pos), int32(qHeads), int32(kvHeads), int32(headDim),
				cfg.RopeTheta, rs.Factor, rs.LowFreqFactor, rs.HighFreqFactor, int32(rs.OriginalMaxPositionEmbeddings)); err != nil {
				return err
			}
		} else {
			if err := emit(launcher(gm.ropeFn).Grid(uint32(qHeads), 1, 1).Block(uint32(headDim/2), 1, 1),
				sp.Q, sp.K, int32(pos), int32(qHeads), int32(kvHeads), int32(headDim), cfg.RopeTheta); err != nil {
				return err
			}
		}

		// Step 2d: Store K and V in KV Cache
		if err := emit(launcher(gm.storeKVFn).Grid(1, 1, 1).Block(128, 1, 1),
			sp.K, sp.V, lkv.K, lkv.V, int32(pos), int32(kvHeads), int32(headDim)); err != nil {
			return err
		}

		// Step 2e: Causal FlashAttention Decode
		if err := emit(launcher(attnFn).Grid(uint32(qHeads), 1, 1).Block(attnThreads, 1, 1).SharedMem(attnShared),
			sp.Q, lkv.K, lkv.V, sp.AttnOut, int32(pos), int32(cfg.MaxSeqLen), int32(qHeads), int32(kvHeads), int32(headDim), attnCfg.Scale); err != nil {
			return err
		}

		// Step 2f: Output GEMV
		if err := emit(launcher(gemv).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1),
			sp.AttnOut, layerW.WO, nullPtr, sp.NormX, int32(qHeads*headDim), int32(hiddenSize)); err != nil {
			return err
		}

		// Step 2g: Residual 1
		if err := emit(launcher(gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1),
			sp.X, sp.NormX, int32(hiddenSize)); err != nil {
			return err
		}

		// Step 2h: FFN Pre-RMSNorm
		if err := emit(launcher(gm.normFn).Grid(1, 1, 1).Block(256, 1, 1),
			sp.X, layerW.PostNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
			return err
		}

		// Step 2i: Gate & Up GEMVs
		if err := emit(launcher(gemv).Grid(uint32(interSize), 1, 1).Block(128, 1, 1),
			sp.NormX, layerW.WGate, nullPtr, sp.Gate, int32(hiddenSize), int32(interSize)); err != nil {
			return err
		}
		if err := emit(launcher(gemv).Grid(uint32(interSize), 1, 1).Block(128, 1, 1),
			sp.NormX, layerW.WUp, nullPtr, sp.Up, int32(hiddenSize), int32(interSize)); err != nil {
			return err
		}

		// Step 2j & 2k: Fused SwiGLU + Down GEMV
		if *fuseEGraphFlag && fusedSwiGLU != 0 {
			// Fused SwiGLU Down Projection (Single kernel launch)
			if err := emit(launcher(fusedSwiGLU).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1),
				sp.Gate, sp.Up, layerW.WDown, sp.FFNOut, int32(hiddenSize), int32(interSize)); err != nil {
				return err
			}
		} else {
			// Step 2j: SwiGLU: ffnAct = silu(gate) * up
			if err := emit(launcher(gm.swigluFn).Grid(1, 1, 1).Block(256, 1, 1),
				sp.Gate, sp.Up, sp.FFNAct, int32(interSize)); err != nil {
				return err
			}

			// Step 2k: Down GEMV
			if err := emit(launcher(gemv).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1),
				sp.FFNAct, layerW.WDown, nullPtr, sp.FFNOut, int32(interSize), int32(hiddenSize)); err != nil {
				return err
			}
		}

		// Step 2l: Residual 2
		if err := emit(launcher(gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1),
			sp.X, sp.FFNOut, int32(hiddenSize)); err != nil {
			return err
		}
	}

	// 3. Final RMSNorm
	if err := emit(launcher(gm.normFn).Grid(1, 1, 1).Block(256, 1, 1),
		sp.X, gpuW.Norm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
		return err
	}

	// 4. LM Head Logits GEMV
	if err := emit(launcher(gemv).Grid(uint32(vocabSize), 1, 1).Block(128, 1, 1),
		sp.NormX, gpuW.LMHead, nullPtr, sp.Logits, int32(hiddenSize), int32(vocabSize)); err != nil {
		return err
	}

	// 5. In-GPU Argmax Sampling
	return emit(launcher(sampleFn).Grid(1, 1, 1).Block(256, 1, 1),
		sp.Logits, sp.TokenOut, int32(vocabSize))
}
