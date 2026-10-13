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
func NewGPUScratchpad(dev *cuda.Device, cfg *ModelConfig) (res *GPUScratchpad, err error) {
	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim

	sp := &GPUScratchpad{
		HostTokenIn:  make([]int32, 1),
		HostTokenOut: make([]int32, 1),
	}
	defer func() {
		if err != nil {
			if ferr := sp.Free(); ferr != nil {
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
		if *b.dst, err = cudamem.Alloc[float32](dev, b.n); err != nil {
			return nil, fmt.Errorf("alloc scratch: %w", err)
		}
	}
	if sp.TokenIn, err = cudamem.Alloc[int32](dev, 1); err != nil {
		return nil, fmt.Errorf("alloc token in: %w", err)
	}
	if sp.TokenOut, err = cudamem.Alloc[int32](dev, 1); err != nil {
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

// Free releases all GPU weight memory, each allocation once: a tied LMHead
// is a borrowed view and never frees the embedding table. A weight whose
// free fails is kept, so that Free can be called again; the errors are
// joined and keep their identity for errors.Is. After a successful Free,
// calling it again does nothing.
func (w *GPUModelWeights) Free() error {
	var errs []error
	for _, m := range w.matrices() {
		errs = append(errs, m.release())
	}
	for _, v := range w.vectors() {
		errs = append(errs, freeOwned(v))
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
func uploadModel(dev *cuda.Device, w *ModelWeights, cfg *ModelConfig, dtype weightDType) (res *GPUModelWeights, err error) {
	if len(w.Layers) != cfg.NumHiddenLayers {
		return nil, fmt.Errorf("model has %d layers, config says %d", len(w.Layers), cfg.NumHiddenLayers)
	}
	gw := &GPUModelWeights{DType: dtype, Layers: make([]GPULayerWeights, cfg.NumHiddenLayers)}
	defer func() {
		if err != nil {
			if ferr := gw.Free(); ferr != nil {
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
		if *mats[i], err = uploadWeight(dev, host, dtype); err != nil {
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
		if *vecs[i], err = cudamem.Alloc[float32](dev, len(host)); err != nil {
			return nil, fmt.Errorf("upload vector weight %d: %w", i, err)
		}
		if err = vecs[i].CopyFromHost(host); err != nil {
			return nil, fmt.Errorf("upload vector weight %d: %w", i, err)
		}
	}
	return gw, nil
}

// fusedKernelsQualified reports whether FusedRMSNormQKV and FusedSwiGLUDown
// have passed fused_parity_cuda_test.go on a device. Until then -fuse-egraph
// is refused rather than silently running unqualified kernels.
var fusedKernelsQualified = false

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
	// before returning ownership to the caller.
	if waitErr := stream.Synchronize(context.Background()); err == nil {
		err = waitErr
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
	if err := gpuW.checkDTypes(); err != nil {
		return 0, err
	}
	if err := checkToken(tokenID, cfg.VocabSize); err != nil {
		return 0, err
	}
	if err := checkPos(pos, cfg.MaxSeqLen); err != nil {
		return 0, err
	}
	gemv, embed := gm.gemvFn, gm.embedFn
	if gpuW.DType == weightBF16 {
		if *fuseEGraphFlag {
			return 0, fmt.Errorf("-fuse-egraph: fused kernels take float32 weights, not %s", gpuW.DType)
		}
		gemv, embed = gm.gemvBF16Fn, gm.embedBF16Fn
	}
	headDim := cfg.HeadDim()
	qHeads := cfg.NumAttentionHeads
	kvHeads := cfg.NumKeyValueHeads
	hiddenSize := cfg.HiddenSize
	interSize := cfg.IntermediateSize
	vocabSize := cfg.VocabSize
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

	ptrOrNull := func(s cudamem.Slice[float32]) driver.DevicePtr {
		if s.Len() == 0 {
			return driver.DevicePtr(0)
		}
		return s.Ptr()
	}
	nullPtr := driver.DevicePtr(0)

	launcher := func(fn driver.Function) *cudalaunch.Launcher {
		l := cudalaunch.Kernel(fn)
		if stream != nil {
			l = l.On(stream)
		}
		return l
	}

	// 1. Embedding lookup
	if err := launcher(embed).Grid(1, 1, 1).Block(256, 1, 1).
		Launch(gpuW.EmbedTokens, sp.TokenIn, sp.X, int32(hiddenSize)); err != nil {
		return 0, err
	}

	// The attention kernel's dynamic shared memory grows with the position.
	// Check it against the device limit and raise the kernel attribute once per
	// step; an oversized request fails here instead of at launch.
	if err := launcher(attnFn).SharedMem(uint32(attnShared)).PrepareSharedMem(sp.X.Device()); err != nil {
		return 0, fmt.Errorf("attention launch at position %d: %w", pos, err)
	}

	// 2. Transformer Decoder Layers
	for i := 0; i < cfg.NumHiddenLayers; i++ {
		layerW := &gpuW.Layers[i]
		layerKV, err := kv.Layer(i)
		if err != nil {
			return 0, err
		}

		// Step 2a & 2b: Input RMSNorm + QKV Linear Projection
		if *fuseEGraphFlag && gm.fusedQKVFn != 0 {
			// Fused RMSNorm + QKV projection: one block per Q, K and V row.
			if err := launcher(gm.fusedQKVFn).Grid(uint32(qHeads*headDim+2*kvHeads*headDim), 1, 1).Block(128, 1, 1).
				Launch(sp.X, layerW.InputNorm, layerW.WQ, layerW.WK, layerW.WV,
					ptrOrNull(layerW.QBias), ptrOrNull(layerW.KBias), ptrOrNull(layerW.VBias),
					sp.Q, sp.K, sp.V,
					int32(hiddenSize), int32(qHeads*headDim), int32(kvHeads*headDim), cfg.RmsNormEps); err != nil {
				return 0, err
			}
		} else {
			// Unfused fallback path
			if err := launcher(gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
				Launch(sp.X, layerW.InputNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
				return 0, err
			}
			if err := launcher(gemv).Grid(uint32(qHeads*headDim), 1, 1).Block(128, 1, 1).
				Launch(sp.NormX, layerW.WQ, ptrOrNull(layerW.QBias), sp.Q, int32(hiddenSize), int32(qHeads*headDim)); err != nil {
				return 0, err
			}
			if err := launcher(gemv).Grid(uint32(kvHeads*headDim), 1, 1).Block(128, 1, 1).
				Launch(sp.NormX, layerW.WK, ptrOrNull(layerW.KBias), sp.K, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
				return 0, err
			}
			if err := launcher(gemv).Grid(uint32(kvHeads*headDim), 1, 1).Block(128, 1, 1).
				Launch(sp.NormX, layerW.WV, ptrOrNull(layerW.VBias), sp.V, int32(hiddenSize), int32(kvHeads*headDim)); err != nil {
				return 0, err
			}
		}

		// Step 2c: RoPE
		if err := launcher(gm.ropeFn).Grid(uint32(qHeads), 1, 1).Block(uint32(headDim/2), 1, 1).
			Launch(sp.Q, sp.K, int32(pos), int32(qHeads), int32(kvHeads), int32(headDim), cfg.RopeTheta); err != nil {
			return 0, err
		}

		// Step 2d: Store K and V in KV Cache
		if err := launcher(gm.storeKVFn).Grid(1, 1, 1).Block(128, 1, 1).
			Launch(sp.K, sp.V, layerKV.K, layerKV.V, int32(pos), int32(kvHeads), int32(headDim)); err != nil {
			return 0, err
		}

		// Step 2e: Causal FlashAttention Decode
		attnCfg := cudaattn.DefaultConfig(qHeads, kvHeads, headDim)
		if err := launcher(attnFn).Grid(uint32(qHeads), 1, 1).Block(attnThreads, 1, 1).SharedMem(uint32(attnShared)).
			Launch(sp.Q, layerKV.K, layerKV.V, sp.AttnOut, int32(pos), int32(cfg.MaxSeqLen), int32(qHeads), int32(kvHeads), int32(headDim), attnCfg.Scale); err != nil {
			return 0, err
		}

		// Step 2f: Output GEMV
		if err := launcher(gemv).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1).
			Launch(sp.AttnOut, layerW.WO, nullPtr, sp.NormX, int32(qHeads*headDim), int32(hiddenSize)); err != nil {
			return 0, err
		}

		// Step 2g: Residual 1
		if err := launcher(gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1).
			Launch(sp.X, sp.NormX, int32(hiddenSize)); err != nil {
			return 0, err
		}

		// Step 2h: FFN Pre-RMSNorm
		if err := launcher(gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
			Launch(sp.X, layerW.PostNorm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
			return 0, err
		}

		// Step 2i: Gate & Up GEMVs
		if err := launcher(gemv).Grid(uint32(interSize), 1, 1).Block(128, 1, 1).
			Launch(sp.NormX, layerW.WGate, nullPtr, sp.Gate, int32(hiddenSize), int32(interSize)); err != nil {
			return 0, err
		}
		if err := launcher(gemv).Grid(uint32(interSize), 1, 1).Block(128, 1, 1).
			Launch(sp.NormX, layerW.WUp, nullPtr, sp.Up, int32(hiddenSize), int32(interSize)); err != nil {
			return 0, err
		}

		// Step 2j & 2k: Fused SwiGLU + Down GEMV
		if *fuseEGraphFlag && gm.fusedSwiGLUFn != 0 {
			// Fused SwiGLU Down Projection (Single kernel launch)
			if err := launcher(gm.fusedSwiGLUFn).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1).
				Launch(sp.Gate, sp.Up, layerW.WDown, sp.FFNOut, int32(hiddenSize), int32(interSize)); err != nil {
				return 0, err
			}
		} else {
			// Step 2j: SwiGLU: ffnAct = silu(gate) * up
			if err := launcher(gm.swigluFn).Grid(1, 1, 1).Block(256, 1, 1).
				Launch(sp.Gate, sp.Up, sp.FFNAct, int32(interSize)); err != nil {
				return 0, err
			}

			// Step 2k: Down GEMV
			if err := launcher(gemv).Grid(uint32(hiddenSize), 1, 1).Block(128, 1, 1).
				Launch(sp.FFNAct, layerW.WDown, nullPtr, sp.FFNOut, int32(interSize), int32(hiddenSize)); err != nil {
				return 0, err
			}
		}

		// Step 2l: Residual 2
		if err := launcher(gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1).
			Launch(sp.X, sp.FFNOut, int32(hiddenSize)); err != nil {
			return 0, err
		}
	}

	// 3. Final RMSNorm
	if err := launcher(gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
		Launch(sp.X, gpuW.Norm, sp.NormX, int32(hiddenSize), cfg.RmsNormEps); err != nil {
		return 0, err
	}

	// 4. LM Head Logits GEMV
	if err := launcher(gemv).Grid(uint32(vocabSize), 1, 1).Block(128, 1, 1).
		Launch(sp.NormX, gpuW.LMHead, nullPtr, sp.Logits, int32(hiddenSize), int32(vocabSize)); err != nil {
		return 0, err
	}

	// 5. In-GPU Argmax Sampling
	if err := launcher(sampleFn).Grid(1, 1, 1).Block(256, 1, 1).
		Launch(sp.Logits, sp.TokenOut, int32(vocabSize)); err != nil {
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
