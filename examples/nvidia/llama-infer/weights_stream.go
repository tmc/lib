package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// A checkpoint reads a model's tensors by name. *safetensors.Sharded and
// *safetensors.File satisfy it.
type checkpoint interface {
	Info(name string) (safetensors.TensorInfo, bool)
	TensorReader(name string) (*io.SectionReader, error)
	Close() error
}

// openCheckpoint opens the checkpoint in dir for streamModel, and reads its
// configuration and whether an absent lm_head is tied to the embedding
// table. With no dir it returns the deterministic testing model.
func openCheckpoint(dir string) (checkpoint, *ModelConfig, bool, error) {
	if dir == "" {
		cfg := DefaultModelConfig()
		f, err := GenerateDeterministicWeights(&cfg).checkpoint(&cfg, safetensors.F32)
		if err != nil {
			return nil, nil, false, fmt.Errorf("testing model: %w", err)
		}
		return f, &cfg, false, nil
	}
	cfg, tied, err := loadModelConfig(dir)
	if err != nil {
		return nil, nil, false, fmt.Errorf("load model %s: %w", dir, err)
	}
	ck, err := safetensors.OpenSharded(dir)
	if err != nil {
		return nil, nil, false, fmt.Errorf("load model %s: open safetensors: %w", dir, err)
	}
	return ck, cfg, tied, nil
}

// streamChunk is the most checkpoint bytes streamModel holds on the host at
// once: Llama-3-8B's 1 GB embedding table goes to the GPU in 16 pieces.
const streamChunk = 64 << 20

// streamModel uploads the model in ck, whose configuration is cfg, to dev,
// with its matrices stored as dtype and its norms and biases as float32. It
// reads each tensor through one host buffer of streamChunk bytes and
// converts it a piece at a time, so the host never holds the model. Each
// layer's q, k and v projections go into one buffer, WQKV, and its gate and
// up projections into another, WGateUp, which WQ, WK, WV, WGate and WUp
// view. An absent lm_head borrows the embedding table if tied is set and is
// an error otherwise.
//
// The device bytes are those uploadModel stores for the weights LoadModel
// reads from ck. On error streamModel frees whatever it uploaded; if that
// free fails too, it returns the weights it could not free, with both
// errors, so the caller can retry Free.
func streamModel(dev *cuda.Device, ck checkpoint, cfg *ModelConfig, tied bool, dtype weightDType) (*GPUModelWeights, error) {
	return streamModelWith(ck, cfg, tied, dtype, make([]byte, streamChunk),
		func(n int, dtype weightDType) (deviceWeight, error) { return allocWeight(dev, n, dtype) },
		func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) },
		writeDevice,
		freeOwned[cudamem.Slice[float32]])
}

// allocWeight allocates a matrix weight of n elements of dtype on dev. As
// with uploadWeight, the weight returned with an error holds any memory
// allocated, for the caller to release.
func allocWeight(dev *cuda.Device, n int, dtype weightDType) (deviceWeight, error) {
	switch dtype {
	case weightF32:
		return allocAs(dtype, n, func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) })
	case weightBF16:
		return allocAs(dtype, n, func(n int) (cudamem.Slice[uint16], error) { return cudamem.Alloc[uint16](dev, n) })
	}
	return deviceWeight{}, fmt.Errorf("unknown weight dtype %q", dtype)
}

// allocAs allocates n elements with alloc. Like uploadAs, it returns any
// nonzero allocation, whatever the error.
func allocAs[S deviceBuffer](dtype weightDType, n int, alloc func(n int) (S, error)) (deviceWeight, error) {
	s, err := alloc(n)
	if err != nil && s.Ptr() == 0 {
		return deviceWeight{}, err
	}
	return deviceWeight{buf: s, dtype: dtype}, err
}

// writeDevice copies b to dst, a cudamem.Slice of uint16 or float32,
// starting off bytes in.
func writeDevice(dst deviceBuffer, off int, b []byte) error {
	var s cudamem.Slice[byte]
	var err error
	switch d := dst.(type) {
	case cudamem.Slice[uint16]:
		s, err = cudamem.Reinterpret[byte](d)
	case cudamem.Slice[float32]:
		s, err = cudamem.Reinterpret[byte](d)
	default:
		return fmt.Errorf("write to %T", dst)
	}
	if err != nil {
		return err
	}
	if off < 0 || off+len(b) > s.Len() {
		return fmt.Errorf("write of %d bytes at %d overflows %d-byte buffer", len(b), off, s.Len())
	}
	return s.Slice(off, off+len(b)).CopyFromHost(b)
}

// streamModelWith is streamModel with its host buffer and device operations
// as arguments: allocMatrix allocates a matrix of n elements of dtype,
// allocVector a float32 vector of n elements, write copies host bytes into
// a matrix or vector at a byte offset, and free frees a vector, which it
// must zero only when the free succeeds, as freeOwned does.
func streamModelWith(ck checkpoint, cfg *ModelConfig, tied bool, dtype weightDType, buf []byte,
	allocMatrix func(n int, dtype weightDType) (deviceWeight, error),
	allocVector func(n int) (cudamem.Slice[float32], error),
	write func(dst deviceBuffer, off int, b []byte) error,
	free func(*cudamem.Slice[float32]) error) (res *GPUModelWeights, err error) {
	esize := map[weightDType]int{weightF32: 4, weightBF16: 2}[dtype]
	if esize == 0 {
		return nil, fmt.Errorf("unknown weight dtype %q", dtype)
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

	// info describes the named tensor, which must hold n elements.
	info := func(name string, n int) (safetensors.TensorInfo, error) {
		t, ok := ck.Info(name)
		if !ok {
			return t, fmt.Errorf("missing tensor %s", name)
		}
		m := 1
		for _, d := range t.Shape {
			m *= d
		}
		if m != n {
			return t, fmt.Errorf("tensor %s has %d elements, want %d", name, m, n)
		}
		return t, nil
	}
	// stream copies the named tensor into dst, starting off bytes in, as
	// dtype to.
	stream := func(dst deviceBuffer, off int, t safetensors.TensorInfo, to weightDType) error {
		r, err := ck.TensorReader(t.Name)
		if err != nil {
			return err
		}
		err = streamTensor(r, t.Dtype, to, buf, func(o int, b []byte) error { return write(dst, off+o, b) })
		if err != nil {
			return fmt.Errorf("upload %s: %w", t.Name, err)
		}
		return nil
	}

	// A part is a tensor of n elements that fills part of a matrix, and
	// the view of that part to set, if any.
	type part struct {
		name string
		n    int
		view *deviceWeight
	}
	// matrix uploads parts, one after another, into one new matrix, *dst.
	matrix := func(dst *deviceWeight, parts ...part) error {
		var ts []safetensors.TensorInfo
		total := 0
		for _, p := range parts {
			t, err := info(p.name, p.n)
			if err != nil {
				return err
			}
			ts = append(ts, t)
			total += p.n
		}
		var err error
		if *dst, err = allocMatrix(total, dtype); err != nil {
			return fmt.Errorf("alloc %s: %w", parts[0].name, err)
		}
		off := 0
		for i, p := range parts {
			if err := stream(dst.buf, off*esize, ts[i], dtype); err != nil {
				return err
			}
			if p.view != nil {
				if *p.view, err = dst.view(off, off+p.n); err != nil {
					return fmt.Errorf("view %s: %w", p.name, err)
				}
			}
			off += p.n
		}
		return nil
	}
	// vector uploads the named tensor into a new float32 vector, *dst. An
	// absent optional tensor leaves *dst zero.
	vector := func(dst *cudamem.Slice[float32], name string, n int, optional bool) error {
		if _, ok := ck.Info(name); !ok && optional {
			return nil
		}
		t, err := info(name, n)
		if err != nil {
			return err
		}
		if *dst, err = allocVector(n); err != nil {
			return fmt.Errorf("alloc %s: %w", name, err)
		}
		return stream(*dst, 0, t, weightF32)
	}

	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim
	hidden, inter, vocab := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	// Upload in checkpoint order, which reads each shard front to back.
	if err := matrix(&gw.EmbedTokens, part{"model.embed_tokens.weight", vocab * hidden, nil}); err != nil {
		return nil, err
	}
	for i := range gw.Layers {
		l := &gw.Layers[i]
		pre := fmt.Sprintf("model.layers.%d.", i)
		for _, step := range []func() error{
			func() error { return vector(&l.InputNorm, pre+"input_layernorm.weight", hidden, false) },
			func() error {
				return matrix(&l.WQKV,
					part{pre + "self_attn.q_proj.weight", qDim * hidden, &l.WQ},
					part{pre + "self_attn.k_proj.weight", kvDim * hidden, &l.WK},
					part{pre + "self_attn.v_proj.weight", kvDim * hidden, &l.WV})
			},
			func() error { return vector(&l.QBias, pre+"self_attn.q_proj.bias", qDim, true) },
			func() error { return vector(&l.KBias, pre+"self_attn.k_proj.bias", kvDim, true) },
			func() error { return vector(&l.VBias, pre+"self_attn.v_proj.bias", kvDim, true) },
			func() error { return matrix(&l.WO, part{pre + "self_attn.o_proj.weight", hidden * qDim, nil}) },
			func() error { return vector(&l.PostNorm, pre+"post_attention_layernorm.weight", hidden, false) },
			func() error {
				return matrix(&l.WGateUp,
					part{pre + "mlp.gate_proj.weight", inter * hidden, &l.WGate},
					part{pre + "mlp.up_proj.weight", inter * hidden, &l.WUp})
			},
			func() error { return matrix(&l.WDown, part{pre + "mlp.down_proj.weight", hidden * inter, nil}) },
		} {
			if err := step(); err != nil {
				return nil, err
			}
		}
	}
	if err := vector(&gw.Norm, "model.norm.weight", hidden, false); err != nil {
		return nil, err
	}
	if _, ok := ck.Info("lm_head.weight"); !ok {
		if !tied {
			return nil, fmt.Errorf("missing tensor lm_head.weight and tie_word_embeddings is false")
		}
		gw.LMHead = gw.EmbedTokens.borrow()
		return gw, nil
	}
	if err := matrix(&gw.LMHead, part{"lm_head.weight", vocab * hidden, nil}); err != nil {
		return nil, err
	}
	return gw, nil
}

// streamTensor reads the elements of dtype from in r, converts them to the
// device dtype to with encodeAs, and hands the converted bytes to write a
// piece at a time, with each piece's byte offset in the converted tensor.
// It reads through buf, at most len(buf) bytes at a time.
func streamTensor(r io.Reader, from safetensors.Dtype, to weightDType, buf []byte, write func(off int, b []byte) error) error {
	esize := from.Size()
	if esize == 0 {
		return fmt.Errorf("unsupported dtype %q", from)
	}
	chunk := buf[:len(buf)/esize*esize]
	if len(chunk) == 0 {
		return fmt.Errorf("buffer of %d bytes holds no %s element", len(buf), from)
	}
	var conv []byte
	for off := 0; ; {
		n, err := io.ReadFull(r, chunk)
		if n > 0 {
			b, cerr := encodeAs(conv, chunk[:n], from, to)
			if cerr != nil {
				return cerr
			}
			if &b[0] != &chunk[0] {
				conv = b
			}
			if werr := write(off, b); werr != nil {
				return werr
			}
			off += len(b)
		}
		switch err {
		case nil:
		case io.EOF, io.ErrUnexpectedEOF:
			return nil
		default:
			return err
		}
	}
}

// encodeAs returns src, whole elements of the checkpoint dtype from, as the
// bytes of the device dtype to, reusing dst's storage when it can. BF16 to
// bf16 and F32 to f32 return src itself. Otherwise each element is widened
// to float32 as LoadModel widens it, and, for bf16, rounded as roundBF16
// rounds it, so a streamed weight has the bytes uploadWeight gives it.
func encodeAs(dst, src []byte, from safetensors.Dtype, to weightDType) ([]byte, error) {
	if from == safetensors.BF16 && to == weightBF16 || from == safetensors.F32 && to == weightF32 {
		return src, nil
	}
	f, err := decodeTensorToFloat32(src, string(from))
	if err != nil {
		return nil, err
	}
	switch to {
	case weightBF16:
		dst = grow(dst, 2*len(f))
		for i, v := range f {
			binary.LittleEndian.PutUint16(dst[2*i:], safetensors.Float32ToBFloat16(v))
		}
	case weightF32:
		dst = grow(dst, 4*len(f))
		for i, v := range f {
			binary.LittleEndian.PutUint32(dst[4*i:], math.Float32bits(v))
		}
	default:
		return nil, fmt.Errorf("unknown weight dtype %q", to)
	}
	return dst, nil
}

// grow returns b resliced to n bytes, reallocated if its capacity is less.
func grow(b []byte, n int) []byte {
	if cap(b) < n {
		return make([]byte, n)
	}
	return b[:n]
}
