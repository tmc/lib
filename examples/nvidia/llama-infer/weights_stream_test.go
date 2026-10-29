package main

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// memBuffer is a fake device matrix held in host memory. A view shares
// its bytes. Free logs the buffer's name.
type memBuffer struct {
	name  string
	b     []byte
	esize int
	log   *[]string
}

func (m *memBuffer) Ptr() driver.DevicePtr { return 1 }

func (m *memBuffer) Free() error {
	*m.log = append(*m.log, m.name)
	return nil
}

func (m *memBuffer) view(lo, hi int) deviceBuffer {
	return &memBuffer{name: m.name + "/view", b: m.b[lo*m.esize : hi*m.esize], esize: m.esize, log: m.log}
}

// memDevice provides the device operations streamModelWith takes, in host
// memory. Its vectors are fake cudamem.Slices, each with its own device
// pointer, whose bytes it keeps by pointer. The *Fail fields fail that
// call of the operation, counting from 1; 0 fails none.
type memDevice struct {
	allocFail, vecFail, writeFail int

	mats   []*memBuffer
	vecs   map[driver.DevicePtr][]byte
	writes int
	freed  []string // matrix frees
	vfreed map[driver.DevicePtr]int
	bad    []string
}

func newMemDevice() *memDevice {
	return &memDevice{vecs: make(map[driver.DevicePtr][]byte), vfreed: make(map[driver.DevicePtr]int)}
}

func (d *memDevice) allocMatrix(n int, dtype weightDType) (deviceWeight, error) {
	if len(d.mats)+1 == d.allocFail {
		d.mats = append(d.mats, nil)
		return deviceWeight{}, errFakeAlloc
	}
	esize := map[weightDType]int{weightF32: 4, weightBF16: 2}[dtype]
	m := &memBuffer{name: fmt.Sprintf("m%d", len(d.mats)), b: make([]byte, n*esize), esize: esize, log: &d.freed}
	d.mats = append(d.mats, m)
	return deviceWeight{buf: m, dtype: dtype}, nil
}

func (d *memDevice) allocVector(n int) (cudamem.Slice[float32], error) {
	if len(d.vecs)+1 == d.vecFail {
		return cudamem.Slice[float32]{}, errFakeAlloc
	}
	s, ok := fakeDeviceSlice[float32](driver.DevicePtr(len(d.vecs) + 1))
	if !ok {
		d.bad = append(d.bad, "cannot make a fake vector")
	}
	d.vecs[s.Ptr()] = make([]byte, 4*n)
	return s, nil
}

func (d *memDevice) write(dst deviceBuffer, off int, b []byte) error {
	d.writes++
	if d.writes == d.writeFail {
		return errFakeCopy
	}
	var mem []byte
	switch dst := dst.(type) {
	case *memBuffer:
		mem = dst.b
	case cudamem.Slice[float32]:
		mem = d.vecs[dst.Ptr()]
	}
	if off+len(b) > len(mem) {
		d.bad = append(d.bad, fmt.Sprintf("write of %d bytes at %d overflows %d", len(b), off, len(mem)))
		return nil
	}
	copy(mem[off:], b)
	return nil
}

func (d *memDevice) free(s *cudamem.Slice[float32]) error {
	if p := s.Ptr(); p != 0 {
		d.vfreed[p]++
	}
	*s = cudamem.Slice[float32]{}
	return nil
}

// streamFake runs streamModelWith on d. Its weights hold no device memory:
// free them with their free method and d.free.
func streamFake(d *memDevice, ck checkpoint, cfg *ModelConfig, tied bool, dtype weightDType, buf []byte) (*GPUModelWeights, error) {
	return streamModelWith(ck, cfg, tied, dtype, buf, d.allocMatrix, d.allocVector, d.write, d.free)
}

// bytesOf returns the bytes d holds for w, a matrix or its view.
func bytesOf(w deviceWeight) []byte {
	if m, ok := w.buf.(*memBuffer); ok {
		return m.b
	}
	return nil
}

// wantBytes returns the bytes uploadWeight stores, as dtype, for the
// tensors LoadModel decodes from the checkpoint bytes of v as ckType.
func wantBytes(t *testing.T, ckType safetensors.Dtype, dtype weightDType, vs ...[]float32) []byte {
	t.Helper()
	var out []byte
	for _, v := range vs {
		raw, err := safetensors.EncodeFloat32(ckType, v)
		if err != nil {
			t.Fatal(err)
		}
		host, err := decodeTensorToFloat32(raw, string(ckType))
		if err != nil {
			t.Fatal(err)
		}
		switch dtype {
		case weightBF16:
			for _, h := range roundBF16(host) {
				out = binary.LittleEndian.AppendUint16(out, h)
			}
		case weightF32:
			for _, f := range host {
				out = binary.LittleEndian.AppendUint32(out, math.Float32bits(f))
			}
		}
	}
	return out
}

// streamTestModel returns a two-layer testing model and its
// configuration, with q, k and v biases if bias is set and an lm_head
// tied to the embedding table if tied is set. No two of its tensors share
// a prefix, so a view of the wrong part of a fused weight shows.
func streamTestModel(bias, tied bool) (*ModelWeights, *ModelConfig) {
	cfg := DefaultModelConfig()
	w := GenerateDeterministicWeights(&cfg)
	for i := range w.Layers {
		for j, t := range layerTensors(&cfg, i, &w.Layers[i]) {
			for k := range *t.dst {
				(*t.dst)[k] += float32(16*i + j)
			}
		}
	}
	if tied {
		w.LMHead = w.EmbedTokens
	}
	if bias {
		kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
		for i := range w.Layers {
			l := &w.Layers[i]
			l.QBias = make([]float32, cfg.HiddenSize)
			l.KBias = make([]float32, kvDim)
			l.VBias = make([]float32, kvDim)
			for j := range l.QBias {
				l.QBias[j] = float32(j+i) / 3
			}
			for j := range l.KBias {
				l.KBias[j], l.VBias[j] = -float32(j)/7, float32(j)*1.001
			}
		}
	}
	return w, &cfg
}

func TestStreamModelWith(t *testing.T) {
	for _, tt := range []struct {
		name    string
		ckType  safetensors.Dtype
		dtype   weightDType
		bias    bool
		tied    bool
		bufSize int
	}{
		{name: "bf16 from BF16", ckType: safetensors.BF16, dtype: weightBF16, bufSize: streamChunk},
		{name: "bf16 from BF16, 6-byte pieces", ckType: safetensors.BF16, dtype: weightBF16, bufSize: 6},
		{name: "bf16 from BF16, 7-byte buffer", ckType: safetensors.BF16, dtype: weightBF16, bufSize: 7},
		{name: "bf16 from F32, tied, bias", ckType: safetensors.F32, dtype: weightBF16, bias: true, tied: true, bufSize: 4096},
		{name: "bf16 from F16, bias", ckType: safetensors.F16, dtype: weightBF16, bias: true, bufSize: 10},
		{name: "f32 from BF16", ckType: safetensors.BF16, dtype: weightF32, bufSize: 4096},
		{name: "f32 from F32, tied", ckType: safetensors.F32, dtype: weightF32, tied: true, bufSize: 12},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w, cfg := streamTestModel(tt.bias, tt.tied)
			ck, err := w.checkpoint(cfg, tt.ckType)
			if err != nil {
				t.Fatal(err)
			}
			d := newMemDevice()
			gw, err := streamFake(d, ck, cfg, tt.tied, tt.dtype, make([]byte, tt.bufSize))
			if err != nil {
				t.Fatal(err)
			}
			for _, b := range d.bad {
				t.Error(b)
			}
			if err := gw.checkDTypes(); err != nil {
				t.Error(err)
			}
			check := func(what string, got deviceWeight, want []byte) {
				t.Helper()
				if got.dtype != tt.dtype {
					t.Errorf("%s: dtype %s, want %s", what, got.dtype, tt.dtype)
				}
				if !bytes.Equal(bytesOf(got), want) {
					t.Errorf("%s: %d device bytes differ from the %d uploadWeight stores", what, len(bytesOf(got)), len(want))
				}
			}
			checkVec := func(what string, got cudamem.Slice[float32], host []float32) {
				t.Helper()
				if len(host) == 0 {
					if got.Ptr() != 0 {
						t.Errorf("%s: absent, but uploaded", what)
					}
					return
				}
				if want := wantBytes(t, tt.ckType, weightF32, host); !bytes.Equal(d.vecs[got.Ptr()], want) {
					t.Errorf("%s: device bytes differ from the float32 LoadModel reads", what)
				}
			}
			want := func(vs ...[]float32) []byte { return wantBytes(t, tt.ckType, tt.dtype, vs...) }
			check("embed_tokens", gw.EmbedTokens, want(w.EmbedTokens))
			check("lm_head", gw.LMHead, want(w.LMHead))
			if tied := gw.LMHead.borrowed && gw.LMHead.buf == gw.EmbedTokens.buf; tied != tt.tied {
				t.Errorf("lm_head borrows the embedding table: %v, want %v", tied, tt.tied)
			}
			checkVec("norm", gw.Norm, w.Norm)
			for i, l := range gw.Layers {
				h := &w.Layers[i]
				p := fmt.Sprintf("layer %d ", i)
				check(p+"qkv", l.WQKV, want(h.WQ, h.WK, h.WV))
				check(p+"gate_up", l.WGateUp, want(h.WGate, h.WUp))
				for _, m := range []struct {
					name string
					got  deviceWeight
					host []float32
				}{
					{"q", l.WQ, h.WQ}, {"k", l.WK, h.WK}, {"v", l.WV, h.WV},
					{"o", l.WO, h.WO}, {"gate", l.WGate, h.WGate}, {"up", l.WUp, h.WUp}, {"down", l.WDown, h.WDown},
				} {
					check(p+m.name, m.got, want(m.host))
				}
				for _, v := range []*deviceWeight{&l.WQ, &l.WK, &l.WV, &l.WGate, &l.WUp} {
					if !v.borrowed {
						t.Errorf("%sview %p owns its memory", p, v)
					}
				}
				checkVec(p+"input_layernorm", l.InputNorm, h.InputNorm)
				checkVec(p+"post_attention_layernorm", l.PostNorm, h.PostNorm)
				checkVec(p+"q bias", l.QBias, h.QBias)
				checkVec(p+"k bias", l.KBias, h.KBias)
				checkVec(p+"v bias", l.VBias, h.VBias)
			}
			// Each layer allocates qkv, o, gate_up and down; the model
			// adds embed_tokens and an untied lm_head.
			nmat := 2 + 4*cfg.NumHiddenLayers
			if tt.tied {
				nmat--
			}
			if len(d.mats) != nmat {
				t.Errorf("allocated %d matrices, want %d", len(d.mats), nmat)
			}
			if len(d.freed) != 0 || len(d.vfreed) != 0 {
				t.Errorf("freed %q and %d vectors on success", d.freed, len(d.vfreed))
			}
			if err := gw.free(d.free); err != nil {
				t.Fatal(err)
			}
			freedOnce(t, d)
		})
	}
}

// freedOnce reports a matrix or vector d allocated that was not freed
// exactly once.
func freedOnce(t *testing.T, d *memDevice) {
	t.Helper()
	for _, m := range d.mats {
		if m == nil {
			continue
		}
		if n := countOf(d.freed, m.name); n != 1 {
			t.Errorf("%s freed %d times, want 1", m.name, n)
		}
	}
	if len(d.freed) != len(slices.DeleteFunc(slices.Clone(d.mats), func(m *memBuffer) bool { return m == nil })) {
		t.Errorf("freed %q", d.freed)
	}
	for p := range d.vecs {
		if d.vfreed[p] != 1 {
			t.Errorf("vector %d freed %d times, want 1", p, d.vfreed[p])
		}
	}
}

func TestStreamModelWithErrors(t *testing.T) {
	for _, tt := range []struct {
		name    string
		edit    func(w *ModelWeights)
		tied    bool
		dtype   weightDType
		dev     memDevice
		want    error
		wantErr string
	}{
		{name: "missing k_proj", edit: func(w *ModelWeights) { w.Layers[1].WK = nil },
			wantErr: "missing tensor model.layers.1.self_attn.k_proj.weight"},
		{name: "short q_proj", edit: func(w *ModelWeights) { w.Layers[0].WQ = w.Layers[0].WQ[1:] },
			wantErr: "tensor model.layers.0.self_attn.q_proj.weight has 4095 elements, want 4096"},
		{name: "missing norm", edit: func(w *ModelWeights) { w.Layers[0].PostNorm = nil },
			wantErr: "missing tensor model.layers.0.post_attention_layernorm.weight"},
		{name: "untied without lm_head", edit: func(w *ModelWeights) { w.LMHead = w.EmbedTokens },
			wantErr: "missing tensor lm_head.weight and tie_word_embeddings is false"},
		{name: "unknown dtype", dtype: "f8", wantErr: `unknown weight dtype "f8"`},
		{name: "first matrix alloc fails", dev: memDevice{allocFail: 1}, want: errFakeAlloc},
		{name: "fourth matrix alloc fails", dev: memDevice{allocFail: 4}, want: errFakeAlloc},
		{name: "vector alloc fails", dev: memDevice{vecFail: 3}, want: errFakeAlloc},
		{name: "write fails", dev: memDevice{writeFail: 5}, want: errFakeCopy},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w, cfg := streamTestModel(false, false)
			if tt.edit != nil {
				tt.edit(w)
			}
			ck, err := w.checkpoint(cfg, safetensors.BF16)
			if err != nil {
				t.Fatal(err)
			}
			d := newMemDevice()
			d.allocFail, d.vecFail, d.writeFail = tt.dev.allocFail, tt.dev.vecFail, tt.dev.writeFail
			dtype := cmp.Or(tt.dtype, weightBF16)
			gw, err := streamFake(d, ck, cfg, tt.tied, dtype, make([]byte, 1000))
			if gw != nil {
				t.Errorf("weights returned with error %v", err)
			}
			switch {
			case err == nil:
				t.Fatal("no error")
			case tt.want != nil && !errors.Is(err, tt.want):
				t.Errorf("error %v, want %v", err, tt.want)
			case tt.wantErr != "" && err.Error() != tt.wantErr:
				t.Errorf("error %q, want %q", err, tt.wantErr)
			}
			freedOnce(t, d)
		})
	}
}

func TestEncodeAs(t *testing.T) {
	le16 := func(vs ...uint16) []byte {
		var b []byte
		for _, v := range vs {
			b = binary.LittleEndian.AppendUint16(b, v)
		}
		return b
	}
	le32 := func(vs ...float32) []byte {
		var b []byte
		for _, v := range vs {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
		}
		return b
	}
	for _, tt := range []struct {
		name    string
		src     []byte
		from    safetensors.Dtype
		to      weightDType
		want    []byte
		same    bool // returns src itself
		wantErr string
	}{
		{name: "BF16 to bf16", src: le16(0x3f80, 0xc040), from: safetensors.BF16, to: weightBF16, want: le16(0x3f80, 0xc040), same: true},
		{name: "F32 to f32", src: le32(1, -0.1), from: safetensors.F32, to: weightF32, want: le32(1, -0.1), same: true},
		{name: "BF16 to f32", src: le16(0x3f80, 0xc040), from: safetensors.BF16, to: weightF32, want: le32(1, -3)},
		// 1+2^-8 lies halfway between bfloat16 1 and 1+2^-7 and rounds to
		// the even 1; 1+3*2^-8 lies halfway between 1+2^-7 and 1+2^-6 and
		// rounds to the even 1+2^-6.
		{name: "F32 to bf16 ties to even", src: le32(1+1.0/256, 1+3.0/256, -0.1), from: safetensors.F32, to: weightBF16, want: le16(0x3f80, 0x3f82, 0xbdcd)},
		{name: "F16 to bf16", src: le16(0x3c00, 0xc000, 0x0001), from: safetensors.F16, to: weightBF16, want: le16(0x3f80, 0xc000, 0x3380)},
		{name: "unsupported dtype", src: []byte{1}, from: safetensors.I8, to: weightBF16, wantErr: `unsupported dtype "I8"`},
		{name: "partial element", src: []byte{1, 2, 3}, from: safetensors.F16, to: weightBF16, wantErr: "3 bytes is not a whole number of F16 elements"},
		{name: "unknown weight dtype", src: le16(1), from: safetensors.F16, to: "f8", wantErr: `unknown weight dtype "f8"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := encodeAs(nil, tt.src, tt.from, tt.to)
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("error %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, tt.want) {
				t.Errorf("encodeAs = %x, want %x", got, tt.want)
			}
			if same := &got[0] == &tt.src[0]; same != tt.same {
				t.Errorf("returned src itself: %v, want %v", same, tt.same)
			}
		})
	}
}

func TestStreamTensorErrors(t *testing.T) {
	for _, tt := range []struct {
		name    string
		from    safetensors.Dtype
		buf     int
		write   error
		wantErr string
	}{
		{name: "buffer below one element", from: safetensors.F32, buf: 3, wantErr: "buffer of 3 bytes holds no F32 element"},
		{name: "unknown dtype", from: "X", buf: 8, wantErr: `unsupported dtype "X"`},
		{name: "write fails", from: safetensors.BF16, buf: 8, write: errFakeCopy, wantErr: errFakeCopy.Error()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := streamTensor(strings.NewReader("0123456789abcdef"), tt.from, weightBF16, make([]byte, tt.buf),
				func(int, []byte) error { return tt.write })
			if err == nil || err.Error() != tt.wantErr {
				t.Errorf("error %v, want %q", err, tt.wantErr)
			}
		})
	}
}

func TestOpenCheckpointTestingModel(t *testing.T) {
	ck, cfg, tied, err := openCheckpoint("")
	if err != nil {
		t.Fatal(err)
	}
	if tied || *cfg != DefaultModelConfig() {
		t.Errorf("testing model: tied %v, config %+v", tied, cfg)
	}
	d := newMemDevice()
	gw, err := streamFake(d, ck, cfg, tied, weightBF16, make([]byte, 64))
	if err != nil {
		t.Fatal(err)
	}
	w := GenerateDeterministicWeights(cfg)
	if want := wantBytes(t, safetensors.F32, weightBF16, w.LMHead); !bytes.Equal(bytesOf(gw.LMHead), want) {
		t.Error("lm_head differs from the testing model's")
	}
	if err := errors.Join(gw.free(d.free), ck.Close()); err != nil {
		t.Fatal(err)
	}
}

func TestStreamModelUnreadTensors(t *testing.T) {
	for _, tt := range []struct {
		name    string
		extra   string // tensor added to the checkpoint, or ""
		wantErr string
	}{
		{name: "none"},
		{name: "rotary inv_freq", extra: "model.layers.0.self_attn.rotary_emb.inv_freq"},
		{name: "q norm", extra: "model.layers.0.self_attn.q_norm.weight", wantErr: "1 tensors the decoder does not read, such as model.layers.0.self_attn.q_norm.weight"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w, cfg := streamTestModel(false, false)
			f, err := w.checkpoint(cfg, safetensors.BF16)
			if err != nil {
				t.Fatal(err)
			}
			var ts []safetensors.Tensor
			for _, info := range f.Tensors {
				tensor, err := f.Tensor(info.Name)
				if err != nil {
					t.Fatal(err)
				}
				ts = append(ts, tensor)
			}
			if tt.extra != "" {
				ts = append(ts, safetensors.Tensor{Name: tt.extra, Dtype: safetensors.BF16, Shape: []int{2}, Data: make([]byte, 4)})
			}
			dir := t.TempDir()
			if err := safetensors.WriteFile(filepath.Join(dir, "model.safetensors"), ts, nil); err != nil {
				t.Fatal(err)
			}
			sh, err := safetensors.OpenSharded(dir)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := sh.Close(); err != nil {
					t.Error(err)
				}
			})
			ck := &recordingCheckpoint{sh, make(map[string]bool)}
			d := newMemDevice()
			_, err = streamFake(d, ck, cfg, false, weightBF16, make([]byte, 4096))
			switch {
			case tt.wantErr == "" && err != nil:
				t.Fatalf("streamModel: %v", err)
			case tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)):
				t.Fatalf("streamModel: err = %v, want %q", err, tt.wantErr)
			}
			if tt.wantErr != "" {
				freedOnce(t, d)
			}
		})
	}
}
