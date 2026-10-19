package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"maps"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

var errFakeUpload = errors.New("fake upload failed")

// hostModel returns a host model with the given number of layers, every
// tensor present, and a config that agrees. Each tensor's first value is
// its id, counting from 1 in upload order: the matrices, then the vectors.
// Matrices have one element; the kth vector, counting from 0, has k+1, so
// that its allocation length says which vector it is. With tied set,
// LMHead is EmbedTokens, as LoadModel makes it.
func hostModel(layers int, tied bool) (*ModelWeights, *ModelConfig) {
	id := float32(0)
	mat := func() []float32 {
		id++
		return []float32{id}
	}
	n := 0
	vec := func() []float32 {
		id++
		n++
		v := make([]float32, n)
		v[0] = id
		return v
	}
	w := &ModelWeights{EmbedTokens: mat()}
	if tied {
		w.LMHead = w.EmbedTokens
	} else {
		w.LMHead = mat()
	}
	w.Layers = make([]LayerWeights, layers)
	for i := range w.Layers {
		l := &w.Layers[i]
		l.WQ, l.WK, l.WV, l.WO, l.WGate, l.WUp, l.WDown = mat(), mat(), mat(), mat(), mat(), mat(), mat()
	}
	w.Norm = vec()
	for i := range w.Layers {
		l := &w.Layers[i]
		l.InputNorm, l.PostNorm, l.QBias, l.KBias, l.VBias = vec(), vec(), vec(), vec(), vec()
	}
	return w, &ModelConfig{NumHiddenLayers: layers}
}

// clearTensors empties the tensors of w whose id, as hostModel numbers
// them, is in ids.
func clearTensors(w *ModelWeights, ids []float32) {
	ts := []*[]float32{&w.EmbedTokens, &w.LMHead, &w.Norm}
	for i := range w.Layers {
		l := &w.Layers[i]
		ts = append(ts, &l.WQ, &l.WK, &l.WV, &l.WO, &l.WGate, &l.WUp, &l.WDown,
			&l.InputNorm, &l.PostNorm, &l.QBias, &l.KBias, &l.VBias)
	}
	for _, t := range ts {
		if len(*t) > 0 && slices.Contains(ids, (*t)[0]) {
			*t = nil
		}
	}
}

// fakeDevice provides the device operations uploadModelWith takes. Its
// upload returns fake buffers named "m" and the matrix id, which log their
// frees in log; the embedding table's buffer fails its first embedFails
// frees. Vectors are zero cudamem.Slices, so free identifies one by its
// slot: the slot filled first fails its first vecFails frees. The *Fail
// fields fail that call of the operation, counting from 1; 0 fails none.
// A failed upload returns its buffer with the error if ownedOnFail is set.
type fakeDevice struct {
	dtype                           weightDType
	uploadFail, allocFail, fillFail int
	ownedOnFail                     bool
	embedFails, vecFails            int

	uploads  []float32 // ids of the matrices uploaded, in order
	bufs     []string  // names of the buffers upload handed over
	allocs   []int     // lengths allocated, in order
	fills    []float32 // ids of the vectors filled, in order
	filled   []*cudamem.Slice[float32]
	attempts map[*cudamem.Slice[float32]]int
	log      []string // matrix frees
	bad      []string
}

func (d *fakeDevice) upload(host []float32, dtype weightDType) (deviceWeight, error) {
	if dtype != d.dtype {
		d.bad = append(d.bad, fmt.Sprintf("upload as %s, want %s", dtype, d.dtype))
	}
	if len(host) == 0 {
		d.bad = append(d.bad, "uploaded an empty matrix")
		return deviceWeight{}, nil
	}
	d.uploads = append(d.uploads, host[0])
	buf := &fakeBuffer{name: fmt.Sprintf("m%v", host[0]), log: &d.log}
	if host[0] == 1 {
		buf.fails, buf.err = d.embedFails, errFakeFree
	}
	if len(d.uploads) == d.uploadFail {
		if !d.ownedOnFail {
			return deviceWeight{}, errFakeUpload
		}
		d.bufs = append(d.bufs, buf.name)
		return deviceWeight{buf: buf, dtype: dtype}, errFakeUpload
	}
	d.bufs = append(d.bufs, buf.name)
	return deviceWeight{buf: buf, dtype: dtype}, nil
}

func (d *fakeDevice) alloc(n int) (cudamem.Slice[float32], error) {
	d.allocs = append(d.allocs, n)
	if len(d.allocs) == d.allocFail {
		return cudamem.Slice[float32]{}, errFakeAlloc
	}
	return cudamem.Slice[float32]{}, nil
}

func (d *fakeDevice) fill(dst *cudamem.Slice[float32], host []float32) error {
	d.filled = append(d.filled, dst)
	if len(host) == 0 {
		d.bad = append(d.bad, "filled an empty vector")
		return nil
	}
	d.fills = append(d.fills, host[0])
	if len(d.fills) == d.fillFail {
		return errFakeCopy
	}
	return nil
}

func (d *fakeDevice) free(s *cudamem.Slice[float32]) error {
	d.attempts[s]++
	if d.vecFails > 0 && len(d.filled) > 0 && s == d.filled[0] {
		d.vecFails--
		return errFakeVec
	}
	return nil
}

type uploadModelFunc func(w *ModelWeights, cfg *ModelConfig, dtype weightDType,
	upload func([]float32, weightDType) (deviceWeight, error),
	alloc func(int) (cudamem.Slice[float32], error),
	fill func(*cudamem.Slice[float32], []float32) error,
	free func(*cudamem.Slice[float32]) error) (*GPUModelWeights, error)

// uploadModelFailures returns how up departs from uploadModelWith, for a
// two-layer model, tied and untied, with a matrix upload, a vector
// allocation or a vector fill failing at a chosen call: the operation's
// error is returned and wraps its cause; nothing is uploaded, allocated or
// filled after the failure; every matrix buffer upload handed over is
// freed exactly once, a tied lm_head never separately; every vector slot
// is passed to free once; and if a free fails, the weights are returned
// with both errors, holding a matrix that could not be freed, and freeing
// them again frees the rest. A successful upload borrows a tied lm_head
// and frees nothing. Absent biases are neither allocated nor filled, but
// their slots are still passed to free; an empty matrix is refused, with
// its index, before anything is uploaded for it.
//
// It cannot see a vector allocation stored in the wrong slot, or a nonzero
// vector returned with an allocation error, since the fake vectors are all
// zero cudamem.Slices.
func uploadModelFailures(up uploadModelFunc) []string {
	const layers = 2
	var bad []string
	for _, tt := range []struct {
		name                            string
		tied                            bool
		uploadFail, allocFail, fillFail int
		ownedOnFail                     bool
		embedFails, vecFails            int
		absent                          []float32 // ids of tensors left empty
		want                            []error
		wantErr                         string // the whole error, if no sentinel names it
	}{
		{name: "success, tied", tied: true},
		{name: "success, untied"},
		{name: "first upload fails", tied: true, uploadFail: 1, want: []error{errFakeUpload}},
		{name: "upload fails holding its buffer", uploadFail: 6, ownedOnFail: true, want: []error{errFakeUpload}},
		{name: "upload after tied head fails", tied: true, uploadFail: 3, want: []error{errFakeUpload}},
		{name: "vector alloc fails", tied: true, allocFail: 4, want: []error{errFakeAlloc}},
		{name: "vector fill fails", fillFail: 7, want: []error{errFakeCopy}},
		{name: "last vector fill fails", tied: true, fillFail: 11, want: []error{errFakeCopy}},
		{name: "fill fails, embedding free fails", tied: true, fillFail: 2, embedFails: 1, want: []error{errFakeCopy, errFakeFree}},
		{name: "upload fails, embedding free fails", uploadFail: 5, embedFails: 1, want: []error{errFakeUpload, errFakeFree}},
		{name: "alloc fails, vector free fails", allocFail: 3, vecFails: 1, want: []error{errFakeAlloc, errFakeVec}},
		// Untied, the vectors are 17 to 27; each layer's biases are its
		// third to fifth. Tied, every id from 3 on is one lower.
		{name: "success, no biases", absent: []float32{20, 21, 22, 25, 26, 27}},
		{name: "fill fails after absent biases", tied: true, fillFail: 4, absent: []float32{19, 20, 21, 24, 25, 26}, want: []error{errFakeCopy}},
		{name: "empty matrix", absent: []float32{4}, wantErr: "upload matrix weight 3: empty"},
		{name: "empty matrix after tied head", tied: true, absent: []float32{3}, wantErr: "upload matrix weight 3: empty"},
	} {
		w, cfg := hostModel(layers, tt.tied)
		clearTensors(w, tt.absent)
		d := &fakeDevice{dtype: weightBF16, uploadFail: tt.uploadFail, allocFail: tt.allocFail, fillFail: tt.fillFail,
			ownedOnFail: tt.ownedOnFail, embedFails: tt.embedFails, vecFails: tt.vecFails,
			attempts: make(map[*cudamem.Slice[float32]]int)}
		res, err := up(w, cfg, weightBF16, d.upload, d.alloc, d.fill, d.free)
		bad = append(bad, d.bad...)
		bad = append(bad, wantJoined(tt.name, err, tt.want...)...)
		if tt.wantErr != "" && (err == nil || err.Error() != tt.wantErr) {
			bad = append(bad, fmt.Sprintf("%s: error %v, want %q", tt.name, err, tt.wantErr))
		}
		failing := len(tt.want) > 0 || tt.wantErr != ""
		if !failing && err != nil {
			bad = append(bad, fmt.Sprintf("%s: %v", tt.name, err))
		}

		// What the calls should have been, up to the failure. A tied
		// lm_head takes no id, an absent vector is skipped, and an empty
		// matrix stops the upload before it.
		nmat, nvec := 2+7*layers, 1+5*layers
		if tt.tied {
			nmat--
		}
		var mats []float32
		for id := range nmat {
			mats = append(mats, float32(id+1))
		}
		var vecs, lens []int
		for k := range nvec {
			if id := nmat + k + 1; !slices.Contains(tt.absent, float32(id)) {
				vecs, lens = append(vecs, id), append(lens, k+1)
			}
		}
		wantFills := vecs
		for i, id := range mats {
			if slices.Contains(tt.absent, id) {
				mats, lens, wantFills = mats[:i], nil, nil
				break
			}
		}
		switch {
		case tt.uploadFail > 0:
			mats, lens, wantFills = mats[:tt.uploadFail], nil, nil
		case tt.allocFail > 0:
			lens, wantFills = lens[:tt.allocFail], vecs[:tt.allocFail-1]
		case tt.fillFail > 0:
			lens, wantFills = lens[:tt.fillFail], vecs[:tt.fillFail]
		}
		if !slices.Equal(d.uploads, mats) {
			bad = append(bad, fmt.Sprintf("%s: uploaded %v, want %v", tt.name, d.uploads, mats))
		}
		if !slices.Equal(d.allocs, lens) {
			bad = append(bad, fmt.Sprintf("%s: allocated lengths %v, want %v", tt.name, d.allocs, lens))
		}
		var fills []float32
		for _, v := range wantFills {
			fills = append(fills, float32(v))
		}
		if !slices.Equal(d.fills, fills) {
			bad = append(bad, fmt.Sprintf("%s: filled %v, want %v", tt.name, d.fills, fills))
		}

		if !failing {
			if res == nil {
				bad = append(bad, tt.name+": no weights")
				continue
			}
			if len(d.log) != 0 || len(d.attempts) != 0 {
				bad = append(bad, fmt.Sprintf("%s: freed %q and %d vectors", tt.name, d.log, len(d.attempts)))
			}
			if res.DType != weightBF16 {
				bad = append(bad, fmt.Sprintf("%s: dtype %s", tt.name, res.DType))
			}
			if tied := res.LMHead.borrowed && res.LMHead.buf == res.EmbedTokens.buf; tied != tt.tied {
				bad = append(bad, fmt.Sprintf("%s: lm_head borrows the embedding table: %v, want %v", tt.name, tied, tt.tied))
			}
			if err := res.free(d.free); err != nil {
				bad = append(bad, fmt.Sprintf("%s: free: %v", tt.name, err))
			}
		}

		// Every vector slot is passed to free once by the failed upload, or
		// by the free above, and once more by each retry; a filled slot
		// must be one of them.
		round := 1
		check := func(when string) {
			if len(d.attempts) != nvec {
				bad = append(bad, fmt.Sprintf("%s, %s: %d vector slots freed, want %d", tt.name, when, len(d.attempts), nvec))
			}
			for _, n := range d.attempts {
				if n != round {
					bad = append(bad, fmt.Sprintf("%s, %s: a vector slot freed %d times, want %d", tt.name, when, n, round))
					break
				}
			}
			for _, s := range d.filled {
				if d.attempts[s] == 0 {
					bad = append(bad, fmt.Sprintf("%s, %s: a filled vector slot never freed", tt.name, when))
					break
				}
			}
		}
		check("first free")

		failed := tt.embedFails > 0 || tt.vecFails > 0
		if failing {
			if got := res != nil; got != failed {
				bad = append(bad, fmt.Sprintf("%s: weights returned %v, want %v", tt.name, got, failed))
			}
			if res != nil && tt.embedFails > 0 && res.EmbedTokens.buf == nil {
				bad = append(bad, tt.name+": embedding table dropped, cannot retry")
			}
			if res != nil {
				round++
				if err := res.free(d.free); err != nil {
					bad = append(bad, fmt.Sprintf("%s: retry: %v", tt.name, err))
				}
				check("retry")
			}
		}
		for _, name := range d.bufs {
			if n := countOf(d.log, name); n != 1 {
				bad = append(bad, fmt.Sprintf("%s: %s freed %d times, want 1", tt.name, name, n))
			}
		}
		if len(d.log) != len(d.bufs) {
			bad = append(bad, fmt.Sprintf("%s: freed %q, want %q once each", tt.name, d.log, d.bufs))
		}
	}

	w, cfg := hostModel(1, false)
	d := &fakeDevice{attempts: make(map[*cudamem.Slice[float32]]int)}
	cfg.NumHiddenLayers = 2
	if res, err := up(w, cfg, weightBF16, d.upload, d.alloc, d.fill, d.free); res != nil || err == nil || len(d.uploads) != 0 {
		bad = append(bad, fmt.Sprintf("layer count mismatch: %v, %v after %d uploads", res, err, len(d.uploads)))
	}
	return bad
}

// countOf returns the number of times s occurs in list.
func countOf(list []string, s string) int {
	n := 0
	for _, x := range list {
		if x == s {
			n++
		}
	}
	return n
}

// TestUploadModelWith checks uploadModelWith's error paths with fake device
// operations. That uploadModel passes the real ones is checked by
// TestDeviceClosures.
func TestUploadModelWith(t *testing.T) {
	for _, f := range uploadModelFailures(uploadModelWith) {
		t.Error(f)
	}
	type (
		upF    = func([]float32, weightDType) (deviceWeight, error)
		allocF = func(int) (cudamem.Slice[float32], error)
		fillF  = func(*cudamem.Slice[float32], []float32) error
		freeF  = func(*cudamem.Slice[float32]) error
	)
	for _, m := range []struct {
		name string
		wrap func(uploadModelFunc) uploadModelFunc
	}{
		{"U1 weights not freed dropped", func(up uploadModelFunc) uploadModelFunc {
			return func(w *ModelWeights, cfg *ModelConfig, dtype weightDType, upload upF, alloc allocF, fill fillF, free freeF) (*GPUModelWeights, error) {
				res, err := up(w, cfg, dtype, upload, alloc, fill, free)
				if err != nil {
					return nil, err
				}
				return res, nil
			}
		}},
		{"U2 fill error ignored", func(up uploadModelFunc) uploadModelFunc {
			return func(w *ModelWeights, cfg *ModelConfig, dtype weightDType, upload upF, alloc allocF, fill fillF, free freeF) (*GPUModelWeights, error) {
				return up(w, cfg, dtype, upload, alloc, func(s *cudamem.Slice[float32], host []float32) error {
					fill(s, host)
					return nil
				}, free)
			}
		}},
		{"U3 alloc error ignored", func(up uploadModelFunc) uploadModelFunc {
			return func(w *ModelWeights, cfg *ModelConfig, dtype weightDType, upload upF, alloc allocF, fill fillF, free freeF) (*GPUModelWeights, error) {
				return up(w, cfg, dtype, upload, func(n int) (cudamem.Slice[float32], error) {
					s, _ := alloc(n)
					return s, nil
				}, fill, free)
			}
		}},
		{"U4 vectors not freed", func(up uploadModelFunc) uploadModelFunc {
			return func(w *ModelWeights, cfg *ModelConfig, dtype weightDType, upload upF, alloc allocF, fill fillF, free freeF) (*GPUModelWeights, error) {
				return up(w, cfg, dtype, upload, alloc, fill, func(*cudamem.Slice[float32]) error { return nil })
			}
		}},
		{"U5 failed upload's buffer dropped", func(up uploadModelFunc) uploadModelFunc {
			return func(w *ModelWeights, cfg *ModelConfig, dtype weightDType, upload upF, alloc allocF, fill fillF, free freeF) (*GPUModelWeights, error) {
				return up(w, cfg, dtype, func(host []float32, dtype weightDType) (deviceWeight, error) {
					m, err := upload(host, dtype)
					if err != nil {
						return deviceWeight{}, err
					}
					return m, nil
				}, alloc, fill, free)
			}
		}},
	} {
		if len(uploadModelFailures(m.wrap(uploadModelWith))) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// fakeDeviceSlice returns a cudamem.Slice[T] with device pointer p and
// nothing allocated behind it, for fakes that never hand it to cudamem.
// cudamem makes a nonzero Slice only by allocating on a device, so this
// sets the unexported pointer field; it reports false if there is no such
// field or Ptr does not then return p.
func fakeDeviceSlice[T any](p driver.DevicePtr) (cudamem.Slice[T], bool) {
	var s cudamem.Slice[T]
	f := reflect.ValueOf(&s).Elem().FieldByName("ptr")
	if !f.IsValid() || f.Type() != reflect.TypeFor[driver.DevicePtr]() {
		return s, false
	}
	reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem().SetUint(uint64(p))
	return s, s.Ptr() == p
}

// fakeScratchDevice provides the device operations newGPUScratchpadWith
// takes. Allocations of either type are numbered from 1 in call order, and
// each returns a buffer whose device pointer is its number. The
// allocFail'th fails, returning its buffer with the error if ownedOnFail is
// set and a zero buffer otherwise. The first free of the buffer numbered
// freeFail fails. A free zeroes a buffer only when it succeeds, as
// freeOwned does.
type fakeScratchDevice struct {
	allocFail   int
	ownedOnFail bool
	freeFail    driver.DevicePtr

	allocs []string           // "f32 n" or "i32 n" for each allocation of n elements
	freed  []driver.DevicePtr // buffers freed, in order
	bad    []string
}

func fakeScratchAlloc[T any](d *fakeScratchDevice, kind string) func(int) (cudamem.Slice[T], error) {
	return func(n int) (cudamem.Slice[T], error) {
		d.allocs = append(d.allocs, fmt.Sprintf("%s %d", kind, n))
		s, ok := fakeDeviceSlice[T](driver.DevicePtr(len(d.allocs)))
		if !ok {
			d.bad = append(d.bad, "cannot make a fake cudamem.Slice")
		}
		if len(d.allocs) == d.allocFail {
			if !d.ownedOnFail {
				return cudamem.Slice[T]{}, errFakeAlloc
			}
			return s, errFakeAlloc
		}
		return s, nil
	}
}

func fakeScratchFree[T any](d *fakeScratchDevice) func(*cudamem.Slice[T]) error {
	return func(s *cudamem.Slice[T]) error {
		p := s.Ptr()
		if p == 0 {
			return nil
		}
		if p == d.freeFail {
			d.freeFail = 0
			return errFakeFree
		}
		d.freed = append(d.freed, p)
		*s = cudamem.Slice[T]{}
		return nil
	}
}

// scratchBuffers returns the device pointer of each of sp's device
// buffers, by field name.
func scratchBuffers(sp *GPUScratchpad) map[string]driver.DevicePtr {
	return map[string]driver.DevicePtr{
		"X": sp.X.Ptr(), "NormX": sp.NormX.Ptr(), "Q": sp.Q.Ptr(), "K": sp.K.Ptr(), "V": sp.V.Ptr(),
		"AttnOut": sp.AttnOut.Ptr(), "Gate": sp.Gate.Ptr(), "Up": sp.Up.Ptr(), "FFNAct": sp.FFNAct.Ptr(),
		"FFNOut": sp.FFNOut.Ptr(), "Logits": sp.Logits.Ptr(),
		"TokenIn": sp.TokenIn.Ptr(), "TokenOut": sp.TokenOut.Ptr(),
	}
}

type scratchpadFunc func(cfg *ModelConfig,
	allocF32 func(int) (cudamem.Slice[float32], error),
	allocI32 func(int) (cudamem.Slice[int32], error),
	freeF32 func(*cudamem.Slice[float32]) error,
	freeI32 func(*cudamem.Slice[int32]) error) (*GPUScratchpad, error)

// scratchpadFailures returns how newSP departs from newGPUScratchpadWith:
// each buffer allocated once, with the type and length its role takes
// under the config, and the host token buffers of length 1; when the kth
// allocation fails, for every k, nothing allocated after it, the
// allocation error returned, and every buffer allocated before it freed
// once, together with a buffer the failed allocation returned; and when a
// free fails too, both errors and the scratchpad returned, holding only
// the buffer that could not be freed, which a retry then frees.
func scratchpadFailures(newSP scratchpadFunc) []string {
	var bad []string
	run := func(cfg *ModelConfig, d *fakeScratchDevice) (*GPUScratchpad, error) {
		sp, err := newSP(cfg, fakeScratchAlloc[float32](d, "f32"), fakeScratchAlloc[int32](d, "i32"),
			fakeScratchFree[float32](d), fakeScratchFree[int32](d))
		bad = append(bad, d.bad...)
		return sp, err
	}
	for _, c := range []struct {
		name                        string
		cfg                         ModelConfig
		hidden, q, kv, inter, vocab int
	}{
		{"explicit head dim", ModelConfig{HiddenSize: 10, NumAttentionHeads: 4, NumKeyValueHeads: 2, ExplicitHeadDim: 3, IntermediateSize: 14, VocabSize: 17},
			10, 4 * 3, 2 * 3, 14, 17},
		{"derived head dim", ModelConfig{HiddenSize: 16, NumAttentionHeads: 4, NumKeyValueHeads: 1, IntermediateSize: 9, VocabSize: 11},
			16, 4 * 4, 1 * 4, 9, 11},
	} {
		f32 := func(n int) string { return fmt.Sprintf("f32 %d", n) }
		want := map[string]string{
			"X": f32(c.hidden), "NormX": f32(c.hidden), "Q": f32(c.q), "K": f32(c.kv), "V": f32(c.kv),
			"AttnOut": f32(c.q), "Gate": f32(c.inter), "Up": f32(c.inter), "FFNAct": f32(c.inter),
			"FFNOut": f32(c.hidden), "Logits": f32(c.vocab),
			"TokenIn": "i32 1", "TokenOut": "i32 1",
		}

		d := &fakeScratchDevice{}
		sp, err := run(&c.cfg, d)
		if err != nil || sp == nil {
			bad = append(bad, fmt.Sprintf("%s: %v, %v", c.name, sp, err))
			continue
		}
		if len(d.allocs) != len(want) || len(d.freed) != 0 {
			bad = append(bad, fmt.Sprintf("%s: allocated %q and freed %v, want %d allocations and no frees", c.name, d.allocs, d.freed, len(want)))
		}
		for name, p := range scratchBuffers(sp) {
			got := "nothing"
			if p > 0 && int(p) <= len(d.allocs) {
				got = d.allocs[p-1]
			}
			if got != want[name] {
				bad = append(bad, fmt.Sprintf("%s: %s holds %s, want %s", c.name, name, got, want[name]))
			}
		}
		if len(sp.HostTokenIn) != 1 || len(sp.HostTokenOut) != 1 {
			bad = append(bad, fmt.Sprintf("%s: host token buffers of length %d and %d, want 1", c.name, len(sp.HostTokenIn), len(sp.HostTokenOut)))
		}
		allocs := d.allocs

		for k := 1; k <= len(allocs); k++ {
			for _, owned := range []bool{false, true} {
				name := fmt.Sprintf("%s, allocation %d fails, buffer returned %v", c.name, k, owned)
				d := &fakeScratchDevice{allocFail: k, ownedOnFail: owned}
				sp, err := run(&c.cfg, d)
				if sp != nil || !errors.Is(err, errFakeAlloc) {
					bad = append(bad, fmt.Sprintf("%s: %v, %v; want no scratchpad and the allocation error", name, sp, err))
				}
				if !slices.Equal(d.allocs, allocs[:k]) {
					bad = append(bad, fmt.Sprintf("%s: allocated %q, want %q", name, d.allocs, allocs[:k]))
				}
				freed := upTo(k - 1)
				if owned {
					freed = upTo(k)
				}
				if got := slices.Sorted(slices.Values(d.freed)); !slices.Equal(got, freed) {
					bad = append(bad, fmt.Sprintf("%s: freed %v, want %v once each", name, got, freed))
				}
			}
		}

		for _, tt := range []struct {
			allocFail int
			owned     bool
			freeFail  driver.DevicePtr
		}{
			{5, false, 2},
			{len(allocs), false, driver.DevicePtr(len(allocs) - 1)},
			{len(allocs), true, driver.DevicePtr(len(allocs))},
			{1, true, 1},
		} {
			name := fmt.Sprintf("%s, allocation %d fails, buffer returned %v, free of %d fails", c.name, tt.allocFail, tt.owned, tt.freeFail)
			d := &fakeScratchDevice{allocFail: tt.allocFail, ownedOnFail: tt.owned, freeFail: tt.freeFail}
			sp, err := run(&c.cfg, d)
			bad = append(bad, wantJoined(name, err, errFakeAlloc, errFakeFree)...)
			if sp == nil {
				bad = append(bad, name+": no scratchpad, cannot retry")
				continue
			}
			for field, p := range scratchBuffers(sp) {
				if p != 0 && p != tt.freeFail {
					bad = append(bad, fmt.Sprintf("%s: %s still holds buffer %d", name, field, p))
				}
			}
			if !slices.Contains(slices.Collect(maps.Values(scratchBuffers(sp))), tt.freeFail) {
				bad = append(bad, fmt.Sprintf("%s: buffer %d not kept", name, tt.freeFail))
			}
			if err := sp.release(fakeScratchFree[float32](d), fakeScratchFree[int32](d)); err != nil {
				bad = append(bad, fmt.Sprintf("%s: retry: %v", name, err))
			}
			freed := upTo(tt.allocFail - 1)
			if tt.owned {
				freed = upTo(tt.allocFail)
			}
			if got := slices.Sorted(slices.Values(d.freed)); !slices.Equal(got, freed) {
				bad = append(bad, fmt.Sprintf("%s: freed %v after the retry, want %v once each", name, got, freed))
			}
		}
	}
	return bad
}

// upTo returns the device pointers 1 to n.
func upTo(n int) []driver.DevicePtr {
	var ps []driver.DevicePtr
	for p := range n {
		ps = append(ps, driver.DevicePtr(p+1))
	}
	return ps
}

// TestNewGPUScratchpadWith checks newGPUScratchpadWith with fake device
// operations. That NewGPUScratchpad passes the real ones is checked by
// TestDeviceClosures.
func TestNewGPUScratchpadWith(t *testing.T) {
	for _, f := range scratchpadFailures(newGPUScratchpadWith) {
		t.Error(f)
	}
	type (
		allocF32 = func(int) (cudamem.Slice[float32], error)
		allocI32 = func(int) (cudamem.Slice[int32], error)
		freeF32  = func(*cudamem.Slice[float32]) error
		freeI32  = func(*cudamem.Slice[int32]) error
	)
	for _, m := range []struct {
		name string
		wrap func(scratchpadFunc) scratchpadFunc
	}{
		{"N1 scratchpad not freed dropped", func(newSP scratchpadFunc) scratchpadFunc {
			return func(cfg *ModelConfig, a32 allocF32, a1 allocI32, f32 freeF32, f1 freeI32) (*GPUScratchpad, error) {
				sp, err := newSP(cfg, a32, a1, f32, f1)
				if err != nil {
					return nil, err
				}
				return sp, nil
			}
		}},
		{"N2 buffer returned with its error dropped", func(newSP scratchpadFunc) scratchpadFunc {
			return func(cfg *ModelConfig, a32 allocF32, a1 allocI32, f32 freeF32, f1 freeI32) (*GPUScratchpad, error) {
				return newSP(cfg, func(n int) (cudamem.Slice[float32], error) {
					s, err := a32(n)
					if err != nil {
						return cudamem.Slice[float32]{}, err
					}
					return s, nil
				}, a1, f32, f1)
			}
		}},
		{"N3 token buffers not freed", func(newSP scratchpadFunc) scratchpadFunc {
			return func(cfg *ModelConfig, a32 allocF32, a1 allocI32, f32 freeF32, f1 freeI32) (*GPUScratchpad, error) {
				return newSP(cfg, a32, a1, f32, func(*cudamem.Slice[int32]) error { return nil })
			}
		}},
		{"N4 token buffer of length 2", func(newSP scratchpadFunc) scratchpadFunc {
			return func(cfg *ModelConfig, a32 allocF32, a1 allocI32, f32 freeF32, f1 freeI32) (*GPUScratchpad, error) {
				return newSP(cfg, a32, func(n int) (cudamem.Slice[int32], error) { return a1(2 * n) }, f32, f1)
			}
		}},
	} {
		if len(scratchpadFailures(m.wrap(newGPUScratchpadWith))) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
	if s, ok := fakeDeviceSlice[float32](7); !ok || s.Ptr() != 7 || s.Len() != 0 {
		t.Errorf("fakeDeviceSlice(7) = %v, %v; want a zero-length slice at 7", s, ok)
	}
}

// deviceClosures lists the functions that pass real device operations to a
// function tested with fakes. Each must be a single return of call, whose
// string form, as types.ExprString writes it, is call; and each function
// literal among its arguments must be a single return of the
// corresponding expression in lits, with its parameters' names for %[1]s
// and %[2]s.
var deviceClosures = []struct {
	file, fn string
	call     string
	lits     []string
}{
	{
		"weights_bf16.go", "uploadWeight",
		"uploadWeightWith(host, dtype, (func(n int) (cudamem.Slice[float32], error) literal), (func(n int) (cudamem.Slice[uint16], error) literal))",
		[]string{"cudamem.Alloc[float32](dev, %[1]s)", "cudamem.Alloc[uint16](dev, %[1]s)"},
	},
	{
		"gpu_model.go", "uploadModel",
		"uploadModelWith(w, cfg, dtype, (func(host []float32, dtype weightDType) (deviceWeight, error) literal), (func(n int) (cudamem.Slice[float32], error) literal), (*cudamem.Slice[float32]).CopyFromHost, freeOwned[cudamem.Slice[float32]])",
		[]string{"uploadWeight(dev, %[1]s, %[2]s)", "cudamem.Alloc[float32](dev, %[1]s)"},
	},
	{
		"gpu_model.go", "NewGPUScratchpad",
		"newGPUScratchpadWith(cfg, (func(n int) (cudamem.Slice[float32], error) literal), (func(n int) (cudamem.Slice[int32], error) literal), freeOwned[cudamem.Slice[float32]], freeOwned[cudamem.Slice[int32]])",
		[]string{"cudamem.Alloc[float32](dev, %[1]s)", "cudamem.Alloc[int32](dev, %[1]s)"},
	},
}

// deviceClosureFailures reports each function in deviceClosures declared
// in src, the contents of file, that departs from its entry.
func deviceClosureFailures(file, src string) ([]string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), file, src, 0)
	if err != nil {
		return nil, err
	}
	var bad []string
	for _, d := range deviceClosures {
		if d.file != file {
			continue
		}
		var fn *ast.FuncDecl
		for _, decl := range f.Decls {
			if fd, ok := decl.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == d.fn {
				fn = fd
			}
		}
		var call ast.Expr
		if fn != nil {
			call = singleReturn(fn.Body)
		}
		if call == nil {
			bad = append(bad, fmt.Sprintf("%s: not a single return of %s", d.fn, d.call))
			continue
		}
		if got := types.ExprString(call); got != d.call {
			bad = append(bad, fmt.Sprintf("%s: returns %s, want %s", d.fn, got, d.call))
			continue
		}
		var lits []*ast.FuncLit
		for _, arg := range call.(*ast.CallExpr).Args {
			if lit, ok := arg.(*ast.FuncLit); ok {
				lits = append(lits, lit)
			}
		}
		if len(lits) != len(d.lits) {
			bad = append(bad, fmt.Sprintf("%s: %d function literals, want %d", d.fn, len(lits), len(d.lits)))
			continue
		}
		for i, lit := range lits {
			var params []any
			for _, field := range lit.Type.Params.List {
				for _, name := range field.Names {
					params = append(params, name.Name)
				}
			}
			want := fmt.Sprintf(d.lits[i], params...)
			got := "not a single return"
			if e := singleReturn(lit.Body); e != nil {
				got = types.ExprString(e)
			}
			if got != want {
				bad = append(bad, fmt.Sprintf("%s: function literal %d returns %s, want %s", d.fn, i, got, want))
			}
		}
	}
	return bad, nil
}

// singleReturn returns the one result of body if body is a single return
// statement with one result, or nil.
func singleReturn(body *ast.BlockStmt) ast.Expr {
	if body == nil || len(body.List) != 1 {
		return nil
	}
	ret, ok := body.List[0].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		return nil
	}
	return ret.Results[0]
}

// TestDeviceClosures checks that uploadWeight, uploadModel and
// NewGPUScratchpad pass the real device operations, which no CPU test
// runs, to the functions tested with fakes: in particular that each
// allocation closure allocates exactly the length it is asked for.
// Must-fail controls edit the real source.
func TestDeviceClosures(t *testing.T) {
	srcs := make(map[string]string)
	for _, d := range deviceClosures {
		b, err := os.ReadFile(d.file)
		if err != nil {
			t.Fatal(err)
		}
		srcs[d.file] = string(b)
	}
	for file, src := range srcs {
		bad, err := deviceClosureFailures(file, src)
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range bad {
			t.Error(f)
		}
	}
	// vec is uploadModel's vector allocator, which NewGPUScratchpad's
	// float32 allocator repeats word for word.
	const vec = "return uploadWeight(dev, host, dtype) },\n\t\tfunc(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) }"
	for _, m := range []struct{ name, file, old, new string }{
		{"C1 f32 allocates n+1", "weights_bf16.go", "return cudamem.Alloc[float32](dev, n) }", "return cudamem.Alloc[float32](dev, n+1) }"},
		{"C2 bf16 allocates n+1", "weights_bf16.go", "return cudamem.Alloc[uint16](dev, n) }", "return cudamem.Alloc[uint16](dev, n+1) }"},
		{"C3 parameter renamed, length fixed", "weights_bf16.go", "func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) }", "func(m int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, len(host)) }"},
		{"C4 vector allocates n+1", "gpu_model.go", vec, strings.Replace(vec, "(dev, n) }", "(dev, n+1) }", 1)},
		{"C5 matrices uploaded as f32", "gpu_model.go", "return uploadWeight(dev, host, dtype) }", "return uploadWeight(dev, host, weightF32) }"},
		{"C6 vector fill skipped", "gpu_model.go", "(*cudamem.Slice[float32]).CopyFromHost,", "func(*cudamem.Slice[float32], []float32) error { return nil },"},
		{"C7 vector free does not zero", "gpu_model.go", "\t\tfreeOwned[cudamem.Slice[float32]])\n}", "\t\tfunc(s *cudamem.Slice[float32]) error { return s.Free() })\n}"},
		{"C8 statement before the return", "gpu_model.go", vec, strings.Replace(vec, "{ return cudamem.Alloc", "{ n++; return cudamem.Alloc", 1)},
		{"C9 f32 closure size doubled", "weights_bf16.go", "func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) }", "func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, 2*n) }"},
		{"C10 scratch f32 allocates n+1", "gpu_model.go", "\treturn newGPUScratchpadWith(cfg,\n\t\tfunc(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) }", "\treturn newGPUScratchpadWith(cfg,\n\t\tfunc(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n+1) }"},
		{"C11 token allocates 2*n", "gpu_model.go", "return cudamem.Alloc[int32](dev, n) }", "return cudamem.Alloc[int32](dev, 2*n) }"},
		{"C12 token free does not zero", "gpu_model.go", "freeOwned[cudamem.Slice[int32]])\n}", "func(s *cudamem.Slice[int32]) error { return s.Free() })\n}"},
		{"C13 scratchpad frees dropped", "gpu_model.go", "freeOwned[cudamem.Slice[float32]], freeOwned[cudamem.Slice[int32]])\n}", "func(*cudamem.Slice[float32]) error { return nil }, freeOwned[cudamem.Slice[int32]])\n}"},
	} {
		src := srcs[m.file]
		if !strings.Contains(src, m.old) {
			t.Fatalf("must-fail control %s: %q not in %s", m.name, m.old, m.file)
		}
		bad, err := deviceClosureFailures(m.file, strings.Replace(src, m.old, m.new, 1))
		if err != nil {
			t.Fatalf("must-fail control %s: %v", m.name, err)
		}
		if len(bad) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// forwardKernelDispatchFailures inspects enqueueGPUForwardPass in src, the
// contents of gpu_model.go, to verify that weightF32 selects gm.fusedQKVFn
// and gm.fusedSwiGLUFn, weightBF16 selects gm.fusedQKVBF16Fn and
// gm.fusedSwiGLUBF16Fn, and the -fuse-egraph branches guard and launch
// fusedQKV and fusedSwiGLU.
func forwardKernelDispatchFailures(src string) ([]string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), "gpu_model.go", src, 0)
	if err != nil {
		return nil, err
	}
	var fn *ast.FuncDecl
	for _, decl := range f.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "enqueueGPUForwardPass" {
			fn = fd
		}
	}
	if fn == nil || fn.Body == nil {
		return []string{"enqueueGPUForwardPass not found"}, nil
	}

	var assigns []string
	var bf16Assigns []string
	var fuseConds []string
	var launched []string
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.AssignStmt:
			var lhs, rhs []string
			for _, e := range s.Lhs {
				lhs = append(lhs, types.ExprString(e))
			}
			for _, e := range s.Rhs {
				rhs = append(rhs, types.ExprString(e))
			}
			line := strings.Join(lhs, ", ") + " " + s.Tok.String() + " " + strings.Join(rhs, ", ")
			if strings.HasPrefix(line, "gemv, embed") || strings.HasPrefix(line, "fusedQKV, fusedSwiGLU") {
				assigns = append(assigns, line)
			}
		case *ast.IfStmt:
			cond := types.ExprString(s.Cond)
			if cond == "gpuW.DType == weightBF16" && s.Body != nil {
				for _, stmt := range s.Body.List {
					as, ok := stmt.(*ast.AssignStmt)
					if !ok {
						bf16Assigns = append(bf16Assigns, fmt.Sprintf("non-assign %T", stmt))
						continue
					}
					var lhs, rhs []string
					for _, e := range as.Lhs {
						lhs = append(lhs, types.ExprString(e))
					}
					for _, e := range as.Rhs {
						rhs = append(rhs, types.ExprString(e))
					}
					bf16Assigns = append(bf16Assigns, strings.Join(lhs, ", ")+" "+as.Tok.String()+" "+strings.Join(rhs, ", "))
				}
			}
			if strings.Contains(cond, "fuseEGraphFlag") {
				fuseConds = append(fuseConds, cond)
			}
		case *ast.CallExpr:
			if id, ok := s.Fun.(*ast.Ident); ok && id.Name == "launcher" && len(s.Args) == 1 {
				launched = append(launched, types.ExprString(s.Args[0]))
			}
		}
		return true
	})

	var bad []string
	wantAssigns := []string{
		"gemv, embed := gm.gemvFn, gm.embedFn",
		"fusedQKV, fusedSwiGLU := gm.fusedQKVFn, gm.fusedSwiGLUFn",
		"gemv, embed = gm.gemvBF16Fn, gm.embedBF16Fn",
		"fusedQKV, fusedSwiGLU = gm.fusedQKVBF16Fn, gm.fusedSwiGLUBF16Fn",
	}
	if !slices.Equal(assigns, wantAssigns) {
		bad = append(bad, fmt.Sprintf("kernel assignments = %q, want %q", assigns, wantAssigns))
	}
	wantBF16 := []string{
		"gemv, embed = gm.gemvBF16Fn, gm.embedBF16Fn",
		"fusedQKV, fusedSwiGLU = gm.fusedQKVBF16Fn, gm.fusedSwiGLUBF16Fn",
	}
	if !slices.Equal(bf16Assigns, wantBF16) {
		bad = append(bad, fmt.Sprintf("weightBF16 block = %q, want %q", bf16Assigns, wantBF16))
	}
	wantConds := []string{
		"*fuseEGraphFlag && !fusedKernelsQualified",
		"*fuseEGraphFlag && fusedQKV != 0",
		"*fuseEGraphFlag && fusedSwiGLU != 0",
	}
	if !slices.Equal(fuseConds, wantConds) {
		bad = append(bad, fmt.Sprintf("fuseEGraphFlag conditions = %q, want %q", fuseConds, wantConds))
	}
	wantLaunched := []string{
		"embed",
		"attnFn",
		"fusedQKV",
		"gm.normFn",
		"gemv",
		"gemv",
		"gemv",
		"gm.ropeFn",
		"gm.storeKVFn",
		"attnFn",
		"gemv",
		"gm.addResFn",
		"gm.normFn",
		"gemv",
		"gemv",
		"fusedSwiGLU",
		"gm.swigluFn",
		"gemv",
		"gm.addResFn",
		"gm.normFn",
		"gemv",
		"sampleFn",
	}
	if !slices.Equal(launched, wantLaunched) {
		bad = append(bad, fmt.Sprintf("launcher arguments = %q, want %q", launched, wantLaunched))
	}
	return bad, nil
}

// TestForwardKernelDispatch checks both qualified and unqualified
// -fuse-egraph states in enqueueGPUForwardPass and verifies that weightBF16
// selects and dispatches fusedQKVBF16Fn and fusedSwiGLUBF16Fn when
// -fuse-egraph is set.
func TestForwardKernelDispatch(t *testing.T) {
	if !fusedKernelsQualified {
		t.Fatal("fusedKernelsQualified = false, want true")
	}
	if *fuseEGraphFlag {
		t.Fatal("-fuse-egraph default = true, want false")
	}

	saveFlag, saveQualified := *fuseEGraphFlag, fusedKernelsQualified
	defer func() { *fuseEGraphFlag, fusedKernelsQualified = saveFlag, saveQualified }()

	var log []string
	w := fakeModel(&log, true, 0)
	cfg := DefaultModelConfig()
	cfg.NumHiddenLayers = 1

	// When unqualified, -fuse-egraph is refused before touching model or token checks.
	*fuseEGraphFlag, fusedKernelsQualified = true, false
	if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, -1, 0, nil); err == nil || !strings.Contains(err.Error(), "not qualified") {
		t.Fatalf("unqualified -fuse-egraph: got %v, want qualification refusal", err)
	}

	// When qualified, -fuse-egraph with weightBF16 is accepted past the qualification
	// and dtype gates and reaches the token and position checks.
	*fuseEGraphFlag, fusedKernelsQualified = true, true
	if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, -1, 0, nil); err == nil || !strings.Contains(err.Error(), "outside vocabulary") {
		t.Fatalf("qualified bf16 -fuse-egraph token check: got %v, want outside vocabulary", err)
	}
	if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, 0, -1, nil); err == nil || !strings.Contains(err.Error(), "outside the kv cache") {
		t.Fatalf("qualified bf16 -fuse-egraph pos check: got %v, want outside the kv cache", err)
	}

	b, err := os.ReadFile("gpu_model.go")
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	bad, err := forwardKernelDispatchFailures(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range bad {
		t.Error(f)
	}

	for _, m := range []struct{ name, old, new string }{
		{"F1 bf16 fused assignment omitted", "\t\tfusedQKV, fusedSwiGLU = gm.fusedQKVBF16Fn, gm.fusedSwiGLUBF16Fn\n", ""},
		{"F2 bf16 fused kernels swapped", "fusedQKV, fusedSwiGLU = gm.fusedQKVBF16Fn, gm.fusedSwiGLUBF16Fn", "fusedQKV, fusedSwiGLU = gm.fusedSwiGLUBF16Fn, gm.fusedQKVBF16Fn"},
		{"F3 qkv launch uses f32 field directly", "launcher(fusedQKV)", "launcher(gm.fusedQKVFn)"},
		{"F4 swiglu launch uses f32 field directly", "launcher(fusedSwiGLU)", "launcher(gm.fusedSwiGLUFn)"},
		{"F5 qkv guard checks f32 field", "*fuseEGraphFlag && fusedQKV != 0", "*fuseEGraphFlag && gm.fusedQKVFn != 0"},
		{"F6 swiglu guard checks f32 field", "*fuseEGraphFlag && fusedSwiGLU != 0", "*fuseEGraphFlag && gm.fusedSwiGLUFn != 0"},
	} {
		if !strings.Contains(src, m.old) {
			t.Fatalf("must-fail control %s: %q not in gpu_model.go", m.name, m.old)
		}
		bad, err := forwardKernelDispatchFailures(strings.Replace(src, m.old, m.new, 1))
		if err != nil {
			t.Fatalf("must-fail control %s: %v", m.name, err)
		}
		if len(bad) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}
