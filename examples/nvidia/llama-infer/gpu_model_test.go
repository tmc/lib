package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"slices"
	"strings"
	"testing"

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
	d.fills = append(d.fills, host[0])
	d.filled = append(d.filled, dst)
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
// and frees nothing.
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
		want                            []error
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
	} {
		w, cfg := hostModel(layers, tt.tied)
		d := &fakeDevice{dtype: weightBF16, uploadFail: tt.uploadFail, allocFail: tt.allocFail, fillFail: tt.fillFail,
			ownedOnFail: tt.ownedOnFail, embedFails: tt.embedFails, vecFails: tt.vecFails,
			attempts: make(map[*cudamem.Slice[float32]]int)}
		res, err := up(w, cfg, weightBF16, d.upload, d.alloc, d.fill, d.free)
		bad = append(bad, d.bad...)
		bad = append(bad, wantJoined(tt.name, err, tt.want...)...)
		if len(tt.want) == 0 && err != nil {
			bad = append(bad, fmt.Sprintf("%s: %v", tt.name, err))
		}

		// What the calls should have been, up to the failure. A tied
		// lm_head takes no id.
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
			vecs, lens = append(vecs, nmat+k+1), append(lens, k+1)
		}
		wantFills := vecs
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

		if len(tt.want) == 0 {
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
		if len(tt.want) > 0 {
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

// TestDeviceClosures checks that uploadWeight and uploadModel pass the
// real device operations, which no CPU test runs, to the functions tested
// with fakes: in particular that each allocation closure allocates exactly
// the length it is asked for. Must-fail controls edit the real source.
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
	for _, m := range []struct{ name, file, old, new string }{
		{"C1 f32 allocates n+1", "weights_bf16.go", "return cudamem.Alloc[float32](dev, n) }", "return cudamem.Alloc[float32](dev, n+1) }"},
		{"C2 bf16 allocates n+1", "weights_bf16.go", "return cudamem.Alloc[uint16](dev, n) }", "return cudamem.Alloc[uint16](dev, n+1) }"},
		{"C3 parameter renamed, length fixed", "weights_bf16.go", "func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) }", "func(m int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, len(host)) }"},
		{"C4 vector allocates n+1", "gpu_model.go", "return cudamem.Alloc[float32](dev, n) }", "return cudamem.Alloc[float32](dev, n+1) }"},
		{"C5 matrices uploaded as f32", "gpu_model.go", "return uploadWeight(dev, host, dtype) }", "return uploadWeight(dev, host, weightF32) }"},
		{"C6 vector fill skipped", "gpu_model.go", "(*cudamem.Slice[float32]).CopyFromHost,", "func(*cudamem.Slice[float32], []float32) error { return nil },"},
		{"C7 vector free does not zero", "gpu_model.go", "\t\tfreeOwned[cudamem.Slice[float32]])\n}", "\t\tfunc(s *cudamem.Slice[float32]) error { return s.Free() })\n}"},
		{"C8 statement before the return", "gpu_model.go", "{ return cudamem.Alloc[float32](dev, n) }", "{ n++; return cudamem.Alloc[float32](dev, n) }"},
		{"C9 f32 closure size doubled", "weights_bf16.go", "func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) }", "func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, 2*n) }"},
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
