package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

type parityDevice struct {
	t   *testing.T
	dev *cuda.Device
	gm  *GPUModules
}

func newParityDevice(t *testing.T) *parityDevice {
	dev := cudatest.RequireDevice(t)
	gm, err := CompileGPUModules(dev)
	if gm != nil {
		t.Cleanup(func() {
			if err := gm.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	return &parityDevice{t: t, dev: dev, gm: gm}
}

func (d *parityDevice) upload(host []float32) cudamem.Slice[float32] {
	d.t.Helper()
	// A nonzero slice returned with an error is ours; a zero one frees as
	// a no-op.
	s, err := cudamem.Alloc[float32](d.dev, len(host))
	d.t.Cleanup(func() {
		if err := s.Free(); err != nil {
			d.t.Error(err)
		}
	})
	if err != nil {
		d.t.Fatal(err)
	}
	if err := s.CopyFromHost(host); err != nil {
		d.t.Fatal(err)
	}
	return s
}

func (d *parityDevice) download(s cudamem.Slice[float32]) []float32 {
	d.t.Helper()
	host := make([]float32, s.Len())
	if err := s.CopyToHost(host); err != nil {
		d.t.Fatal(err)
	}
	return host
}

func TestFusedRMSNormQKVParity(t *testing.T) {
	d := newParityDevice(t)
	for _, tt := range qkvParityCases {
		for _, bias := range qkvBiasModes {
			t.Run(tt.name+"/bias "+bias, func(t *testing.T) {
				f := newQKVFixture(tt.in, tt.qDim, tt.kvDim, bias)
				dx, dg := d.upload(f.x), d.upload(f.g)
				dw := []cudamem.Slice[float32]{d.upload(f.wq), d.upload(f.wk), d.upload(f.wv)}
				ptr := func(b []float32) any {
					if b == nil {
						return nullDevicePtr
					}
					return d.upload(b)
				}
				dims := []int{tt.qDim, tt.kvDim, tt.kvDim}

				// Fused kernel.
				fused := []cudamem.Slice[float32]{d.upload(make([]float32, tt.qDim)), d.upload(make([]float32, tt.kvDim)), d.upload(make([]float32, tt.kvDim))}
				bq, bk, bv := ptr(f.bq), ptr(f.bk), ptr(f.bv)
				if err := d.dev.WithContext(func() error {
					return cudalaunch.Kernel(d.gm.fusedQKVFn).Grid(uint32(tt.qDim+2*tt.kvDim), 1, 1).Block(128, 1, 1).
						Launch(dx, dg, dw[0], dw[1], dw[2], bq, bk, bv, fused[0], fused[1], fused[2],
							int32(tt.in), int32(tt.qDim), int32(tt.kvDim), float32(qkvParityEps))
				}); err != nil {
					t.Fatal(err)
				}
				// Unfused path: RMSNorm then one LinearGEMV per projection.
				dn := d.upload(make([]float32, tt.in))
				if err := d.dev.WithContext(func() error {
					return cudalaunch.Kernel(d.gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
						Launch(dx, dg, dn, int32(tt.in), float32(qkvParityEps))
				}); err != nil {
					t.Fatal(err)
				}
				unfused := make([]cudamem.Slice[float32], 3)
				for i, b := range [][]float32{f.bq, f.bk, f.bv} {
					unfused[i] = d.upload(make([]float32, dims[i]))
					db := ptr(b)
					if err := d.dev.WithContext(func() error {
						return cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(dims[i]), 1, 1).Block(128, 1, 1).
							Launch(dn, dw[i], db, unfused[i], int32(tt.in), int32(dims[i]))
					}); err != nil {
						t.Fatal(err)
					}
				}

				names := []string{"q", "k", "v"}
				got := make([][]float32, 3)
				for i, want := range f.want {
					got[i] = d.download(fused[i])
					if j := parityMismatch(got[i], want, kernelTolerance); j >= 0 {
						t.Errorf("fused %s[%d]: got %g want %g", names[i], j, got[i][j], want[j])
					}
					u := d.download(unfused[i])
					if j := parityMismatch(u, want, kernelTolerance); j >= 0 {
						t.Errorf("unfused %s[%d]: got %g want %g", names[i], j, u[j], want[j])
					}
				}
				// The comparison must reject the defects of the previous fused kernel.
				for _, c := range f.controls {
					if parityMismatch(got[c.out], c.want, kernelTolerance) < 0 {
						t.Errorf("%s not detected", c.name)
					}
				}
			})
		}
	}
}

func TestFusedSwiGLUDownParity(t *testing.T) {
	d := newParityDevice(t)
	for _, tt := range swigluParityCases {
		t.Run(tt.name, func(t *testing.T) {
			f := newSwiGLUFixture(tt.hidden, tt.inter, swigluGateSalt)
			dg, du, dw := d.upload(f.gate), d.upload(f.up), d.upload(f.w)
			fused := d.upload(make([]float32, tt.hidden))
			if err := d.dev.WithContext(func() error {
				return cudalaunch.Kernel(d.gm.fusedSwiGLUFn).Grid(uint32(tt.hidden), 1, 1).Block(128, 1, 1).
					Launch(dg, du, dw, fused, int32(tt.hidden), int32(tt.inter))
			}); err != nil {
				t.Fatal(err)
			}
			da, unfused := d.upload(make([]float32, tt.inter)), d.upload(make([]float32, tt.hidden))
			if err := d.dev.WithContext(func() error {
				return cudalaunch.Kernel(d.gm.swigluFn).Grid(1, 1, 1).Block(256, 1, 1).
					Launch(dg, du, da, int32(tt.inter))
			}); err != nil {
				t.Fatal(err)
			}
			if err := d.dev.WithContext(func() error {
				return cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(tt.hidden), 1, 1).Block(128, 1, 1).
					Launch(da, dw, nullDevicePtr, unfused, int32(tt.inter), int32(tt.hidden))
			}); err != nil {
				t.Fatal(err)
			}
			got := d.download(fused)
			if i := parityMismatch(got, f.want, kernelTolerance); i >= 0 {
				t.Errorf("fused out[%d]: got %g want %g", i, got[i], f.want[i])
			}
			if u := d.download(unfused); parityMismatch(u, f.want, kernelTolerance) >= 0 {
				i := parityMismatch(u, f.want, kernelTolerance)
				t.Errorf("unfused out[%d]: got %g want %g", i, u[i], f.want[i])
			}
			for _, c := range f.controls {
				if parityMismatch(got, c.want, kernelTolerance) < 0 {
					t.Errorf("%s not detected", c.name)
				}
			}
		})
	}
}

func TestOddRowCopies(t *testing.T) {
	d := newParityDevice(t)
	sentinel := float32(-77)

	const hidden, vocab, token = 7, 5, 3
	table := parityValues(hidden*vocab, 1)
	dTable, dOut := d.upload(table), d.upload(make([]float32, hidden))
	dTok, err := cudamem.Alloc[int32](d.dev, 1)
	t.Cleanup(func() {
		if err := dTok.Free(); err != nil {
			t.Error(err)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := dTok.CopyFromHost([]int32{token}); err != nil {
		t.Fatal(err)
	}
	if err := d.dev.WithContext(func() error {
		return cudalaunch.Kernel(d.gm.embedFn).Grid(1, 1, 1).Block(256, 1, 1).
			Launch(dTable, dTok, dOut, int32(hidden))
	}); err != nil {
		t.Fatal(err)
	}
	got := d.download(dOut)
	for i, v := range table[token*hidden : (token+1)*hidden] {
		if math.Float32bits(got[i]) != math.Float32bits(v) {
			t.Fatalf("embedding[%d]: got %g want %g", i, got[i], v)
		}
	}

	const kvHeads, headDim, maxSeq, pos = 1, 6, 5, 3
	kvDim := kvHeads * headDim
	k, v := parityValues(kvDim, 3), parityValues(kvDim, 5)
	cache := make([]float32, maxSeq*kvDim)
	for i := range cache {
		cache[i] = sentinel
	}
	dk, dv, kc, vc := d.upload(k), d.upload(v), d.upload(cache), d.upload(cache)
	if err := d.dev.WithContext(func() error {
		return cudalaunch.Kernel(d.gm.storeKVFn).Grid(1, 1, 1).Block(128, 1, 1).
			Launch(dk, dv, kc, vc, int32(pos), int32(kvHeads), int32(headDim))
	}); err != nil {
		t.Fatal(err)
	}
	for name, pair := range map[string][2][]float32{"key": {d.download(kc), k}, "value": {d.download(vc), v}} {
		got, row := pair[0], pair[1]
		for i, x := range got {
			want := sentinel
			if i/kvDim == pos {
				want = row[i%kvDim]
			}
			if math.Float32bits(x) != math.Float32bits(want) {
				t.Fatalf("%s cache[%d]: got %g want %g", name, i, x, want)
			}
		}
	}
}

func TestGPUForwardParity(t *testing.T) {
	d := newParityDevice(t)
	major, minor, err := d.dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	attnCompiled, err := cudaattn.Compile(major, minor)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	attnMod, err := attnCompiled.Load(d.dev)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	t.Cleanup(func() {
		if err := attnMod.Close(); err != nil {
			t.Errorf("close attention module: %v", err)
		}
	})
	attnFn, err := attnMod.Function("CausalFlashAttentionDecode")
	if err != nil {
		t.Fatal(err)
	}
	sampleCompiled, err := cudasample.Compile(major, minor)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	sampleMod, err := sampleCompiled.Load(d.dev)
	if err != nil {
		t.Fatal(releaseErrorOwner(err))
	}
	t.Cleanup(func() {
		if err := sampleMod.Close(); err != nil {
			t.Errorf("close sampler module: %v", err)
		}
	})
	sampleFn, err := sampleMod.Function("ArgmaxLogits")
	if err != nil {
		t.Fatal(err)
	}

	odd := DefaultModelConfig()
	// Query width equals hidden: CPU Forward's output projection assumes it.
	odd.HiddenSize, odd.IntermediateSize, odd.NumAttentionHeads, odd.NumKeyValueHeads, odd.VocabSize = 18, 30, 3, 1, 50
	for _, tt := range []struct {
		name  string
		cfg   ModelConfig
		fused bool
	}{
		{"default", DefaultModelConfig(), false},
		{"default fused", DefaultModelConfig(), true},
		{"odd rows", odd, false},
		{"odd rows fused", odd, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.fused && (d.gm.fusedQKVFn == 0 || d.gm.fusedSwiGLUFn == 0) {
				t.Fatal("fused kernels missing from module; the fused path would silently fall back")
			}
			saveFlag, saveQualified := *fuseEGraphFlag, fusedKernelsQualified
			*fuseEGraphFlag, fusedKernelsQualified = tt.fused, tt.fused
			t.Cleanup(func() { *fuseEGraphFlag, fusedKernelsQualified = saveFlag, saveQualified })

			cfg := tt.cfg
			w := GenerateDeterministicWeights(&cfg)
			gw, err := UploadModelToGPU(d.dev, w, &cfg)
			if gw != nil {
				t.Cleanup(func() {
					if err := gw.Free(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			kv, err := cudakv.New(d.dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
			if kv != nil {
				t.Cleanup(func() {
					if err := kv.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			sp, err := NewGPUScratchpad(d.dev, &cfg)
			if sp != nil {
				t.Cleanup(func() {
					if err := sp.Free(); err != nil {
						t.Error(err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			hostKV := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, cfg.NumKeyValueHeads*cfg.HeadDim())

			tok := int32(7)
			for pos := 0; pos < 6; pos++ {
				want := Forward(w, &cfg, hostKV, tok, pos)
				var next int32
				if err := d.dev.WithContext(func() error {
					var err error
					next, err = GPUForwardPass(gw, &cfg, kv, sp, d.gm, attnFn, sampleFn, tok, pos)
					return err
				}); err != nil {
					t.Fatal(err)
				}
				got := d.download(sp.Logits)
				if i := parityMismatch(got, want, logitTolerance); i >= 0 {
					t.Fatalf("pos %d logit %d: got %g want %g", pos, i, got[i], want[i])
				}
				if hostNext := sampleArgmax(want); next != hostNext {
					t.Fatalf("pos %d: device token %d, host token %d", pos, next, hostNext)
				}
				tok = next
			}
		})
	}
}

func TestFuseEGraphRefusedUntilQualified(t *testing.T) {
	if !fusedKernelsQualified {
		t.Fatal("fusedKernelsQualified = false, want true")
	}
	saveFlag, saveQualified := *fuseEGraphFlag, fusedKernelsQualified
	defer func() { *fuseEGraphFlag, fusedKernelsQualified = saveFlag, saveQualified }()
	*fuseEGraphFlag, fusedKernelsQualified = true, false
	// The refusal precedes any device or argument use.
	_, err := GPUForwardPass(nil, nil, nil, nil, nil, 0, 0, 0, 0)
	if err == nil || !strings.Contains(err.Error(), "-fuse-egraph") {
		t.Fatalf("unqualified -fuse-egraph: got %v, want refusal", err)
	}

	*fuseEGraphFlag, fusedKernelsQualified = true, true
	var log []string
	w := fakeModel(&log, true, 0)
	cfg := DefaultModelConfig()
	cfg.NumHiddenLayers = 1
	if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, -1, 0, nil); err == nil || !strings.Contains(err.Error(), "outside vocabulary") {
		t.Fatalf("qualified -fuse-egraph token check: got %v, want outside vocabulary", err)
	}
	if _, err := enqueueGPUForwardPass(w, &cfg, nil, nil, nil, 0, 0, 0, -1, nil); err == nil || !strings.Contains(err.Error(), "outside the kv cache") {
		t.Fatalf("qualified -fuse-egraph pos check: got %v, want outside the kv cache", err)
	}
}

var nullDevicePtr = driver.DevicePtr(0)

// withContextBody returns the FuncLit argument of the innermost enclosing
// <recv>.WithContext(func() error { ... }) call on stack, or nil.
func withContextBody(stack []ast.Node) *ast.FuncLit {
	for i := len(stack) - 1; i >= 1; i-- {
		lit, ok := stack[i].(*ast.FuncLit)
		if !ok {
			continue
		}
		call, ok := stack[i-1].(*ast.CallExpr)
		if !ok || len(call.Args) != 1 || call.Args[0] != lit {
			continue
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if ok && sel.Sel.Name == "WithContext" {
			return lit
		}
	}
	return nil
}

// containsUploadCall reports whether root contains a call to upload or
// uploadBF16, which enter and exit WithContext and should happen outside the
// launch closure.
func containsUploadCall(root ast.Node) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch fn := call.Fun.(type) {
		case *ast.Ident:
			if fn.Name == "uploadBF16" {
				found = true
			}
		case *ast.SelectorExpr:
			if fn.Sel.Name == "upload" {
				found = true
			}
		}
		return !found
	})
	return found
}

// isEnclosingLaunchChain reports whether stack around a cudalaunch.Kernel(fn)
// call is a chained .Launch(...) call, and returns that Launch CallExpr.
func isEnclosingLaunchChain(stack []ast.Node, kernelCall *ast.CallExpr) *ast.CallExpr {
	cur := ast.Expr(kernelCall)
	for i := len(stack) - 1; i >= 0; i-- {
		switch p := stack[i].(type) {
		case *ast.SelectorExpr:
			if p.X != cur {
				return nil
			}
			if i == 0 {
				return nil
			}
			call, ok := stack[i-1].(*ast.CallExpr)
			if !ok || call.Fun != p {
				return nil
			}
			if p.Sel.Name == "Launch" {
				return call
			}
			cur = call
			i--
		default:
			return nil
		}
	}
	return nil
}

// uncontextualizedLaunches parses src and counts cudalaunch.Kernel(...).Launch
// and device GPUForwardPass calls, reporting any that are not enclosed in a
// <dev>.WithContext(func() error { ... }) closure or that call upload/uploadBF16
// inside the WithContext closure.
func uncontextualizedLaunches(name string, src []byte) (launches int, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, nil, err
	}
	var stack []ast.Node
	ast.Inspect(f, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		switch x := n.(type) {
		case *ast.SelectorExpr:
			pkg, ok := x.X.(*ast.Ident)
			if !ok || pkg.Name != "cudalaunch" || x.Sel.Name != "Kernel" {
				break
			}
			launches++
			var kcall *ast.CallExpr
			if len(stack) > 0 {
				kcall, _ = stack[len(stack)-1].(*ast.CallExpr)
			}
			if kcall == nil || kcall.Fun != x {
				bad = append(bad, fmt.Sprintf("%s: cudalaunch.Kernel referenced without calling it", fset.Position(x.Pos())))
				return false
			}
			lcall := isEnclosingLaunchChain(stack[:len(stack)-1], kcall)
			if lcall == nil {
				bad = append(bad, fmt.Sprintf("%s: cudalaunch.Kernel without chained .Launch", fset.Position(x.Pos())))
				return false
			}
			lit := withContextBody(stack)
			if lit == nil {
				bad = append(bad, fmt.Sprintf("%s: cudalaunch.Kernel.Launch outside WithContext", fset.Position(lcall.Pos())))
				return false
			}
			if containsUploadCall(lit) {
				bad = append(bad, fmt.Sprintf("%s: upload call inside WithContext launch closure", fset.Position(lcall.Pos())))
			}
			return false
		case *ast.Ident:
			if x.Name != "GPUForwardPass" {
				break
			}
			if len(stack) > 0 {
				if sel, ok := stack[len(stack)-1].(*ast.SelectorExpr); ok && sel.Sel == x {
					return false
				}
			}
			var call *ast.CallExpr
			if len(stack) > 0 {
				call, _ = stack[len(stack)-1].(*ast.CallExpr)
			}
			if call == nil || call.Fun != x {
				launches++
				bad = append(bad, fmt.Sprintf("%s: GPUForwardPass referenced without calling it", fset.Position(x.Pos())))
				return false
			}
			if len(call.Args) > 0 {
				if arg0, ok := call.Args[0].(*ast.Ident); ok && arg0.Name == "nil" {
					return false
				}
			}
			launches++
			lit := withContextBody(stack)
			if lit == nil {
				bad = append(bad, fmt.Sprintf("%s: GPUForwardPass outside WithContext", fset.Position(call.Pos())))
				return false
			}
			if containsUploadCall(lit) {
				bad = append(bad, fmt.Sprintf("%s: upload call inside WithContext launch closure", fset.Position(call.Pos())))
			}
			return false
		}
		stack = append(stack, n)
		return true
	})
	return launches, bad, nil
}

// TestCUDATestLaunchesInContext checks that every cudalaunch.Kernel.Launch
// and device GPUForwardPass call in *_cuda_test.go runs inside a
// WithContext closure, because cudamem and nvrtcx operations each restore the
// previous CUDA context (0) on return.
func TestCUDATestLaunchesInContext(t *testing.T) {
	paths, err := filepath.Glob("*_cuda_test.go")
	if err != nil {
		t.Fatal(err)
	}
	got := make(map[string]int)
	var bad []string
	for _, p := range paths {
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		n, b, err := uncontextualizedLaunches(p, src)
		if err != nil {
			t.Fatal(err)
		}
		got[p] = n
		bad = append(bad, b...)
	}
	want := map[string]int{
		"bf16_parity_cuda_test.go":        6,
		"context_experiment_cuda_test.go": 0,
		"fused_parity_cuda_test.go":       9,
		"graph_parity_cuda_test.go":       7,
	}
	for file, w := range want {
		if got[file] != w {
			t.Errorf("%s: %d launches found, want %d", file, got[file], w)
		}
	}
	for _, b := range bad {
		t.Error(b)
	}
}

// TestUncontextualizedLaunchesControls checks uncontextualizedLaunches on
// positive and must-fail forms, and on mutations of fused_parity_cuda_test.go
// and bf16_parity_cuda_test.go.
func TestUncontextualizedLaunchesControls(t *testing.T) {
	for _, tt := range []struct {
		name     string
		body     string
		launches int
		bad      bool
	}{
		{"positive control: Kernel.Launch in WithContext", `
		_ = d.dev.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(128, 1, 1).Launch(x)
		})`, 1, false},
		{"positive control: GPUForwardPass in WithContext", `
		_ = d.dev.WithContext(func() error {
			_, err := GPUForwardPass(gw, &cfg, kv, sp, gm, attnFn, sampleFn, tok, pos)
			return err
		})`, 1, false},
		{"positive control: nil GPUForwardPass refusal check", `
		_, _ = GPUForwardPass(nil, nil, nil, nil, nil, 0, 0, 0, 0)`, 0, false},
		{"Kernel.Launch outside WithContext", `
		_ = cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(128, 1, 1).Launch(x)`, 1, true},
		{"GPUForwardPass outside WithContext", `
		_, _ = GPUForwardPass(gw, &cfg, kv, sp, gm, attnFn, sampleFn, tok, pos)`, 1, true},
		{"upload inside WithContext closure", `
		_ = d.dev.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(128, 1, 1).Launch(d.upload(x))
		})`, 1, true},
		{"uploadBF16 inside WithContext closure", `
		_ = d.dev.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(1, 1, 1).Block(128, 1, 1).Launch(uploadBF16(d, x))
		})`, 1, true},
		{"cudalaunch.Kernel without Launch", `
		_ = d.dev.WithContext(func() error {
			_ = cudalaunch.Kernel(fn)
			return nil
		})`, 1, true},
		{"GPUForwardPass referenced as value", `
		f := GPUForwardPass
		_ = f`, 1, true},
	} {
		src := "package main\nfunc f() {\n" + tt.body + "\n}\n"
		n, bad, err := uncontextualizedLaunches("control_cuda_test.go", []byte(src))
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if n != tt.launches {
			t.Errorf("%s: launches = %d, want %d", tt.name, n, tt.launches)
		}
		if flagged := len(bad) > 0; flagged != tt.bad {
			t.Errorf("%s: flagged = %v, want %v (%v)", tt.name, flagged, tt.bad, bad)
		}
	}

	for _, file := range []string{"fused_parity_cuda_test.go", "bf16_parity_cuda_test.go"} {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		mutated := strings.Replace(string(src), ".WithContext(func() error {\n", ".WithoutContext(func() error {\n", 1)
		if mutated == string(src) {
			t.Fatalf("mutation did not apply to %s", file)
		}
		if _, bad, err := uncontextualizedLaunches(file, []byte(mutated)); err != nil || len(bad) == 0 {
			t.Errorf("%s with unwrapped launch: %v, %v; want flagged", file, bad, err)
		}
	}
}
