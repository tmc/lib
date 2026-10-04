package cudafuse

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestAttentionResidual_SourcePinsIndependentOfPackageDirectory(t *testing.T) {
	root := t.TempDir()
	name := "nvidia/x/cudafuse/attention_residual.go"
	source := filepath.Join(root, "source", filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(source), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(source, []byte("sealed source"), 0600); err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(root, "source-manifest.json")
	for _, cwd := range []string{root, filepath.Join(root, "source", "nvidia"), filepath.Dir(source)} {
		t.Chdir(cwd)
		got, err := attentionResidualSourceBytes(manifest, name)
		if err != nil || string(got) != "sealed source" {
			t.Fatalf("cwd %s: source=%q err=%v", cwd, got, err)
		}
	}
	for _, bad := range []string{"../" + name, "/" + name, "nvidia/x/cudafuse/../cudafuse/attention_residual.go", "nvidia/x/cudafuse/other.go"} {
		if _, err := attentionResidualSourceBytes(manifest, bad); err == nil {
			t.Fatalf("unsafe source accepted: %s", bad)
		}
	}
	if _, err := attentionResidualSourceBytes("source-manifest.json", name); err == nil {
		t.Fatal("relative manifest accepted")
	}
	if err := os.Remove(source); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(root, "outside.go"), source); err != nil {
		t.Fatal(err)
	}
	if _, err := attentionResidualSourceBytes(manifest, name); err == nil {
		t.Fatal("symlink source accepted")
	}
}

func TestAttentionResidual_FailedPackedCleanupRetainsRetry(t *testing.T) {
	for _, name := range []string{"nil", "closed", "partial-allocation"} {
		t.Run(name, func(t *testing.T) {
			var e *AttentionResidualExecution
			if name == "closed" {
				e = &AttentionResidualExecution{closed: true}
			}
			cause := errors.New("operation failed")
			failure := errors.New("free failed")
			calls := 0
			free := func() error {
				calls++
				if calls < 3 {
					return failure
				}
				return nil
			}
			e, err := attentionResidualCleanup(e, cause, free)
			if e == nil || !errors.Is(err, cause) || !errors.Is(err, failure) {
				t.Fatal("cleanup owner or original cause lost", e, err)
			}
			if err := e.Close(); !errors.Is(err, failure) || e.closed {
				t.Fatal("failed retry closed owner", err)
			}
			if err := e.Close(); err != nil || !e.closed || calls != 3 {
				t.Fatal("successful retry did not release owner", err, calls)
			}
			if err := e.Close(); err != nil || calls != 3 {
				t.Fatal("cleanup repeated", err, calls)
			}
		})
	}
}

func TestAttentionResidual_OrderedCleanupFailureAndRetry(t *testing.T) {
	for _, failed := range []string{"drain", "stream"} {
		t.Run(failed, func(t *testing.T) {
			var calls []string
			fail := true
			s := &attentionResidualCleanupStack{}
			for _, name := range []string{"device", "stream", "backing", "drain"} {
				s.add(func() error {
					calls = append(calls, name)
					if name == failed && fail {
						return errors.New("retained for retry")
					}
					return nil
				})
			}
			if err := s.close(); err == nil {
				t.Fatal("failed owner forgotten")
			}
			want := []string{"drain"}
			if failed == "stream" {
				want = []string{"drain", "backing", "stream"}
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatal("dependent owner closed after failure", calls)
			}
			calls = nil
			fail = false
			if err := s.close(); err != nil {
				t.Fatal(err)
			}
			want = []string{"drain", "backing", "stream", "device"}
			if failed == "stream" {
				want = []string{"stream", "device"}
			}
			if !reflect.DeepEqual(calls, want) || len(s.steps) != 0 {
				t.Fatal("retry repeated completed teardown", calls)
			}
		})
	}
}

func TestAttentionResidual_PostExecuteAdoptionBeforeAssertion(t *testing.T) {
	assertion, drain := errors.New("post-execute assertion"), errors.New("drain")
	fail := true
	var calls []string
	outer, inner := &attentionResidualCleanupStack{}, &attentionResidualCleanupStack{}
	for _, name := range []string{"device", "stream"} {
		outer.add(func() error { calls = append(calls, name); return nil })
	}
	outer.add(inner.close)
	inner.add(func() error { calls = append(calls, "backing"); return nil })
	e := &AttentionResidualExecution{
		wait: func(context.Context) error {
			calls = append(calls, "execution")
			if fail {
				return drain
			}
			return nil
		},
		cleanup: []func() error{func() error { calls = append(calls, "scratch"); return nil }},
	}
	// The same adoption helper runs immediately after the native Execute.
	operation := func() error {
		inner.adopt(e != nil, e.Close)
		return assertion // No assertion/readback path may precede adoption.
	}
	if !errors.Is(operation(), assertion) || len(inner.steps) != 2 {
		t.Fatal("execution was not adopted exactly once")
	}
	if err := outer.close(); !errors.Is(err, drain) || e.closed || len(inner.steps) != 2 || len(outer.steps) != 3 || !reflect.DeepEqual(calls, []string{"execution"}) {
		t.Fatal("failed drain lost execution or crossed its owner boundary", err, calls)
	}
	fail, calls = false, nil
	if err := outer.close(); err != nil || !e.closed || !reflect.DeepEqual(calls, []string{"execution", "scratch", "backing", "stream", "device"}) {
		t.Fatal("retry did not drain execution before its dependencies", err, calls)
	}
	if err := outer.close(); err != nil || len(calls) != 5 {
		t.Fatal("completed callback repeated", err, calls)
	}
}

func TestAttentionResidual_TimerPartialConstructionAdoption(t *testing.T) {
	construction, closeFailure := errors.New("timer construction"), errors.New("timer close")
	fail := true
	var calls []string
	s := &attentionResidualCleanupStack{}
	for _, name := range []string{"device", "stream", "backing"} {
		s.add(func() error { calls = append(calls, name); return nil })
	}
	// A nonnil error result carries the actual owner's retry callback.
	construct := func() (func() error, error) {
		return func() error {
			calls = append(calls, "timer")
			if fail {
				return closeFailure
			}
			return nil
		}, construction
	}
	closeTimer, err := construct()
	s.adopt(closeTimer != nil, closeTimer)
	if !errors.Is(err, construction) || len(s.steps) != 4 {
		t.Fatal("partial timer error result was not adopted exactly once")
	}
	s.adopt(false, func() error { t.Fatal("nil timer adopted"); return nil })
	if err := s.close(); !errors.Is(err, closeFailure) || len(s.steps) != 4 || !reflect.DeepEqual(calls, []string{"timer"}) {
		t.Fatal("failed timer close crossed its owner boundary", err, calls)
	}
	fail, calls = false, nil
	if err := s.close(); err != nil || !reflect.DeepEqual(calls, []string{"timer", "backing", "stream", "device"}) {
		t.Fatal("timer retry lost owner or repeated completed teardown", err, calls)
	}
	if err := s.close(); err != nil || len(calls) != 4 {
		t.Fatal("completed timer callback repeated", err, calls)
	}
}

func TestAttentionResidual_SingleCandidateIdentity(t *testing.T) {
	for _, d := range []int{1, 31, 32, 33, 129} {
		c := AttentionResidualConfig{1, 2, d, 1e-5}
		v, g, w, _ := attentionResidualFixture(c)
		got := attentionResidualFP32(c, v, g, w)
		if !reflect.DeepEqual(got, v) {
			t.Fatal("sole raw candidate was normalized")
		}
	}
	c := AttentionResidualConfig{1, 1, 3, 1e-5}
	v := []float32{math.Float32frombits(0x80000000), math.Float32frombits(1), math.Float32frombits(0x80000001)}
	got := attentionResidualFP32(c, v, []float32{1, 1, 1}, []float32{1, 2, 3})
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(v[i]) {
			t.Fatal("raw signed zero or subnormal lost")
		}
	}
}

func TestAttentionResidual_EqualScoresMean(t *testing.T) {
	c := AttentionResidualConfig{3, 2, 33, 0.001}
	v, g, w, _ := attentionResidualFixture(c)
	clear(w)
	got := attentionResidualFP32(c, v, g, w)
	for row := 0; row < c.Rows; row++ {
		for d := 0; d < c.Width; d++ {
			var sum float64
			for k := 0; k < c.Candidates; k++ {
				sum += float64(v[(k*c.Rows+row)*c.Width+d])
			}
			if math.Abs(float64(got[row*c.Width+d])-sum/3) > 2e-6 {
				t.Fatal("equal scores did not mix raw mean")
			}
		}
	}
}

func TestAttentionResidual_PermutationEquivariance(t *testing.T) {
	c := AttentionResidualConfig{3, 2, 33, 1e-5}
	v, g, w, _ := attentionResidualFixture(c)
	want, _ := attentionResidualReference(c, v, g, w, 0)
	permuted := append([]float32(nil), v...)
	n := c.Rows * c.Width
	copy(permuted[:n], v[2*n:])
	copy(permuted[2*n:], v[:n])
	if err := attentionResidualCompare(attentionResidualFP32(c, permuted, g, w), want); err != nil {
		t.Fatal(err)
	}
}

func TestAttentionResidual_DenseFloat64Reference(t *testing.T) {
	for _, d := range []int{1, 31, 32, 33, 129} {
		c := AttentionResidualConfig{4, 2, d, 1e-5}
		v, g, w, _ := attentionResidualFixture(c)
		want, p := attentionResidualReference(c, v, g, w, 0)
		if err := attentionResidualCompare(attentionResidualFP32(c, v, g, w), want); err != nil {
			t.Fatal(err)
		}
		for r := 0; r < 2; r++ {
			var sum float64
			for k := 0; k < 4; k++ {
				sum += p[k*2+r]
			}
			if math.Abs(sum-1) > 1e-12 {
				t.Fatal("wrong candidate axis")
			}
		}
		if d == 33 {
			best := func(row int) int {
				k := 0
				for i := 1; i < 4; i++ {
					if p[i*2+row] > p[k*2+row] {
						k = i
					}
				}
				return k
			}
			if best(0) == best(1) {
				t.Fatal("fixture lacks different token rankings")
			}
		}
	}
}

func TestAttentionResidual_IrregularWidthAndGuards(t *testing.T) {
	for _, d := range []int{1, 31, 32, 33, 129} {
		c := AttentionResidualConfig{3, 2, d, 1e-6}
		v, g, w, _ := attentionResidualFixture(c)
		clear(v)
		got := attentionResidualFP32(c, v, g, w)
		for _, x := range got {
			if x != 0 {
				t.Fatal("zero candidates nonzero")
			}
		}
		for _, path := range []AttentionResidualPath{AttentionResidualComposed, AttentionResidualFused} {
			s, err := AttentionResidualSource(c, path)
			if err != nil || len(s) == 0 {
				t.Fatal(err)
			}
			if !strings.Contains(s[0].Source, "d<AR_D;d+=32") || !strings.Contains(s[0].Source, "blockDim.x==32") {
				t.Fatal("tail/resource contract missing")
			}
		}
	}
}

func TestAttentionResidual_Validation(t *testing.T) {
	good := AttentionResidualConfig{3, 2, 33, 1e-5}
	for _, c := range []AttentionResidualConfig{{0, 2, 33, 1e-5}, {9, 2, 33, 1e-5}, {3, 5, 33, 1e-5}, {3, 2, 130, 1e-5}, {math.MaxInt, 2, 33, 1e-5}, {3, 2, 33, 0}, {3, 2, 33, float32(math.Inf(1))}, {3, 2, 33, float32(math.NaN())}} {
		if c.validate() == nil {
			t.Fatalf("invalid config admitted %+v", c)
		}
	}
	rs := []attentionResidualRange{{0x1000, 3 * 2 * 33 * 4, 3 * 2 * 33}, {0x3000, 33 * 4, 33}, {0x4000, 33 * 4, 33}, {0x5000, 2 * 33 * 4, 2 * 33}}
	if err := attentionResidualRanges(good, rs); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func([]attentionResidualRange){func(r []attentionResidualRange) { r[3].ptr = r[0].ptr + 4 }, func(r []attentionResidualRange) { r[1].ptr++ }, func(r []attentionResidualRange) { r[2].ptr = math.MaxUint64 - 3 }, func(r []attentionResidualRange) { r[0].length-- }} {
		v := append([]attentionResidualRange(nil), rs...)
		mutate(v)
		if attentionResidualRanges(good, v) == nil {
			t.Fatal("range refusal missing")
		}
	}
	for _, s := range [][]int32{nil, {-1}, {1}, {2}, {3}, {0, -1}} {
		if attentionResidualStatus(s) == nil {
			t.Fatal("invalid status accepted")
		}
	}
	if attentionResidualStatus([]int32{0, 0}) != nil {
		t.Fatal("good status refused")
	}
}

func TestAttentionResidual_Mutants(t *testing.T) {
	c := AttentionResidualConfig{4, 2, 33, 1e-5}
	v, g, w, offset := attentionResidualFixture(c)
	for i := range g {
		if g[i] != 1+offset[i] {
			t.Fatal("fixture norm offsets not converted")
		}
	}
	want, _ := attentionResidualReference(c, v, g, w, 0)
	for m := 1; m <= 5; m++ {
		wrong, _ := attentionResidualReference(c, v, g, w, m)
		got := make([]float32, len(wrong))
		for i := range got {
			got[i] = float32(wrong[i])
		}
		if attentionResidualCompare(got, want) == nil {
			t.Fatalf("math mutant %d not observed", m)
		}
	}
}

func TestAttentionResidual_SnapshotCompletionAndAliasing(t *testing.T) {
	var drainFail = true
	var published, freed int
	e := &AttentionResidualExecution{wait: func(ctx context.Context) error {
		if ctx != nil && ctx.Err() != nil {
			return ctx.Err()
		}
		if drainFail {
			return errors.New("drain failed")
		}
		return nil
	}, inspect: func() error { return nil }, publish: func() error { published++; return nil }, cleanup: []func() error{func() error { freed++; return nil }}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if !errors.Is(e.Wait(ctx), context.Canceled) {
		t.Fatal("cancel not preserved")
	}
	if e.Close() == nil || published != 0 || freed != 0 {
		t.Fatal("failed drain released or published")
	}
	drainFail = false
	if err := e.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := e.Wait(nil); err != nil {
		t.Fatal(err)
	}
	if published != 1 {
		t.Fatal("duplicate publication")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if freed != 1 {
		t.Fatal("cleanup not once")
	}
	c := AttentionResidualConfig{3, 2, 33, 1e-5}
	v, g, w, _ := attentionResidualFixture(c)
	old := append([]float32(nil), v[:2*c.Rows*c.Width]...)
	first, _ := attentionResidualReference(c, v, g, w, 0)
	for i := 2 * c.Rows * c.Width; i < len(v); i++ {
		v[i] += 0.7
	}
	second, _ := attentionResidualReference(c, v, g, w, 0)
	if reflect.DeepEqual(first, second) || !reflect.DeepEqual(old, v[:len(old)]) {
		t.Fatal("partial update changed snapshot or did not affect result")
	}
}

func TestAttentionResidual_HeaderAndSourceBinding(t *testing.T) {
	h, err := attentionResidualHeader(cudablock.Header)
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(cudablock.Header, "// Warp-level sum reduction")
	end := strings.Index(cudablock.Header, "// Block-level cooperative 2D")
	if !strings.Contains(h, cudablock.Header[start:end]) {
		t.Fatal("collective bodies changed")
	}
	for _, bad := range []string{strings.Replace(cudablock.Header, "// Block-level cooperative 2D", "// missing", 1), cudablock.Header + "// Block-level cooperative 2D", strings.Replace(cudablock.Header, "#include <cuda_fp16.h>", "#include <extra.h>", 1)} {
		if _, err := attentionResidualHeader(bad); err == nil {
			t.Fatal("unexpected trim admitted")
		}
	}
	c := AttentionResidualConfig{3, 2, 33, 1e-5}
	sources, err := AttentionResidualSource(c, AttentionResidualFused)
	if err != nil {
		t.Fatal(err)
	}
	schema, _ := AttentionResidualSchema()
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("host-only synthetic image"), sources[0].Name, schema, "compute_121", cudakernel.WithCompilerOptions([]string{"--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"}), cudakernel.WithDependencies([]cudakernel.Dependency{{Path: "attention_residual.cu", Digest: attentionResidualDigest(sources[0].Source), Tracked: true}, {Path: "cudablock-owner.cuh", Digest: attentionResidualDigest(cudablock.Header), Tracked: true}, {Path: "cudablock-fp32.cuh", Digest: attentionResidualDigest(h), Tracked: true}}))
	if err != nil {
		t.Fatal(err)
	}
	owned, err := attentionResidualArtifact(a, sources[0])
	if err != nil {
		t.Fatal(err)
	}
	a.Raw[0] = 'x'
	a.Schema.Params[0].Name = "wrong"
	if owned.Raw[0] == 'x' || owned.Schema.Params[0].Name == "wrong" {
		t.Fatal("artifact not detached")
	}
	for _, mutate := range []func(*cudakernel.Artifact){func(a *cudakernel.Artifact) { a.Dependencies[2].Digest = attentionResidualDigest(h + "\n") }, func(a *cudakernel.Artifact) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) }, func(a *cudakernel.Artifact) { a.EntryPoint = "wrong" }, func(a *cudakernel.Artifact) { a.CompilerOpts = append(a.CompilerOpts, "--use_fast_math") }} {
		b := *owned
		b.Dependencies = append([]cudakernel.Dependency(nil), owned.Dependencies...)
		b.CompilerOpts = append([]string(nil), owned.CompilerOpts...)
		mutate(&b)
		if _, err := attentionResidualArtifact(&b, sources[0]); err == nil {
			t.Fatal("artifact mutation admitted")
		}
	}
}

func FuzzAttentionResidual_Extents(f *testing.F) {
	f.Add(3, 2, 33, float32(1e-5))
	f.Add(math.MaxInt, 1, 1, float32(1))
	f.Fuzz(func(t *testing.T, c, r, d int, epsilon float32) {
		v := AttentionResidualConfig{c, r, d, epsilon}
		if v.validate() == nil {
			if c*r*d > 8*4*129 || c*r*d <= 0 {
				t.Fatal("unchecked multiplication")
			}
			if _, err := AttentionResidualSource(v, AttentionResidualFused); err != nil {
				t.Fatal(err)
			}
		}
	})
}

func TestAttentionResidual_ActualPointerSlots(t *testing.T) {
	schema, err := AttentionResidualSchema()
	if err != nil {
		t.Fatal(err)
	}
	ptrs := []driver.DevicePtr{0x1004, 0x2004, 0x3004, 0x4004, 0x5004, 0x6004, 0x7004}
	args := attentionResidualArguments(ptrs)
	if len(args) != len(schema.Params) {
		t.Fatal("argument extent differs")
	}
	for i, a := range args {
		if schema.Params[i].ByteWidth != 8 || schema.Params[i].Alignment != 8 || uintptr(a)%8 != 0 || *(*uint64)(unsafe.Pointer(a)) != uint64(ptrs[i]) {
			t.Fatalf("pointer ABI slot %d differs", i)
		}
	}
}

func TestAttentionResidual_ZeroValuesAndDiagnosticPublication(t *testing.T) {
	var e AttentionResidualExecution
	if e.Wait(nil) == nil {
		t.Fatal("uninitialized execution admitted")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	published, freed := 0, 0
	e = AttentionResidualExecution{wait: func(context.Context) error { return nil }, inspect: func() error { return attentionResidualStatus([]int32{2}) }, publish: func() error { published++; return nil }, cleanup: []func() error{func() error { freed++; return nil }}}
	if e.Wait(nil) == nil || published != 0 {
		t.Fatal("overflow published")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if freed != 1 {
		t.Fatal("refused completed invocation leaked")
	}
}

func TestAttentionResidual_CancelAfterDrainAndAbandonedPublication(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	released, published := 0, 0
	held := true
	r := func() {
		if held {
			held = false
			released++
		}
	}
	e := &AttentionResidualExecution{wait: func(context.Context) error { cancel(); return nil }, inspect: func() error { return nil }, publish: func() error {
		if !held {
			t.Fatal("output lease lost before publication")
		}
		published++
		return nil
	}, finishRelease: r, cleanup: []func() error{func() error { r(); return nil }}}
	if !errors.Is(e.Wait(ctx), context.Canceled) || released != 0 || published != 0 {
		t.Fatal("cancel after drain released unpublished output")
	}
	if err := e.Wait(nil); err != nil {
		t.Fatal(err)
	}
	if released != 1 || published != 1 {
		t.Fatal("recovered publication differs")
	}
	if err := e.Close(); err != nil || released != 1 {
		t.Fatal("output released twice", err)
	}
	fail := true
	e = &AttentionResidualExecution{wait: func(context.Context) error { return nil }, inspect: func() error { return nil }, publish: func() error { t.Fatal("abandoned output published"); return nil }, cleanup: []func() error{func() error {
		if fail {
			return errors.New("cleanup failed")
		}
		return nil
	}}}
	if e.Close() == nil {
		t.Fatal("cleanup failure hidden")
	}
	if e.Wait(nil) == nil {
		t.Fatal("abandoned invocation resumed")
	}
	fail = false
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
}
