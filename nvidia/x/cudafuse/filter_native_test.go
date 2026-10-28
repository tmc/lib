package cudafuse

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

const (
	filterNativeEnvVar     = "CUDAFUSE_FILTER_NATIVE"
	filterNativeTargetVar  = "CUDAFUSE_FILTER_TARGET"
	filterNativeDefaultSM  = "sm_121a"
	filterNativePoisonBits = uint32(0x7fc01234)
	filterInputGuardLow    = float32(-9876.5)
	filterInputGuardHigh   = float32(9876.5)
)

var (
	errFilterNativeOptIn  = errors.New("cudafuse: native filter qualification pending: CUDAFUSE_FILTER_NATIVE=1 required")
	errFilterNativePoison = errors.New("cudafuse: filter output contains unwritten poison or nonfinite value")
)

type filterNativeGate struct {
	Owner  cudalock.Owner
	Target string
}

// filterNativeAdmission validates explicit opt-in, target support, and the
// inherited gpu-lock before any CUDA driver or NVRTC initialization occurs.
func filterNativeAdmission(getenv func(string) string) (filterNativeGate, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if getenv(filterNativeEnvVar) != "1" {
		return filterNativeGate{}, errFilterNativeOptIn
	}
	target := strings.TrimSpace(getenv(filterNativeTargetVar))
	if target == "" {
		target = filterNativeDefaultSM
	}
	if !isKnownTarget(target) {
		return filterNativeGate{}, fmt.Errorf("cudafuse: filter native target %q: %w", target, cudaerrors.ErrUnsupported)
	}
	if getenv("GPU_LOCK_HELD") == "" {
		return filterNativeGate{}, errors.New("cudafuse: native filter gate enabled without GPU_LOCK_HELD: run under gpu-lock")
	}
	owner, err := cudalock.Inherited(getenv)
	if err != nil {
		return filterNativeGate{}, fmt.Errorf("cudafuse: native filter gate requires gpu-lock: %w", err)
	}
	return filterNativeGate{Owner: owner, Target: target}, nil
}

type filterNativeCase struct {
	Name      string
	Length    int
	Mode      string // "none", "all", "mixed"
	Consumer  string // "filter", "pointwise", "sum", "scan"
	DataA     []float32
	DataB     []float32
	Predicate []float32
	Scalar    float32
}

type filterNativeBuilt struct {
	Case        filterNativeCase
	Plan        *MaterializedPlan
	NodeA       *LazyNode
	NodeB       *LazyNode
	NodeP       *LazyNode
	NodeS       *LazyNode
	Want        []float32
	WantShape   []int
	Selected    int
	InputDigest string
}

func filterNativeCases() []filterNativeCase {
	type spec struct {
		name   string
		length int
		mode   string
	}
	specs := []spec{
		{name: "none-33", length: 33, mode: "none"},
		{name: "all-33", length: 33, mode: "all"},
		{name: "mixed-signed-zero-33", length: 33, mode: "mixed"},
		{name: "mixed-tail-259", length: 259, mode: "mixed"},
	}
	consumers := []string{"filter", "pointwise", "sum", "scan"}
	var out []filterNativeCase
	for _, s := range specs {
		a, b, p := filterNativeInputs(s.length, s.mode)
		for _, c := range consumers {
			out = append(out, filterNativeCase{
				Name:      fmt.Sprintf("%s/%s", s.name, c),
				Length:    s.length,
				Mode:      s.mode,
				Consumer:  c,
				DataA:     slices.Clone(a),
				DataB:     slices.Clone(b),
				Predicate: slices.Clone(p),
				Scalar:    7.5,
			})
		}
	}
	return out
}

func filterNativeInputs(n int, mode string) (a, b, predicate []float32) {
	a = make([]float32, n)
	b = make([]float32, n)
	predicate = make([]float32, n)
	negZero := float32(math.Copysign(0, -1))
	for i := 0; i < n; i++ {
		a[i] = float32((i*13)%53-26) / 4
		b[i] = float32((i*7)%37-18) / 8
		switch mode {
		case "none":
			if i%2 == 1 {
				predicate[i] = negZero
			} else {
				predicate[i] = 0
			}
		case "all":
			predicate[i] = 1
		case "mixed":
			switch {
			case i == 0 || i == n-1 || i%3 == 0 || i%7 == 2:
				predicate[i] = 1
			case i%2 == 0:
				predicate[i] = negZero
			default:
				predicate[i] = 0
			}
		default:
			panic(fmt.Sprintf("unknown filter predicate mode %q", mode))
		}
	}
	return a, b, predicate
}

// evaluateFilterNativeOracle computes the dense reference on the CPU without
// calling any cudafuse kernel generator or prefix/count helper.
func evaluateFilterNativeOracle(tc filterNativeCase) (want []float32, wantShape []int, selected int, err error) {
	if len(tc.DataA) != tc.Length || len(tc.DataB) != tc.Length || len(tc.Predicate) != tc.Length {
		return nil, nil, 0, fmt.Errorf("cudafuse: mismatched filter fixture input lengths")
	}
	if tc.Length < 0 || tc.Length > maxFilterElements {
		return nil, nil, 0, fmt.Errorf("cudafuse: filter fixture length %d outside domain", tc.Length)
	}
	compacted := make([]float32, 0, tc.Length)
	for i := 0; i < tc.Length; i++ {
		p := tc.Predicate[i]
		if math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) || (p != 0 && p != 1) {
			return nil, nil, 0, fmt.Errorf("cudafuse: invalid filter predicate at %d: %v", i, p)
		}
		// p == -(-p) preserves IEEE-754 sign; both +0.0 and -0.0 compare equal to 0.0 and must be false.
		evalPred := -(-p)
		if evalPred == 1 {
			v := (tc.DataA[i] + tc.DataB[i]) * tc.DataA[i]
			compacted = append(compacted, v)
		}
	}
	selected = len(compacted)
	switch tc.Consumer {
	case "filter":
		return compacted, []int{selected}, selected, nil
	case "pointwise":
		out := make([]float32, selected)
		for i, v := range compacted {
			out[i] = -(v + tc.Scalar)
		}
		return out, []int{selected}, selected, nil
	case "sum":
		var acc float64
		for _, v := range compacted {
			acc += float64(v)
		}
		return []float32{float32(acc)}, []int{1}, selected, nil
	case "scan":
		out := make([]float32, selected)
		for i, v := range compacted {
			if i == 0 {
				out[i] = v
			} else {
				out[i] = out[i-1] + v
			}
		}
		return out, []int{selected}, selected, nil
	default:
		return nil, nil, 0, fmt.Errorf("cudafuse: unsupported filter fixture consumer %q", tc.Consumer)
	}
}

func buildFilterNativeFixture(tc filterNativeCase) (*filterNativeBuilt, error) {
	want, wantShape, selected, err := evaluateFilterNativeOracle(tc)
	if err != nil {
		return nil, err
	}
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{tc.Length}, cudair.Float32)
	b := NewLazyArray(ctx, []int{tc.Length}, cudair.Float32)
	p := NewLazyArray(ctx, []int{tc.Length}, cudair.Float32)
	filtered := a.Add(b).Mul(a).Filter(p.Map("neg").Map("neg"))

	var plan *MaterializedPlan
	var nodeS *LazyNode
	switch tc.Consumer {
	case "filter":
		plan, err = filtered.Materialize()
	case "pointwise":
		s := NewLazyArray(ctx, []int{1}, cudair.Float32)
		nodeS = s.Node
		plan, err = filtered.Add(s).Map("neg").Materialize()
	case "sum":
		plan, err = filtered.Sum()
	case "scan":
		plan, err = filtered.Scan("sum")
	default:
		return nil, fmt.Errorf("cudafuse: unknown filter fixture consumer %q", tc.Consumer)
	}
	if err != nil {
		return nil, err
	}
	if tc.Consumer != "sum" {
		if !unresolvedShape(plan.RootNode.Shape) || plan.AllocatedBytes != -1 {
			return nil, fmt.Errorf("cudafuse: filter plan resolved dynamic shape prematurely: shape=%v bytes=%d", plan.RootNode.Shape, plan.AllocatedBytes)
		}
	} else if !slices.Equal(plan.RootNode.Shape, []int{1}) {
		return nil, fmt.Errorf("cudafuse: filter sum plan shape=%v, want [1]", plan.RootNode.Shape)
	}

	h := sha256.New()
	writeSlice := func(xs []float32) {
		for _, x := range xs {
			var bits [4]byte
			binary.LittleEndian.PutUint32(bits[:], math.Float32bits(x))
			h.Write(bits[:])
		}
	}
	writeSlice(tc.DataA)
	writeSlice(tc.DataB)
	writeSlice(tc.Predicate)
	if nodeS != nil {
		writeSlice([]float32{tc.Scalar})
	}
	return &filterNativeBuilt{
		Case:        tc,
		Plan:        plan,
		NodeA:       a.Node,
		NodeB:       b.Node,
		NodeP:       p.Node,
		NodeS:       nodeS,
		Want:        want,
		WantShape:   wantShape,
		Selected:    selected,
		InputDigest: fmt.Sprintf("%x", h.Sum(nil)),
	}, nil
}

func (b *filterNativeBuilt) outputPolicy(target, sourceDigest string) cudatest.CUDAOutputPolicy {
	if target == "" {
		target = "host-fixture"
	}
	if sourceDigest == "" {
		sourceDigest = strings.Repeat("0", 64)
	}
	return cudatest.CUDAOutputPolicy{
		Target:       target,
		DType:        "fp32",
		MathMode:     "--fmad=false; no fast math",
		SourceDigest: sourceDigest,
		InputDigest:  b.InputDigest,
		Elements:     len(b.Want),
		NaN:          "reject",
		Infinity:     "reject",
		Zero:         "equal",
	}
}

func checkFilterFiniteAndUnpoisoned(values []float32) error {
	for i, v := range values {
		bits := math.Float32bits(v)
		if bits == filterNativePoisonBits {
			return fmt.Errorf("element %d retains unwritten poison: %w", i, ErrOutputUnwritten)
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("element %d is nonfinite (%v): %w", i, v, errFilterNativePoison)
		}
	}
	return nil
}

// compareFilterNativeResult validates the resolved shape, selected extent,
// finite/poison-free elements, and exact numerical match against the CPU oracle.
func compareFilterNativeResult(b *filterNativeBuilt, got []float32, gotShape []int, target, sourceDigest string) error {
	if b == nil {
		return errors.New("cudafuse: nil filter native fixture")
	}
	if !slices.Equal(gotShape, b.WantShape) {
		return fmt.Errorf("cudafuse: filter output shape = %v, want %v", gotShape, b.WantShape)
	}
	if len(got) != len(b.Want) {
		return fmt.Errorf("cudafuse: filter output length = %d, want %d", len(got), len(b.Want))
	}
	if err := checkFilterFiniteAndUnpoisoned(got); err != nil {
		return err
	}
	if err := checkFilterFiniteAndUnpoisoned(b.Want); err != nil {
		return err
	}
	if len(b.Want) == 0 {
		return nil
	}
	policy := b.outputPolicy(target, sourceDigest)
	if err := cudatest.CompareCUDAOutputFP32(got, b.Want, policy); err != nil {
		return err
	}
	return nil
}

func expectedFilterNativeStats(tc filterNativeCase, selected int, eager bool) ExecutionStats {
	if tc.Length == 0 {
		if tc.Consumer == "sum" {
			return ExecutionStats{Allocations: 1, Launches: 1}
		}
		return ExecutionStats{}
	}
	allocs := 4 // upstream/predicate count scratch: data, predicate, prefix, status_count
	launches := 1
	if eager {
		// In eager mode, upstream add(a,b), mul(...,a), neg(p), neg(neg(p)) run before filter.
		allocs += 4
		launches += 4
	}
	if selected > 0 {
		allocs++
		launches++
	}
	switch tc.Consumer {
	case "filter":
	case "pointwise":
		if selected > 0 {
			if eager {
				allocs += 2
				launches += 2
			} else {
				allocs++
				launches++
			}
		}
	case "sum":
		allocs++
		launches++
	case "scan":
		if selected > 0 {
			allocs++
			launches++
		}
	}
	return ExecutionStats{Allocations: allocs, Launches: launches}
}

func verifyFilterNativeStats(tc filterNativeCase, selected int, eager bool, stats ExecutionStats, receipts []KernelReceipt, closed bool) error {
	want := expectedFilterNativeStats(tc, selected, eager)
	if closed {
		want.Frees = want.Allocations
	}
	if stats != want {
		return fmt.Errorf("cudafuse: filter stats = %+v, want %+v", stats, want)
	}
	if len(receipts) != want.Launches {
		return fmt.Errorf("cudafuse: filter receipts = %d, want %d", len(receipts), want.Launches)
	}
	for i, r := range receipts {
		if r.Name == "" || len(r.SourceDigest) != 64 || len(r.PTXDigest) != 64 || len(r.CompilerKey) != 64 || r.Target == "" {
			return fmt.Errorf("cudafuse: incomplete filter kernel receipt %d: %+v", i, r)
		}
	}
	return nil
}

type filterNativeDeviceInputs struct {
	bindings map[string]ExecutionBinding
	owners   []cudamem.Slice[float32]
	expected [][]float32
}

func allocFilterNativeDeviceInputs(t *testing.T, dev *cuda.Device, b *filterNativeBuilt) filterNativeDeviceInputs {
	t.Helper()
	type named struct {
		node   *LazyNode
		values []float32
	}
	items := []named{
		{b.NodeA, b.Case.DataA},
		{b.NodeB, b.Case.DataB},
		{b.NodeP, b.Case.Predicate},
	}
	if b.NodeS != nil {
		items = append(items, named{b.NodeS, []float32{b.Case.Scalar}})
	}
	out := filterNativeDeviceInputs{bindings: make(map[string]ExecutionBinding, len(items))}
	for _, item := range items {
		n := len(item.values)
		storage, err := cudamem.Alloc[float32](dev, n+2)
		if storage.Ptr() != 0 {
			t.Cleanup(func() {
				if err := storage.Free(); err != nil {
					t.Error(err)
				}
			})
		}
		if err != nil {
			t.Fatal(err)
		}
		host := make([]float32, n+2)
		host[0] = filterInputGuardLow
		host[n+1] = filterInputGuardHigh
		copy(host[1:n+1], item.values)
		if err := storage.CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
		out.owners = append(out.owners, storage)
		out.expected = append(out.expected, host)
		out.bindings[item.node.Name()] = ExecutionBinding{
			Buffer: storage.Slice(1, n+1),
			Shape:  slices.Clone(item.node.Shape),
			DType:  cudair.Float32,
		}
	}
	return out
}

func (in filterNativeDeviceInputs) verifyUnmodified(t *testing.T) {
	t.Helper()
	for i, storage := range in.owners {
		got := make([]float32, storage.Len())
		if err := storage.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		for j, v := range got {
			if math.Float32bits(v) != math.Float32bits(in.expected[i][j]) {
				t.Fatalf("input storage %d modified at %d: got %v, want %v", i, j, v, in.expected[i][j])
			}
		}
	}
}

func retainFilterNativePacket(t *testing.T, e *Execution, out, want []float32, policy cudatest.CUDAOutputPolicy) {
	t.Helper()
	t.Logf("filter native calls: %+v; shape: %v; receipts: %+v; policy: %+v", e.Stats(), e.Shape(), e.Receipts(), policy)
	if dir := os.Getenv("CUDAFUSE_FILTER_ARTIFACT_DIR"); dir != "" {
		if !filepath.IsAbs(dir) {
			t.Fatal("filter artifact directory must be absolute")
		}
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(dir, strings.ReplaceAll(t.Name(), "/", "-")+".json")
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		encErr := json.NewEncoder(file).Encode(struct {
			Shape            []int
			Stats            ExecutionStats
			Receipts         []KernelReceipt
			Policy           cudatest.CUDAOutputPolicy
			Actual, Expected []float32
		}{e.Shape(), e.Stats(), e.Receipts(), policy, out, want})
		closeErr := file.Close()
		if encErr != nil || closeErr != nil {
			t.Fatal(errors.Join(encErr, closeErr))
		}
	}
}

func requireFilterNativeStream(t *testing.T) (*cuda.Device, *cuda.Stream, filterNativeGate) {
	t.Helper()
	gate, err := filterNativeAdmission(os.Getenv)
	if errors.Is(err, errFilterNativeOptIn) {
		t.Skip(err.Error())
	}
	if err != nil {
		t.Fatal(err)
	}
	dev := cudatest.RequireDevice(t)
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Logf("filter native gate admitted: lock_pid=%d label=%q target=%s device_sm=%d.%d", gate.Owner.PID, gate.Owner.Label, gate.Target, major, minor)
	return dev, stream, gate
}

func TestNativeFilterQualification(t *testing.T) {
	dev, stream, _ := requireFilterNativeStream(t)
	for _, tc := range filterNativeCases() {
		for _, eager := range []bool{false, true} {
			if eager && tc.Length != 33 {
				continue
			}
			t.Run(fmt.Sprintf("%s/eager=%v", tc.Name, eager), func(t *testing.T) {
				b, err := buildFilterNativeFixture(tc)
				if err != nil {
					t.Fatal(err)
				}
				in := allocFilterNativeDeviceInputs(t, dev, b)
				var e *Execution
				if eager {
					e, err = b.Plan.ExecuteEager(context.Background(), stream, in.bindings)
				} else {
					e, err = b.Plan.Execute(context.Background(), stream, in.bindings)
				}
				if e != nil {
					t.Cleanup(func() {
						if err := e.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				if err != nil {
					t.Fatal(err)
				}
				out, err := e.Readback(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				receipts := e.Receipts()
				hash := sha256.New()
				target := ""
				for _, r := range receipts {
					hash.Write([]byte(r.SourceDigest))
					if target == "" {
						target = r.Target
					}
				}
				sourceDigest := fmt.Sprintf("%x", hash.Sum(nil))
				if err := compareFilterNativeResult(b, out, e.Shape(), target, sourceDigest); err != nil {
					t.Fatal(err)
				}
				if err := verifyFilterNativeStats(tc, b.Selected, eager, e.Stats(), receipts, false); err != nil {
					t.Fatal(err)
				}
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
				if err := verifyFilterNativeStats(tc, b.Selected, eager, e.Stats(), receipts, true); err != nil {
					t.Fatal(err)
				}
				in.verifyUnmodified(t)
				retainFilterNativePacket(t, e, out, b.Want, b.outputPolicy(target, sourceDigest))
			})
		}
	}
}

// filterCountMutationEvidence records completed device reads from the count kernel.
// A compiler or runtime error alone cannot establish semantic discrimination.
type filterCountMutationEvidence struct {
	compiled, launched bool
	reads              [3][]float32
	allocations        int
}

func (e *filterCountMutationEvidence) observe(ops executionOps) executionOps {
	alloc := ops.alloc
	ops.alloc = func(n int) (*executionBuffer, error) {
		b, err := alloc(n)
		i := e.allocations
		e.allocations++
		if b != nil && i >= 1 && i <= 3 {
			read := b.read
			b.read = func(out []float32) error {
				if err := read(out); err != nil {
					return err
				}
				e.reads[i-1] = slices.Clone(out)
				return nil
			}
		}
		return b, err
	}
	compile := ops.compile
	ops.compile = func(source FusedKernel) (*executionKernel, error) {
		k, err := compile(source)
		if err == nil && k != nil && strings.Contains(source.Source, "float filter_selected =") && strings.Contains(source.Source, "== 0.0f ? 1.0f : 0.0f;") {
			e.compiled = true
			launch := k.launch
			k.launch = func(ptrs []driver.DevicePtr, n int) error {
				if err := launch(ptrs, n); err != nil {
					return err
				}
				e.launched = true
				return nil
			}
		}
		return k, err
	}
	return ops
}

func (e *filterCountMutationEvidence) verify(owner *Execution, execErr error, predicate []float32) error {
	if !e.compiled || !e.launched || owner == nil || owner.Stats().Allocations != 4 || owner.Stats().Frees != 4 || owner.Stats().Launches != 1 || owner.Shape() != nil {
		return fmt.Errorf("inverted count lacks completed kernel evidence")
	}
	if !slices.Equal(e.reads[0], predicate) || len(e.reads[1]) != len(predicate) || len(e.reads[2]) != 2 {
		return fmt.Errorf("inverted count lacks numeric readback evidence")
	}
	count := 0
	for i, p := range predicate {
		if p != 0 && p != 1 {
			return fmt.Errorf("invalid control predicate")
		}
		if p == 0 {
			count++
		}
		if e.reads[1][i] != float32(count) {
			return fmt.Errorf("inverted prefix mismatch at %d", i)
		}
	}
	if e.reads[2][0] != 0 || e.reads[2][1] != float32(count) {
		return fmt.Errorf("inverted count metadata mismatch")
	}
	_, semanticErr := filterCount(e.reads[0], e.reads[1], e.reads[2])
	if semanticErr == nil || execErr == nil || execErr.Error() != semanticErr.Error() {
		return fmt.Errorf("inverted count error = %v, want numeric semantic refusal %v", execErr, semanticErr)
	}
	return nil
}

func TestNativeFilterInvalidPredicateAndOutputControls(t *testing.T) {
	dev, stream, _ := requireFilterNativeStream(t)
	base := filterNativeCase{
		Name:     "control-33",
		Length:   33,
		Mode:     "mixed",
		Consumer: "pointwise",
		Scalar:   7.5,
	}
	base.DataA, base.DataB, base.Predicate = filterNativeInputs(base.Length, base.Mode)

	for _, badPred := range []struct {
		name  string
		value float32
	}{
		{"fractional", 0.5},
		{"negative", -1.0},
		{"nan", float32(math.NaN())},
		{"inf", float32(math.Inf(1))},
	} {
		t.Run("invalid-predicate-"+badPred.name, func(t *testing.T) {
			b, err := buildFilterNativeFixture(base)
			if err != nil {
				t.Fatal(err)
			}
			b.Case.Predicate = slices.Clone(base.Predicate)
			b.Case.Predicate[base.Length/2] = badPred.value
			in := allocFilterNativeDeviceInputs(t, dev, b)
			e, err := b.Plan.Execute(context.Background(), stream, in.bindings)
			if e != nil {
				t.Cleanup(func() {
					if err := e.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if e == nil || err == nil {
				t.Fatalf("invalid predicate %s admitted: e=%v err=%v", badPred.name, e, err)
			}
			if e.Stats().Allocations != 4 || e.Stats().Launches != 1 || e.Shape() != nil {
				t.Fatalf("invalid predicate allocated dynamic output or resolved shape: stats=%+v shape=%v", e.Stats(), e.Shape())
			}
			if _, readErr := e.Readback(context.Background()); readErr == nil {
				t.Fatal("failed filter execution permitted Readback")
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if e.Stats().Frees != 4 {
				t.Fatalf("scratch allocations not freed on close: %+v", e.Stats())
			}
			in.verifyUnmodified(t)
		})
	}

	for _, control := range []string{"invert-predicate", "reverse-scatter-order", "omit-last-scatter", "write-past-scatter-guard"} {
		t.Run(control, func(t *testing.T) {
			b, err := buildFilterNativeFixture(base)
			if err != nil {
				t.Fatal(err)
			}
			in := allocFilterNativeDeviceInputs(t, dev, b)
			plan, err := snapshotExecution(b.Plan, false)
			if err != nil {
				t.Fatal(err)
			}
			var evidence filterCountMutationEvidence
			ops := evidence.observe(nativeExecutionOps(dev, stream))
			compile := ops.compile
			ops.compile = func(source FusedKernel) (*executionKernel, error) {
				old := source.Source
				switch control {
				case "invert-predicate":
					if strings.Contains(source.Source, "float filter_selected =") {
						source.Source = strings.Replace(source.Source, "== 1.0f ? 1.0f : 0.0f;", "== 0.0f ? 1.0f : 0.0f;", 1)
					}
				case "reverse-scatter-order":
					if strings.Contains(source.Source, "unsigned int destination =") {
						source.Source = strings.Replace(source.Source, "unsigned int destination = (unsigned int)"+source.Inputs[2]+"[idx] - 1u;", fmt.Sprintf("unsigned int destination = %du - (unsigned int)%s[idx];", b.Selected, source.Inputs[2]), 1)
					}
				case "omit-last-scatter":
					if strings.Contains(source.Source, "unsigned int destination =") {
						source.Source = strings.Replace(source.Source, fmt.Sprintf("if (destination < %du)", b.Selected), fmt.Sprintf("if (destination + 1u < %du)", b.Selected), 1)
					}
				case "write-past-scatter-guard":
					if strings.Contains(source.Source, "unsigned int destination =") {
						source.Source = strings.Replace(source.Source, fmt.Sprintf("if (destination < %du)", b.Selected), fmt.Sprintf("if (destination <= %du) out_%s[%du] = 1.0f;\n            if (destination < %du)", b.Selected, source.Outputs[0], b.Selected, b.Selected), 1)
					}
				}
				if strings.Contains(old, "unsigned int destination =") || (control == "invert-predicate" && strings.Contains(old, "float filter_selected =")) {
					if source.Source == old {
						return nil, fmt.Errorf("control %s did not mutate kernel source", control)
					}
				}
				return compile(source)
			}
			inputs := make(map[string]executionInput, len(in.bindings))
			for name, binding := range in.bindings {
				binding := binding
				inputs[name] = executionInput{
					description: executionBindingDescription{
						shape:  binding.Shape,
						dtype:  binding.DType,
						length: binding.Buffer.Len(),
						ptr:    uint64(binding.Buffer.Ptr()),
						bytes:  uint64(binding.Buffer.SizeBytes()),
						device: binding.Buffer.Device(),
					},
					borrow: func(d *cuda.Device) (driver.DevicePtr, func(), error) {
						release, err := binding.Buffer.Borrow(d)
						return binding.Buffer.Ptr(), release, err
					},
				}
			}
			e, execErr := executePlanInputs(context.Background(), plan, dev, inputs, ops)
			if e != nil {
				t.Cleanup(func() {
					if err := e.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			switch control {
			case "invert-predicate":
				if e != nil {
					if err := e.Close(); err != nil {
						t.Fatal(err)
					}
				}
				if err := evidence.verify(e, execErr, base.Predicate); err != nil {
					t.Fatal(err)
				}
			case "reverse-scatter-order":
				if execErr != nil {
					t.Fatal(execErr)
				}
				out, err := e.Readback(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				if err := compareFilterNativeResult(b, out, e.Shape(), "native-control", ""); err == nil {
					t.Fatal("reversed scatter order accepted by comparator")
				}
			case "omit-last-scatter":
				if !errors.Is(execErr, ErrOutputUnwritten) {
					t.Fatalf("omit-last-scatter error = %v, want ErrOutputUnwritten", execErr)
				}
			case "write-past-scatter-guard":
				if !errors.Is(execErr, ErrOutputGuard) {
					t.Fatalf("write-past-scatter-guard error = %v, want ErrOutputGuard", execErr)
				}
			}
			if e != nil {
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
				if e.Stats().Frees != e.Stats().Allocations {
					t.Fatalf("control cleanup leaked allocations: %+v", e.Stats())
				}
			}
			in.verifyUnmodified(t)
		})
	}
}
