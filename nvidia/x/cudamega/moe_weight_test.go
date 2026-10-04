package cudamega

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestMoEWeightTypedRunner(t *testing.T) {
	for _, i := range []int{8, 16} {
		for _, h := range []int{17, 32} {
			c := MoEWeightConfig{3, 2, i, h}
			s, err := MoEWeightSource(c, "sm_121")
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"const float *input", "float *output", "if (owns_0 && !done_0)", "remaining--", "atomicAdd(completed + 0, 1u)", "start_order[1 +", "fence.sc.gpu", "__fadd_rn(gate,__fmul_rn", "scales[gate_row]", "gate/__fadd_rn(1.0f,expf(-gate))"} {
				if !strings.Contains(s, want) {
					t.Fatalf("missing task contract %q", want)
				}
			}
			if strings.Contains(s, "unsigned *input") || strings.Contains(s, "op_0(") || strings.Contains(s, "ordinary_task") {
				t.Fatal("scalar operation substituted")
			}
			ordinary, _, _ := cudamoe.MoEWeightSource(cudamoe.MoEWeightSlabConsumer)
			decode, _, _ := strings.Cut(ordinary, "extern \"C\"")
			if !strings.HasPrefix(s, decode) {
				t.Fatal("decode differs from qualified consumer")
			}
			p, err := moeWeightProgram(c, "sm_121")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := p.Source(); err == nil {
				t.Fatal("scalar renderer accepted typed operation")
			}
			steps := p.Steps()
			steps[0].Operation.moe.Hidden = 999
			if p.steps[0].Operation.moe.Hidden != h {
				t.Fatal("step metadata aliases caller")
			}
		}
	}
}

func TestMoEWeightActualBorrowMetadata(t *testing.T) {
	dev := &cuda.Device{}
	c := MoEWeightConfig{3, 2, 16, 17}
	b := cudamoe.MoEWeightBorrowed{Descriptor: cudamoe.MoEWeightDescriptor{Device: dev, WeightBase: 0x1000, ScaleBase: 0x10000, WeightOffsetBytes: 2720, ScaleOffsetBytes: 640, Layer: 2, Expert: 1, Intermediate: 16, Hidden: 17, Generation: 1, Format: "e4m3fn-gate-up-interleave8-v1"}, Shape: cudamoe.MoEWeightShape{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17, ActiveLayers: []int{0, 2}}, WeightBytes: 3264, ScaleBytes: 768}
	if err := checkMoEBorrow(c, dev, b); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		edit func(*cudamoe.MoEWeightBorrowed)
	}{
		{"shape", func(b *cudamoe.MoEWeightBorrowed) { b.Shape.Experts = 1 }},
		{"padded", func(b *cudamoe.MoEWeightBorrowed) { b.Shape.ActiveLayers = []int{0, 1} }},
		{"device", func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.Device = &cuda.Device{} }},
		{"generation", func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.Generation = 0 }},
		{"offset", func(b *cudamoe.MoEWeightBorrowed) { b.Descriptor.WeightOffsetBytes = 0 }},
		{"extent", func(b *cudamoe.MoEWeightBorrowed) { b.WeightBytes = 1 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v := b
			tt.edit(&v)
			if err := checkMoEBorrow(c, dev, v); err == nil {
				t.Fatal("invalid actual metadata admitted")
			}
		})
	}
}

func TestMoEWeightTypedRefusals(t *testing.T) {
	for _, c := range []MoEWeightConfig{{}, {2, 2, 16, 17}, {3, 1, 16, 17}, {3, 2, 7, 17}, {3, 2, 16, 18}} {
		if _, err := MoEWeightSource(c, "sm_121"); err == nil {
			t.Fatal("invalid shape admitted", c)
		}
	}
	if _, err := MoEWeightSource(MoEWeightConfig{3, 2, 16, 17}, SM121a); err == nil {
		t.Fatal("unqualified target admitted")
	}
	g, _ := cudataskgraph.New([]cudataskgraph.Task{{ID: "gate-up", Scope: cudataskgraph.ScopeDevice}})
	if _, err := NewProgram(g, map[string]Operation{"gate-up": {Kind: MoEGateUp}}, Grid{CTAs: 1, ThreadsPerCTA: 32}, "sm_121"); err == nil {
		t.Fatal("public scalar admission accepted typed payload")
	}
	c := MoEWeightConfig{3, 2, 16, 17}
	for _, grid := range []Grid{{CTAs: 2, ThreadsPerCTA: 32}, {CTAs: 1, ThreadsPerCTA: 64}} {
		if _, err := newProgram(g, map[string]Operation{"gate-up": {Kind: MoEGateUp, moe: &c}}, grid, "sm_121", true); err == nil {
			t.Fatal("unsupported geometry admitted")
		}
	}
	var zero MoEWeightConsumer
	if err := zero.Execute(context.Background(), nil, nil, MoEWeightBuffers{}, 1); err == nil {
		t.Fatal("zero consumer admitted")
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
}

func moeHostArtifact(t *testing.T) *cudakernel.Artifact {
	t.Helper()
	s, err := MoEWeightSource(MoEWeightConfig{3, 2, 16, 17}, "sm_121")
	if err != nil {
		t.Fatal(err)
	}
	schema, _ := MoEWeightSchema()
	// A synthetic ELF header exercises receipt admission only; it contains no
	// executable device code and is never passed to a CUDA driver.
	var ident [16]byte
	copy(ident[:], "\x7fELF")
	ident[4] = 2
	ident[5] = 1
	ident[6] = 1
	ident[7] = 0x41
	ident[8] = 8
	var raw bytes.Buffer
	if err := binary.Write(&raw, binary.LittleEndian, elf.Header64{Ident: ident, Type: 2, Machine: uint16(elf.EM_CUDA), Version: 1, Flags: 0x0a007902, Ehsize: 64}); err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw.Bytes(), "moe_weight_worker", schema, "sm_121", cudakernel.WithDependencies([]cudakernel.Dependency{{Path: "cudamega_moe_weight.cu", Tracked: true, Digest: fmt.Sprintf("%x", sha256.Sum256([]byte(s)))}}), cudakernel.WithCompilerOptions([]string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}))
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func TestMoEWeightArtifactIdentity(t *testing.T) {
	c := MoEWeightConfig{3, 2, 16, 17}
	a := moeHostArtifact(t)
	owned, err := checkedMoEArtifact(c, a)
	if err != nil {
		t.Fatal(err)
	}
	a.Raw[0] = 'x'
	a.Schema.Params[0].Name = "x"
	a.Dependencies[0].Digest = "x"
	a.CompilerOpts[0] = "x"
	if owned.Raw[0] == 'x' || owned.Schema.Params[0].Name == "x" || owned.Dependencies[0].Digest == "x" || owned.CompilerOpts[0] == "x" {
		t.Fatal("artifact aliases caller")
	}
	for _, tt := range []struct {
		name string
		edit func(*cudakernel.Artifact)
	}{
		{"entry", func(a *cudakernel.Artifact) { a.EntryPoint = "ordinary_task" }},
		{"kind", func(a *cudakernel.Artifact) { a.Kind = cudakernel.KindPTX }},
		{"target", func(a *cudakernel.Artifact) { a.Target = "sm_120" }},
		{"schema", func(a *cudakernel.Artifact) { a.Schema.Params[0].Name = "different" }},
		{"options", func(a *cudakernel.Artifact) { a.CompilerOpts = []string{"--use_fast_math"} }},
		{"missing", func(a *cudakernel.Artifact) { a.Dependencies = nil }},
		{"duplicate", func(a *cudakernel.Artifact) { a.Dependencies = append(a.Dependencies, a.Dependencies[0]) }},
		{"untracked", func(a *cudakernel.Artifact) { a.Dependencies[0].Tracked = false }},
		{"bytes", func(a *cudakernel.Artifact) { a.Raw[0] = 'x' }},
		{"static-resource", func(a *cudakernel.Artifact) { a.Resources.StaticSharedMemory = math.MaxUint32 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := moeHostArtifact(t)
			tt.edit(a)
			if _, err := checkedMoEArtifact(c, a); err == nil {
				t.Fatal("invalid identity admitted")
			}
		})
	}
	if _, err := checkedMoEArtifact(c, nil); err == nil {
		t.Fatal("nil admitted")
	}
	schema, _ := MoEWeightSchema()
	if len(schema.Params) != 11 || schema.TotalParamBytes != 80 {
		t.Fatalf("ABI differs %+v", schema)
	}
}

// This host model checks the consumer's cleanup state, not GPU completion.
type moeDrainModel struct {
	err      error
	released bool
	calls    int
}

func (m *moeDrainModel) Wait(ctx context.Context) error {
	m.calls++
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if m.err != nil {
		return m.err
	}
	m.released = true
	return nil
}
func TestMoEWeightConsumerDrainRetry(t *testing.T) {
	cause := errors.New("injected drain failure")
	m := &moeDrainModel{err: cause}
	c := &MoEWeightConsumer{pending: m}
	if !errors.Is(c.Submit(nil, nil, MoEWeightBuffers{}, 1), cudamem.ErrInUse) || m.calls != 0 {
		t.Fatal("pending submit reused resources or attempted drain")
	}
	if !errors.Is(c.Execute(context.Background(), nil, nil, MoEWeightBuffers{}, 1), cudamem.ErrInUse) || m.calls != 0 {
		t.Fatal("pending execute reused resources or attempted drain")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if !errors.Is(c.Wait(ctx), context.Canceled) || c.pending == nil || m.released {
		t.Fatal("canceled drain released ownership")
	}
	if !errors.Is(c.Close(), cause) || c.pending == nil || m.released {
		t.Fatal("failed close released ownership")
	}
	m.err = nil
	if err := c.Close(); err != nil || c.pending != nil || !m.released {
		t.Fatal("successful drain did not release", err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if m.calls != 3 {
		t.Fatal("extra drain after completion")
	}
	cause = errors.New("partial launch failed")
	m = &moeDrainModel{}
	c = &MoEWeightConsumer{pending: m, failure: cause}
	if !errors.Is(c.Wait(context.Background()), cause) || c.pending != nil || !m.released {
		t.Fatal("partial submission did not drain before returning cause")
	}
	if !reflect.DeepEqual(c.failure, nil) {
		t.Fatal("cause retained after successful drain")
	}
}

func TestMoEWeightSubmitAdmission(t *testing.T) {
	// These are the validators called by Submit before any host copy or launch.
	// Full owner slab ranges include unselected global layers and experts.
	ranges := []moeWeightRegion{
		{ptr: 0x1000, bytes: 68},    // input
		{ptr: 0x2000, bytes: 64},    // output
		{ptr: 0x3000, bytes: 4},     // completed
		{ptr: 0x4000, bytes: 4},     // abort
		{ptr: 0x5000, bytes: 16},    // start order
		{ptr: 0x10000, bytes: 3264}, // full weights
		{ptr: 0x20000, bytes: 768},  // full scales
	}
	if err := checkMoERanges(ranges); err != nil {
		t.Fatal("disjoint ranges refused", err)
	}
	for _, tt := range []struct {
		name  string
		index int
		ptr   driver.DevicePtr
		bytes uint64
	}{
		{"input-output", 1, 0x1004, 64},
		{"output-completed", 2, 0x2004, 4},
		{"control-control", 3, 0x3000, 4},
		{"input-unselected-weight-range", 0, 0x10010, 68},
		{"output-unselected-scale-range", 1, 0x20004, 64},
		{"overflow", 0, driver.DevicePtr(math.MaxUint64 - 31), 68},
		{"null", 0, 0, 68},
		{"empty", 0, 0x1000, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			rs := append([]moeWeightRegion(nil), ranges...)
			rs[tt.index].ptr, rs[tt.index].bytes = tt.ptr, tt.bytes
			if err := checkMoERanges(rs); err == nil {
				t.Fatal("invalid production range admitted")
			}
		})
	}
	if err := checkMoEInput([]float32{-1, 0, float32(math.Copysign(0, -1)), math.MaxFloat32}); err != nil {
		t.Fatal("finite input refused", err)
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if err := checkMoEInput([]float32{0, v, 1}); err == nil {
			t.Fatal("nonfinite production input admitted", v)
		}
	}
}

func TestMoEWeightMeasuredOccupancy(t *testing.T) {
	for _, blocks := range []int32{math.MinInt32, -1, 0} {
		if err := checkMoEOccupancy(blocks); err == nil {
			t.Fatal("empty measurement admitted", blocks)
		}
	}
	for _, blocks := range []int32{1, 4} {
		if err := checkMoEOccupancy(blocks); err != nil {
			t.Fatal("positive measurement refused", blocks, err)
		}
	}
}

func FuzzMoEWeightRanges(f *testing.F) {
	f.Add(uint64(4096), uint64(68), uint64(8192), uint64(64))
	f.Add(uint64(math.MaxUint64-1), uint64(8), uint64(8192), uint64(64))
	f.Add(uint64(4096), uint64(68), uint64(4100), uint64(64))
	f.Fuzz(func(t *testing.T, a, n, b, m uint64) {
		endA, carryA := bits.Add64(a, n, 0)
		endB, carryB := bits.Add64(b, m, 0)
		valid := a != 0 && b != 0 && n != 0 && m != 0 && carryA == 0 && carryB == 0
		if valid {
			valid = endA <= b || endB <= a
		}
		err := checkMoERanges([]moeWeightRegion{{ptr: driver.DevicePtr(a), bytes: n}, {ptr: driver.DevicePtr(b), bytes: m}})
		if (err == nil) != valid {
			t.Fatalf("range admission differs: a=%d n=%d b=%d m=%d err=%v", a, n, b, m, err)
		}
	})
}
