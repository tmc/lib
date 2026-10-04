//go:build native

package cudablock

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

type topkPacketEntry struct {
	Name, SourceSHA256, ArtifactSHA256, Target, CompilerPath string
	CompilerMajor, CompilerMinor                             int32
	CompilerOptions                                          []string
}

func topkDigest(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func topkWriteJSON(t *testing.T, path string, v any) {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}

// TestBuildTopKPacket compiles six fixed CUDA images without a device or launch.
// Run it outside gpu-lock with TOPK_BUILD_ONLY=1, TOPK_TARGET=sm_121 and a fresh
// scoped TOPK_PACKET_DIR. TestNativeTopKReference only loads this retained packet.
func TestBuildTopKPacket(t *testing.T) {
	if os.Getenv("TOPK_BUILD_ONLY") != "1" {
		t.Skip("offline packet build not requested")
	}
	dir, target := os.Getenv("TOPK_PACKET_DIR"), os.Getenv("TOPK_TARGET")
	if !filepath.IsAbs(dir) || target == "" {
		t.Fatal("absolute packet directory and explicit target required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := nvrtc.Load(""); err != nil {
		t.Skipf("NVRTC unavailable: %v", err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	sources := map[string]string{"baseline": TopKSource}
	for _, m := range topkMutations {
		s, err := topkMutated(m)
		if err != nil {
			t.Fatal(err)
		}
		sources[m.name] = s
	}
	for name, source := range sources {
		artifact, err := topkCompileArtifact(source, target)
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		entry := topkPacketEntry{name, topkDigest([]byte(source)), artifact.Digest, target, nvrtc.Path(), major, minor, artifact.CompilerOpts}
		topkWriteJSON(t, filepath.Join(dir, name+".artifact.json"), artifact)
		topkWriteJSON(t, filepath.Join(dir, name+".receipt.json"), entry)
		if err := os.WriteFile(filepath.Join(dir, name+".cu"), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("offline image %s source=%s image=%s target=%s nvrtc=%d.%d", name, entry.SourceSHA256, entry.ArtifactSHA256, target, major, minor)
	}
}

func topkReadPacket(t *testing.T, dir, name, source, target string) (*cudakernel.Artifact, topkPacketEntry) {
	t.Helper()
	read := func(suffix string, v any) {
		data, err := os.ReadFile(filepath.Join(dir, name+suffix))
		if err != nil {
			t.Fatal(err)
		}
		if len(data) > 16<<20 {
			t.Fatal("oversized topk packet")
		}
		if err := json.Unmarshal(data, v); err != nil {
			t.Fatal(err)
		}
	}
	var entry topkPacketEntry
	var a cudakernel.Artifact
	read(".receipt.json", &entry)
	read(".artifact.json", &a)
	s, err := os.ReadFile(filepath.Join(dir, name+".cu"))
	if err != nil {
		t.Fatal(err)
	}
	if string(s) != source || entry.Name != name || entry.SourceSHA256 != topkDigest([]byte(source)) || entry.ArtifactSHA256 != a.Digest || a.Target != target || entry.Target != target || a.EntryPoint != "topk_fp32" || a.Kind != cudakernel.KindCUBIN {
		t.Fatal("topk packet identity mismatch")
	}
	schema, err := TopKSchema()
	if err != nil {
		t.Fatal(err)
	}
	aSchema, _ := json.Marshal(a.Schema)
	wantSchema, _ := json.Marshal(schema)
	if string(aSchema) != string(wantSchema) {
		t.Fatal("topk packet schema mismatch")
	}
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
	return &a, entry
}

type topkNativeReceipt struct {
	Fixture                                         string
	Config                                          TopKConfig
	Image                                           topkPacketEntry
	InputBits                                       []uint32
	Lengths, Offsets                                []int32
	Indices, Counts, Status                         []int32
	ValueBits                                       []uint32
	ExpectedIndices, ExpectedCounts, ExpectedStatus []int32
	ExpectedValueBits                               []uint32
	ExecuteError                                    string
	Published, RedzonesIntact                       bool
	ValuePolicy                                     *cudatest.CUDAOutputPolicy
	ValueComparisonError                            string
	LoadedResources                                 map[string]int
}

func topkFloatBits(s []float32) []uint32 {
	b := make([]uint32, len(s))
	for i, v := range s {
		b[i] = math.Float32bits(v)
	}
	return b
}

func topkValuePolicy(f topkCase, entry topkPacketEntry) cudatest.CUDAOutputPolicy {
	var input bytes.Buffer
	// Fixed-width exact input bits and metadata bind the comparison receipt.
	for _, v := range []any{topkFloatBits(f.input), f.lengths, f.offsets} {
		_ = binary.Write(&input, binary.LittleEndian, v)
	}
	return cudatest.CUDAOutputPolicy{Target: entry.Target, DType: "fp32", MathMode: "ordered-comparisons-copy-no-fast-math", SourceDigest: entry.SourceSHA256, InputDigest: topkDigest(input.Bytes()), Elements: f.c.Rows * f.c.K, NaN: "reject", Infinity: "same-sign", Zero: "bits"}
}

func topkNativeFloat(t *testing.T, dev *cuda.Device, n int) (cudamem.Slice[float32], func()) {
	t.Helper()
	allocation, err := cudamem.Alloc[float32](dev, n+2)
	if err != nil {
		if allocation.Ptr() != 0 {
			_ = allocation.Free()
		}
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := allocation.Free(); err != nil {
			t.Error(err)
		}
	})
	guard := make([]float32, n+2)
	for i := range guard {
		guard[i] = math.Float32frombits(0x5a5a5a5a)
	}
	if err := allocation.CopyFromHost(guard); err != nil {
		t.Fatal(err)
	}
	check := func() {
		got := make([]float32, n+2)
		if err := allocation.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		if math.Float32bits(got[0]) != 0x5a5a5a5a || math.Float32bits(got[n+1]) != 0x5a5a5a5a {
			t.Fatal("float redzone damaged")
		}
	}
	return allocation.Slice(1, n+1), check
}

func topkNativeInt(t *testing.T, dev *cuda.Device, n int) (cudamem.Slice[int32], func()) {
	t.Helper()
	allocation, err := cudamem.Alloc[int32](dev, n+2)
	if err != nil {
		if allocation.Ptr() != 0 {
			_ = allocation.Free()
		}
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := allocation.Free(); err != nil {
			t.Error(err)
		}
	})
	guard := make([]int32, n+2)
	for i := range guard {
		guard[i] = 0x5a5a5a5a
	}
	if err := allocation.CopyFromHost(guard); err != nil {
		t.Fatal(err)
	}
	check := func() {
		got := make([]int32, n+2)
		if err := allocation.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		if got[0] != 0x5a5a5a5a || got[n+1] != 0x5a5a5a5a {
			t.Fatal("int redzone damaged")
		}
	}
	return allocation.Slice(1, n+1), check
}

func topkNativeCase(t *testing.T, dev *cuda.Device, artifact *cudakernel.Artifact, f topkCase) (topkAnswer, error) {
	t.Helper()
	e, err := f.c.Extents()
	if err != nil {
		t.Fatal(err)
	}
	selection, err := topkLoadArtifact(dev, f.c, artifact)
	if selection != nil {
		defer func() {
			if err := selection.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
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
	var b TopKBuffers
	var checks []func()
	floatBuf := func(n int) cudamem.Slice[float32] {
		s, check := topkNativeFloat(t, dev, n)
		checks = append(checks, check)
		return s
	}
	intBuf := func(n int) cudamem.Slice[int32] {
		s, check := topkNativeInt(t, dev, n)
		checks = append(checks, check)
		return s
	}
	b.Input = floatBuf(e.Input)
	b.Lengths = intBuf(e.RowMetadata)
	b.Offsets = intBuf(e.RowMetadata)
	b.ScratchIndices = intBuf(e.Results)
	b.ScratchCounts = intBuf(e.RowMetadata)
	b.Status = intBuf(e.RowMetadata)
	b.Indices = intBuf(e.Results)
	b.Counts = intBuf(e.RowMetadata)
	if f.c.Values {
		b.ScratchValues = floatBuf(e.Results)
		b.Values = floatBuf(e.Results)
	}
	if err := b.Input.CopyFromHost(f.input); err != nil {
		t.Fatal(err)
	}
	if err := b.Lengths.CopyFromHost(f.lengths); err != nil {
		t.Fatal(err)
	}
	if err := b.Offsets.CopyFromHost(f.offsets); err != nil {
		t.Fatal(err)
	}
	err = selection.Execute(stream, b)
	// A validation error follows successful synchronization. A CUDA failure is
	// terminal for this packet; do not read buffers or pretend a safe result.
	if err != nil && !strings.Contains(err.Error(), "device status") {
		t.Fatalf("native completion/launch/publication failed: %v", err)
	}
	a := topkAnswer{indices: make([]int32, e.Results), counts: make([]int32, e.RowMetadata), status: make([]int32, e.RowMetadata), values: make([]float32, e.Results)}
	a.resources = make(map[string]int)
	for _, fact := range []struct {
		name      string
		attribute driver.FunctionAttribute
	}{
		{"allocated_registers_per_thread", driver.FuncAttributeNumRegs},
		{"local_bytes_per_thread", driver.FuncAttributeLocalSizeBytes},
		{"static_shared_bytes", driver.FuncAttributeSharedSizeBytes},
		{"max_threads_per_block", driver.FuncAttributeMaxThreadsPerBlock},
	} {
		v, err := selection.kernel.Attribute(fact.attribute)
		if err != nil {
			t.Fatal(err)
		}
		a.resources[fact.name] = v
	}
	readInt := func(s cudamem.Slice[int32], out []int32) {
		if err := s.CopyToHost(out); err != nil {
			t.Fatal(err)
		}
	}
	readInt(b.Status, a.status)
	if err == nil {
		readInt(b.Indices, a.indices)
		readInt(b.Counts, a.counts)
		if f.c.Values {
			if e := b.Values.CopyToHost(a.values); e != nil {
				t.Fatal(e)
			}
		}
	} else {
		// All public outputs must still be poisoned when any row refuses status.
		indices, counts := make([]int32, e.Results), make([]int32, e.RowMetadata)
		readInt(b.Indices, indices)
		readInt(b.Counts, counts)
		for _, v := range append(indices, counts...) {
			if v != 0x5a5a5a5a {
				t.Fatal("refused invocation published indices/counts")
			}
		}
		if f.c.Values {
			values := make([]float32, e.Results)
			if e := b.Values.CopyToHost(values); e != nil {
				t.Fatal(e)
			}
			for _, v := range values {
				if math.Float32bits(v) != 0x5a5a5a5a {
					t.Fatal("refused invocation published values")
				}
			}
		}
		readInt(b.ScratchIndices, a.indices)
		readInt(b.ScratchCounts, a.counts)
		if f.c.Values {
			if e := b.ScratchValues.CopyToHost(a.values); e != nil {
				t.Fatal(e)
			}
		}
	}
	for _, check := range checks {
		check()
	}
	return a, err
}

// TestNativeTopKReference executes the fixed healthy fixtures and five mutated
// CUDA kernels. Compilation is separate; no fuzzing, search or timing promotion
// is performed. The caller must wrap this test with the existing gpu-lock.
func TestNativeTopKReference(t *testing.T) {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("native topk requires inherited gpu-lock: %v", err)
	}
	dir := os.Getenv("TOPK_PACKET_DIR")
	out := os.Getenv("TOPK_OUTPUT_DIR")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(out) {
		t.Skip("native topk requires retained offline packet and output directories")
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		t.Fatal(err)
	}
	dev := cudatest.Require(t)
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	target := fmt.Sprintf("sm_%d%d", major, minor)
	fixtures := topkFixtures()
	if len(fixtures)+len(topkMutations) > 48 {
		t.Fatal("native packet exceeds fixed-case budget")
	}
	baseline, entry := topkReadPacket(t, dir, "baseline", TopKSource, target)
	t.Logf("native topk target=%s healthy=%d mutants=%d max_rows=4 max_width=4096 max_k=32", target, len(fixtures), len(topkMutations))
	retain := func(name string, f topkCase, entry topkPacketEntry, got, want topkAnswer, err error) {
		r := topkNativeReceipt{Fixture: f.name, Config: f.c, Image: entry, InputBits: topkFloatBits(f.input), Lengths: f.lengths, Offsets: f.offsets, Indices: got.indices, Counts: got.counts, Status: got.status, ValueBits: topkFloatBits(got.values), ExpectedIndices: want.indices, ExpectedCounts: want.counts, ExpectedStatus: want.status, ExpectedValueBits: topkFloatBits(want.values), Published: err == nil, RedzonesIntact: true}
		r.LoadedResources = got.resources
		if err != nil {
			r.ExecuteError = err.Error()
		}
		if f.c.Values {
			policy := topkValuePolicy(f, entry)
			r.ValuePolicy = &policy
			if err := cudatest.CompareCUDAOutputFP32(got.values, want.values, policy); err != nil {
				r.ValueComparisonError = err.Error()
			}
		}
		topkWriteJSON(t, filepath.Join(out, name+".json"), r)
	}
	for _, f := range fixtures {
		t.Run(f.name, func(t *testing.T) {
			got, err := topkNativeCase(t, dev, baseline, f)
			want := topkSortReference(f)
			retain("healthy-"+f.name, f, entry, got, want, err)
			if f.c.Values {
				if err := cudatest.CompareCUDAOutputFP32(got.values, want.values, topkValuePolicy(f, entry)); err != nil {
					t.Fatal(err)
				}
			}
			if comparison := topkCompare(f.c, got, want); comparison != nil {
				t.Fatalf("%v: got %+v want %+v", comparison, got, want)
			}
			if (err == nil) != (CheckTopKStatus(want.status) == nil) {
				t.Fatal("status publication admission mismatch", err)
			}
		})
	}
	if t.Failed() {
		t.Fatal("baseline failed; mutant kernels refused")
	}
	for _, m := range topkMutations {
		t.Run("mutant_"+m.name, func(t *testing.T) {
			source, err := topkMutated(m)
			if err != nil {
				t.Fatal(err)
			}
			a, entry := topkReadPacket(t, dir, m.name, source, target)
			found := false
			for _, f := range fixtures {
				if f.name != m.fixture {
					continue
				}
				found = true
				got, err := topkNativeCase(t, dev, a, f)
				want := topkSortReference(f)
				retain("mutant-"+m.name, f, entry, got, want, err)
				if err := topkCompare(f.c, got, want); err == nil {
					t.Fatal("native behavioral mutant survived")
				} else {
					t.Logf("rejected native source mutant %s: %v", m.name, err)
				}
			}
			if !found {
				t.Fatal("missing mutant control")
			}
		})
	}
}
