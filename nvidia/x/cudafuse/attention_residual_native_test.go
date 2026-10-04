//go:build native

package cudafuse

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudablock"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type attentionResidualCase struct {
	Name   string
	Config AttentionResidualConfig
	Path   AttentionResidualPath
	Mutant int
	Kind   string
}

func attentionResidualCases() []attentionResidualCase {
	var out []attentionResidualCase
	for _, d := range []int{1, 31, 32, 33, 129} {
		for _, path := range []AttentionResidualPath{AttentionResidualComposed, AttentionResidualFused} {
			out = append(out, attentionResidualCase{fmt.Sprintf("dense-D%d-P%d", d, path), AttentionResidualConfig{4, 2, d, 1e-5}, path, 0, "dense"})
		}
	}
	for _, path := range []AttentionResidualPath{AttentionResidualComposed, AttentionResidualFused} {
		out = append(out, attentionResidualCase{fmt.Sprintf("single-P%d", path), AttentionResidualConfig{1, 2, 33, 1e-5}, path, 0, "single"})
	}
	out = append(out, attentionResidualCase{"equal-scores", AttentionResidualConfig{3, 2, 31, 0.001}, AttentionResidualComposed, 0, "equal"})
	for _, path := range []AttentionResidualPath{AttentionResidualComposed, AttentionResidualFused} {
		for call := 0; call < 2; call++ {
			out = append(out, attentionResidualCase{fmt.Sprintf("snapshot-P%d-call%d", path, call), AttentionResidualConfig{3, 2, 33, 1e-5}, path, 0, fmt.Sprintf("snapshot%d", call)})
		}
	}
	for mutant := 1; mutant <= 5; mutant++ {
		out = append(out, attentionResidualCase{fmt.Sprintf("mutant%d", mutant), AttentionResidualConfig{4, 2, 33, 1e-5}, AttentionResidualFused, mutant, "mutant"})
	}
	out = append(out, attentionResidualCase{"resident-nan", AttentionResidualConfig{3, 2, 33, 1e-5}, AttentionResidualFused, 0, "nonfinite"}, attentionResidualCase{"resident-square-overflow", AttentionResidualConfig{3, 2, 33, 1e-5}, AttentionResidualFused, 0, "overflow"})
	return out
}

func attentionResidualCaseSources(c attentionResidualCase) ([]FusedKernel, error) {
	s, err := AttentionResidualSource(c.Config, c.Path)
	if err != nil {
		return nil, err
	}
	if c.Mutant != 0 {
		for i := range s {
			s[i].Source = strings.Replace(s[i].Source, "#define AR_MUTANT 0", fmt.Sprintf("#define AR_MUTANT %d", c.Mutant), 1)
		}
	}
	return s, nil
}

func attentionResidualWrite(t *testing.T, dir, name string, b []byte) {
	t.Helper()
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute packet directory required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.Write(b)
	if err := errors.Join(err, f.Close()); err != nil {
		t.Fatal(err)
	}
}
func attentionResidualJSON(t *testing.T, dir, name string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	attentionResidualWrite(t, dir, name, append(b, '\n'))
}
func attentionResidualRead(t *testing.T, path string) []byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || !st.Mode().IsRegular() || st.Size() <= 0 || st.Size() > 16<<20 {
		t.Fatalf("invalid packet file %s: %v", path, err)
	}
	b, err := io.ReadAll(io.LimitReader(f, (16<<20)+1))
	if err != nil || len(b) > 16<<20 {
		t.Fatal("packet file limit", err)
	}
	return b
}
func attentionResidualDecode(t *testing.T, path string, v any) {
	t.Helper()
	b := attentionResidualRead(t, path)
	d := json.NewDecoder(strings.NewReader(string(b)))
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		t.Fatal(err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		t.Fatal("trailing packet JSON")
	}
}
func attentionResidualBits(v []float32) []uint32 {
	out := make([]uint32, len(v))
	for i, x := range v {
		out[i] = math.Float32bits(x)
	}
	return out
}

type attentionResidualPrepared struct {
	Case          attentionResidualCase
	Artifacts     []string
	ImageDigests  []string
	SourceDigests []string
}
type attentionResidualPreparation struct {
	Version                          int
	Target                           string
	CompilerMajor, CompilerMinor     int
	CompilerPath                     string
	FullHeaderDigest, FragmentDigest string
	Cases                            []attentionResidualPrepared
}

// Offline NVRTC only; never initialize CUDA or submit device work.
func TestAttentionResidual_Prepare(t *testing.T) {
	if os.Getenv("ATTENTION_RESIDUAL_PREPARE") != "1" {
		t.Skip("explicit offline preparation required")
	}
	dir := os.Getenv("ATTENTION_RESIDUAL_ARTIFACTS")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute artifact directory required")
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	fragment, err := attentionResidualHeader(cudablock.Header)
	if err != nil {
		t.Fatal(err)
	}
	prep := attentionResidualPreparation{Version: 1, Target: "compute_121", CompilerMajor: int(major), CompilerMinor: int(minor), CompilerPath: nvrtc.Path(), FullHeaderDigest: attentionResidualDigest(cudablock.Header), FragmentDigest: attentionResidualDigest(fragment)}
	attentionResidualWrite(t, dir, "cudablock-owner.cuh", []byte(cudablock.Header))
	attentionResidualWrite(t, dir, "cudablock-fp32.cuh", []byte(fragment))
	for _, c := range attentionResidualCases() {
		s, err := attentionResidualCaseSources(c)
		if err != nil {
			t.Fatal(err)
		}
		arts, err := compileAttentionResidualSources(s, 12, 1)
		if err != nil {
			t.Fatal(err)
		}
		p := attentionResidualPrepared{Case: c}
		for i, a := range arts {
			if _, err := attentionResidualArtifact(a, s[i]); err != nil {
				t.Fatal(err)
			}
			name := fmt.Sprintf("%s-stage%d.json", c.Name, i)
			attentionResidualJSON(t, dir, name, a)
			attentionResidualWrite(t, dir, fmt.Sprintf("%s-stage%d.cu", c.Name, i), []byte(s[i].Source))
			p.Artifacts = append(p.Artifacts, name)
			p.ImageDigests = append(p.ImageDigests, a.Digest)
			p.SourceDigests = append(p.SourceDigests, attentionResidualDigest(s[i].Source))
		}
		prep.Cases = append(prep.Cases, p)
	}
	attentionResidualJSON(t, dir, "compiler.json", prep)
}

func TestAttentionResidual_PacketBounds(t *testing.T) {
	cases := attentionResidualCases()
	if len(cases) != 24 {
		t.Fatalf("cases=%d", len(cases))
	}
	launches := 0
	for _, c := range cases {
		if err := c.Config.validate(); err != nil {
			t.Fatal(err)
		}
		if c.Path == AttentionResidualComposed {
			launches += 4
		} else {
			launches++
		}
	}
	if launches != 51 {
		t.Fatalf("launches=%d", launches)
	}
}

// This deterministic context cancels the observation only after the launch
// loop's last check. It does not assert that the GPU is still executing then.
type attentionResidualCancel struct {
	context.Context
	checks, after int
}

func (c *attentionResidualCancel) Err() error {
	c.checks++
	if c.checks >= c.after {
		return context.Canceled
	}
	return nil
}
func (c *attentionResidualCancel) Done() <-chan struct{}       { return nil }
func (c *attentionResidualCancel) Deadline() (time.Time, bool) { return time.Time{}, false }

type attentionResidualNativeRecord struct {
	Case                                                                      attentionResidualCase
	Values, Multiplier, Query, NormOffsets, Output, InputGuards, OutputGuards []uint32
	Expected                                                                  []float64
	Images                                                                    []string
	SourceDigests                                                             []string
	Resources                                                                 [][3]int
	Launches, ScratchBytes                                                    int
	Diagnostic                                                                string
	SnapshotRetained, CancelRetained, Recovered                               bool
}

// Actual 24-case/51-launch packet: 17 healthy cases (including two calls per
// snapshot path), five executed mathematical controls, and two resident-device
// refusals. Every healthy output and all input/output guards are retained.
func TestAttentionResidual_Native(t *testing.T) {
	if os.Getenv("ATTENTION_RESIDUAL_NATIVE") != "1" {
		t.Skip("explicit native qualification required")
	}
	pid, err := strconv.Atoi(os.Getenv("GPU_LOCK_HELD"))
	if err != nil || pid <= 0 {
		t.Fatal("requires actual gpu-lock holder pid")
	}
	dir, outDir := os.Getenv("ATTENTION_RESIDUAL_ARTIFACTS"), os.Getenv("ATTENTION_RESIDUAL_OUTPUT")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(outDir) {
		t.Fatal("absolute native artifact/output directories required")
	}
	manifestPath := os.Getenv("ATTENTION_RESIDUAL_MANIFEST")
	if !filepath.IsAbs(manifestPath) {
		t.Fatal("absolute source manifest required")
	}
	manifestBytes := attentionResidualRead(t, manifestPath)
	var manifest struct {
		Owned map[string]string `json:"owned"`
	}
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatal(err)
	}
	ownedNames := []string{"attention_residual.go", "attention_residual_source.go", "attention_residual_test.go", "attention_residual_ref_test.go", "attention_residual_native_test.go", "attention_residual_example_test.go"}
	if len(manifest.Owned) != len(ownedNames) {
		t.Fatal("source pin count differs")
	}
	for _, name := range ownedNames {
		p := "nvidia/x/cudafuse/" + name
		b, err := attentionResidualSourceBytes(os.Getenv("ATTENTION_RESIDUAL_MANIFEST"), p)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != manifest.Owned[p] {
			t.Fatal("source pin differs", p)
		}
	}
	var prep attentionResidualPreparation
	attentionResidualDecode(t, filepath.Join(dir, "compiler.json"), &prep)
	fragment, err := attentionResidualHeader(cudablock.Header)
	if err != nil {
		t.Fatal(err)
	}
	if prep.Version != 1 || prep.Target != "compute_121" || len(prep.Cases) != 24 || prep.FullHeaderDigest != attentionResidualDigest(cudablock.Header) || prep.FragmentDigest != attentionResidualDigest(fragment) || string(attentionResidualRead(t, filepath.Join(dir, "cudablock-owner.cuh"))) != cudablock.Header || string(attentionResidualRead(t, filepath.Join(dir, "cudablock-fp32.cuh"))) != fragment {
		t.Fatal("stale header/compiler packet")
	}
	// Validate every image before CUDA initialization, including all mutant
	// source associations. Mutants are test-only trusted compilation inputs.
	var allArtifacts [][]*cudakernel.Artifact
	for i, c := range attentionResidualCases() {
		p := prep.Cases[i]
		if !reflect.DeepEqual(p.Case, c) {
			t.Fatal("packet fixture differs")
		}
		s, err := attentionResidualCaseSources(c)
		if err != nil {
			t.Fatal(err)
		}
		if len(p.Artifacts) != len(s) || len(p.ImageDigests) != len(s) || len(p.SourceDigests) != len(s) {
			t.Fatal("missing stage binding")
		}
		var arts []*cudakernel.Artifact
		for j, name := range p.Artifacts {
			if name != fmt.Sprintf("%s-stage%d.json", c.Name, j) {
				t.Fatal("unsafe artifact path")
			}
			var a cudakernel.Artifact
			attentionResidualDecode(t, filepath.Join(dir, name), &a)
			if a.Digest != p.ImageDigests[j] || p.SourceDigests[j] != attentionResidualDigest(s[j].Source) || string(attentionResidualRead(t, filepath.Join(dir, fmt.Sprintf("%s-stage%d.cu", c.Name, j)))) != s[j].Source {
				t.Fatal("stale image/source association")
			}
			owned, err := attentionResidualArtifact(&a, s[j])
			if err != nil {
				t.Fatal(err)
			}
			arts = append(arts, owned)
		}
		allArtifacts = append(allArtifacts, arts)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	cleanup := &attentionResidualCleanupStack{}
	cleanup.add(dev.Close)
	defer func() {
		if err := cleanup.close(); err != nil {
			attentionResidualUnresolved = append(attentionResidualUnresolved, cleanup)
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil || major != 12 || minor != 1 {
		t.Fatal("requires SM121", err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	cleanup.add(stream.Close)
	type snapshotState struct {
		s        *AttentionResidualSnapshots
		owners   []cudamem.Slice[float32]
		original [][]float32
	}
	states := map[AttentionResidualPath]*snapshotState{}
	cleanup.add(func() error {
		for _, s := range states {
			if err := s.s.Close(); err != nil {
				return err
			}
			for _, b := range s.owners {
				if err := b.Free(); err != nil {
					return err
				}
			}
		}
		return nil
	})
	var unresolvedCase *attentionResidualCleanupStack
	cleanup.add(func() error {
		if unresolvedCase != nil {
			return unresolvedCase.close()
		}
		return nil
	})
	totalLaunches, totalValues := 0, 0
	for index, c := range attentionResidualCases() {
		func() {
			cleanup := &attentionResidualCleanupStack{}
			defer func() {
				if err := cleanup.close(); err != nil {
					attentionResidualUnresolved = append(attentionResidualUnresolved, cleanup)
					unresolvedCase = cleanup
					t.Fatal(err)
				}
			}()
			v, g, w, offset := attentionResidualFixture(c.Config)
			if c.Kind == "single" {
				v[0] = math.Float32frombits(0x80000000)
				v[1] = math.Float32frombits(1)
				v[2] = math.Float32frombits(0x80000001)
			}
			if c.Kind == "equal" {
				clear(w)
			}
			if c.Kind == "snapshot1" {
				for i := 2 * c.Config.Rows * c.Config.Width; i < len(v); i++ {
					v[i] += 0.7
				}
			}
			if c.Kind == "nonfinite" {
				v[len(v)-1] = float32(math.NaN())
			}
			if c.Kind == "overflow" {
				v[0] = float32(math.MaxFloat32)
			}
			var expected []float64
			if c.Kind != "nonfinite" && c.Kind != "overflow" {
				expected, _ = attentionResidualReference(c.Config, v, g, w, 0)
			}
			alloc := func(host []float32) (cudamem.Slice[float32], cudamem.Slice[float32]) {
				owner, err := cudamem.Alloc[float32](dev, len(host)+2)
				if err != nil {
					if owner.Ptr() != 0 {
						cleanup.add(owner.Free)
					}
					t.Fatal(err)
				}
				h := append(append([]float32{attentionResidualGuard}, host...), attentionResidualGuard)
				if err := owner.CopyFromHost(h); err != nil {
					cleanup.add(owner.Free)
					t.Fatal(err)
				}
				return owner, owner.Slice(1, len(h)-1)
			}
			var owners []cudamem.Slice[float32]
			var bindings []cudamem.Slice[float32]
			cleanup.add(func() error {
				for _, b := range owners {
					if err := b.Free(); err != nil {
						return err
					}
				}
				return nil
			})
			poison := make([]float32, c.Config.Rows*c.Config.Width)
			for i := range poison {
				poison[i] = attentionResidualPoison
			}
			for _, h := range [][]float32{v, g, w, poison} {
				owner, slice := alloc(h)
				owners = append(owners, owner)
				bindings = append(bindings, slice)
			}
			sources, err := attentionResidualCaseSources(c)
			if err != nil {
				t.Fatal(err)
			}
			x, err := loadAttentionResidualSources(dev, c.Config, c.Path, allArtifacts[index], sources)
			if x != nil {
				cleanup.add(x.Close)
			}
			if err != nil {
				t.Fatal(err)
			}
			// Public admission must refuse a test-only mutant before any launch.
			if c.Mutant != 0 {
				if admitted, err := LoadAttentionResidual(dev, c.Config, c.Path, allArtifacts[index]); err == nil {
					if admitted != nil {
						cleanup.add(admitted.Close)
					}
					t.Fatal("public loader admitted mathematical mutant")
				}
			}
			record := attentionResidualNativeRecord{Case: c, Values: attentionResidualBits(v), Multiplier: attentionResidualBits(g), Query: attentionResidualBits(w), NormOffsets: attentionResidualBits(offset), Expected: expected, Images: prep.Cases[index].ImageDigests, SourceDigests: prep.Cases[index].SourceDigests}
			for _, k := range x.kernels {
				var facts [3]int
				for j, a := range []driver.FunctionAttribute{driver.FuncAttributeNumRegs, driver.FuncAttributeSharedSizeBytes, driver.FuncAttributeLocalSizeBytes} {
					facts[j], err = k.Attribute(a)
					if err != nil {
						t.Fatal(err)
					}
				}
				if facts[0] <= 0 {
					t.Fatal("missing actual register facts")
				}
				record.Resources = append(record.Resources, facts)
			}
			// An aliased destination is refused before kernels or allocation.
			alias, aliasErr := x.Execute(context.Background(), stream, AttentionResidualInputs{bindings[0], bindings[1], bindings[2]}, bindings[0].Slice(0, c.Config.Rows*c.Config.Width))
			cleanup.adopt(alias != nil, alias.Close)
			if aliasErr == nil {
				t.Fatal("aliased output accepted")
			}
			var e *AttentionResidualExecution
			if strings.HasPrefix(c.Kind, "snapshot") {
				state := states[c.Path]
				if state == nil {
					state = &snapshotState{}
					states[c.Path] = state
					for k := 0; k < 2; k++ {
						n := c.Config.Rows * c.Config.Width
						host := append([]float32(nil), v[k*n:(k+1)*n]...)
						owner, slice := alloc(host)
						state.owners = append(state.owners, owner)
						state.original = append(state.original, host)
						if state.s == nil {
							state.s = &AttentionResidualSnapshots{}
						}
						_ = slice
					}
					completed := []cudamem.Slice[float32]{state.owners[0].Slice(1, state.owners[0].Len()-1), state.owners[1].Slice(1, state.owners[1].Len()-1)}
					state.s, err = NewAttentionResidualSnapshots(dev, c.Config.Rows, c.Config.Width, completed)
					if err != nil {
						t.Fatal(err)
					}
					states[c.Path] = state
				}
				partial := bindings[0].Slice(2*c.Config.Rows*c.Config.Width, bindings[0].Len())
				ctx := context.Context(context.Background())
				if c.Kind == "snapshot0" {
					after := 3
					if c.Path == AttentionResidualComposed {
						after = 6
					}
					ctx = &attentionResidualCancel{Context: context.Background(), after: after}
				}
				e, err = state.s.Execute(ctx, x, stream, partial, bindings[1], bindings[2], bindings[3])
				cleanup.adopt(e != nil, e.Close)
				if c.Kind == "snapshot0" {
					if !errors.Is(err, context.Canceled) || e == nil {
						t.Fatal("queued snapshot observation did not cancel", err)
					}
					if state.s.Close() != cudamem.ErrInUse || bindings[0].Free() != cudamem.ErrInUse || bindings[3].Free() != cudamem.ErrInUse || x.Close() != cudamem.ErrInUse {
						t.Fatal("canceled snapshot reads were not retained")
					}
					record.CancelRetained = true
					err = e.Wait(context.Background())
					record.Recovered = err == nil
				}
				for k, b := range state.owners {
					h := make([]float32, b.Len())
					if err := b.CopyToHost(h); err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(h[1:len(h)-1], state.original[k]) {
						t.Fatal("completed snapshot changed")
					}
					if b.Free() != cudamem.ErrInUse {
						t.Fatal("snapshot owner lease missing")
					}
				}
				record.SnapshotRetained = true
			} else {
				e, err = x.Execute(context.Background(), stream, AttentionResidualInputs{bindings[0], bindings[1], bindings[2]}, bindings[3])
				cleanup.adopt(e != nil, e.Close)
			}
			if e != nil {
				record.Launches, record.ScratchBytes = e.Stats()
				totalLaunches += record.Launches
			}
			if c.Kind == "nonfinite" || c.Kind == "overflow" {
				if err == nil || e == nil {
					t.Fatal("resident invalid input published")
				}
				record.Diagnostic = err.Error()
				want := "nonfinite resident"
				if c.Kind == "overflow" {
					want = "intermediate overflow"
				}
				if !strings.Contains(err.Error(), want) {
					t.Fatal("wrong resident diagnostic", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			output := make([]float32, bindings[3].Len())
			if err := bindings[3].CopyToHost(output); err != nil {
				t.Fatal(err)
			}
			record.Output = attentionResidualBits(output)
			if c.Kind == "single" && !reflect.DeepEqual(record.Output, attentionResidualBits(v)) {
				t.Fatal("single candidate did not preserve raw FP32 bits")
			}
			if c.Kind == "nonfinite" || c.Kind == "overflow" {
				for _, b := range record.Output {
					if b != math.Float32bits(attentionResidualPoison) {
						t.Fatal("invalid resident input published output")
					}
				}
			} else {
				err := attentionResidualCompare(output, expected)
				if c.Mutant != 0 {
					if err == nil {
						t.Fatal("executed mathematical mutant accepted")
					}
					record.Diagnostic = err.Error()
				} else if err != nil {
					t.Fatal(err)
				}
				totalValues += len(output)
			}
			for i, b := range owners {
				h := make([]float32, b.Len())
				if err := b.CopyToHost(h); err != nil {
					t.Fatal(err)
				}
				if h[0] != attentionResidualGuard || h[len(h)-1] != attentionResidualGuard {
					t.Fatal("caller guard changed")
				}
				if i < 3 {
					for j, want := range [][]float32{v, g, w}[i] {
						if math.Float32bits(h[j+1]) != math.Float32bits(want) {
							t.Fatal("input mutated")
						}
					}
					record.InputGuards = append(record.InputGuards, math.Float32bits(h[0]), math.Float32bits(h[len(h)-1]))
				} else {
					record.OutputGuards = []uint32{math.Float32bits(h[0]), math.Float32bits(h[len(h)-1])}
				}
			}
			attentionResidualJSON(t, outDir, c.Name+".json", record)
		}()
	}
	if totalLaunches != 51 {
		t.Fatalf("actual launches=%d", totalLaunches)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary := attentionResidualRead(t, self)
	attentionResidualWrite(t, outDir, "executed.test", binary)
	attentionResidualWrite(t, outDir, "source-manifest.json", manifestBytes)
	attentionResidualWrite(t, outDir, "compiler.json", attentionResidualRead(t, filepath.Join(dir, "compiler.json")))
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	attentionResidualJSON(t, outDir, "execution.json", struct {
		Device, UUID, GoVersion, ExecutableSHA256 string
		DriverVersion, Launches, CheckedValues    int
	}{name, fmt.Sprintf("%x", uuid.Bytes), runtime.Version(), fmt.Sprintf("%x", sha256.Sum256(binary)), version, totalLaunches, totalValues})
}

// The separate performance packet is gated on actual first-packet receipts.
// It interleaves five samples per path after two warmups per path: 35 launches.
// Event intervals include host submission gaps and timer joins, not host setup,
// allocation or transfers. No speedup or model-level claim follows from them.
func TestAttentionResidual_Timing(t *testing.T) {
	if os.Getenv("ATTENTION_RESIDUAL_TIMING") != "1" {
		t.Skip("explicit post-correctness timing packet required")
	}
	pid, err := strconv.Atoi(os.Getenv("GPU_LOCK_HELD"))
	if err != nil || pid <= 0 {
		t.Fatal("requires actual gpu-lock holder pid")
	}
	dir, outDir, qualified := os.Getenv("ATTENTION_RESIDUAL_ARTIFACTS"), os.Getenv("ATTENTION_RESIDUAL_TIMING_OUTPUT"), os.Getenv("ATTENTION_RESIDUAL_QUALIFIED")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(outDir) || !filepath.IsAbs(qualified) {
		t.Fatal("absolute performance receipt directories required")
	}
	var run struct {
		Device, UUID, GoVersion, ExecutableSHA256 string
		DriverVersion, Launches, CheckedValues    int
	}
	attentionResidualDecode(t, filepath.Join(qualified, "execution.json"), &run)
	if run.Launches != 51 || run.CheckedValues <= 0 || len(run.ExecutableSHA256) != 64 {
		t.Fatal("unqualified correctness packet")
	}
	actual := attentionResidualRead(t, filepath.Join(qualified, "executed.test"))
	if fmt.Sprintf("%x", sha256.Sum256(actual)) != run.ExecutableSHA256 {
		t.Fatal("stale correctness binary")
	}
	manifest := attentionResidualRead(t, os.Getenv("ATTENTION_RESIDUAL_MANIFEST"))
	if string(manifest) != string(attentionResidualRead(t, filepath.Join(qualified, "source-manifest.json"))) {
		t.Fatal("correctness source differs")
	}
	var pins struct {
		Owned map[string]string `json:"owned"`
	}
	if err := json.Unmarshal(manifest, &pins); err != nil || len(pins.Owned) != 6 {
		t.Fatal("missing performance source pins", err)
	}
	for p, want := range pins.Owned {
		if !strings.HasPrefix(p, "nvidia/x/cudafuse/attention_residual") || strings.Contains(p, "..") {
			t.Fatal("unsafe source path")
		}
		b, err := attentionResidualSourceBytes(os.Getenv("ATTENTION_RESIDUAL_MANIFEST"), p)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatal("stale performance source")
		}
	}
	var prep attentionResidualPreparation
	attentionResidualDecode(t, filepath.Join(dir, "compiler.json"), &prep)
	if string(attentionResidualRead(t, filepath.Join(dir, "compiler.json"))) != string(attentionResidualRead(t, filepath.Join(qualified, "compiler.json"))) {
		t.Fatal("correctness compiler differs")
	}
	c := AttentionResidualConfig{4, 2, 129, 1e-5}
	v, g, w, _ := attentionResidualFixture(c)
	expected, _ := attentionResidualReference(c, v, g, w, 0)
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	cleanup := &attentionResidualCleanupStack{}
	cleanup.add(dev.Close)
	defer func() {
		if err := cleanup.close(); err != nil {
			attentionResidualUnresolved = append(attentionResidualUnresolved, cleanup)
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil || major != 12 || minor != 1 {
		t.Fatal("requires SM121", err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	cleanup.add(stream.Close)
	var owners []cudamem.Slice[float32]
	var views []cudamem.Slice[float32]
	var releases []func()
	var consumers []*AttentionResidual
	status, err := cudamem.Alloc[int32](dev, c.Rows+2)
	if status.Ptr() != 0 {
		cleanup.add(status.Free)
	}
	if err != nil {
		t.Fatal(err)
	}
	cleanup.add(func() error {
		if err := stream.Synchronize(context.Background()); err != nil {
			return fmt.Errorf("unresolved performance resources retained: %w", err)
		}
		for _, r := range releases {
			r()
		}
		releases = nil
		for _, b := range owners {
			if err := b.Free(); err != nil {
				return err
			}
		}
		for _, x := range consumers {
			if err := x.Close(); err != nil {
				return err
			}
		}
		return nil
	})
	requested := status.SizeBytes()
	for _, h := range [][]float32{v, g, w, make([]float32, c.Candidates*c.Rows), make([]float32, c.Candidates*c.Rows), make([]float32, c.Rows*c.Width)} {
		b, err := cudamem.Alloc[float32](dev, len(h)+2)
		if b.Ptr() != 0 {
			owners = append(owners, b)
			requested += b.SizeBytes()
		}
		if err != nil {
			t.Fatal(err)
		}
		padded := append(append([]float32{attentionResidualGuard}, h...), attentionResidualGuard)
		if err := b.CopyFromHost(padded); err != nil {
			t.Fatal(err)
		}
		view := b.Slice(1, b.Len()-1)
		r, err := view.Borrow(dev)
		if err != nil {
			t.Fatal(err)
		}
		releases = append(releases, r)
		views = append(views, view)
	}
	r, err := status.Borrow(dev)
	if err != nil {
		t.Fatal(err)
	}
	releases = append(releases, r)
	stat := status.Slice(1, status.Len()-1)
	for i, path := range []AttentionResidualPath{AttentionResidualComposed, AttentionResidualFused} {
		index := 8 + i
		if index >= len(prep.Cases) || prep.Cases[index].Case.Config != c || prep.Cases[index].Case.Path != path || prep.Cases[index].Case.Mutant != 0 {
			t.Fatal("wrong performance fixture")
		}
		var arts []*cudakernel.Artifact
		for _, name := range prep.Cases[index].Artifacts {
			if filepath.Base(name) != name {
				t.Fatal("unsafe artifact name")
			}
			var a cudakernel.Artifact
			attentionResidualDecode(t, filepath.Join(dir, name), &a)
			arts = append(arts, &a)
		}
		x, err := LoadAttentionResidual(dev, c, path, arts)
		if x != nil {
			consumers = append(consumers, x)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	timer, err := cuda.NewEventTimer(dev, []*cuda.Stream{stream})
	cleanup.adopt(timer != nil, func() error {
		if err := stream.Synchronize(context.Background()); err != nil {
			return fmt.Errorf("performance timer drain: %w", err)
		}
		return timer.Close()
	})
	if err != nil {
		t.Fatal(err)
	}
	if !timer.Checked() {
		t.Fatal("GPU event interval precondition unavailable")
	}
	ptrs := []driver.DevicePtr{views[0].Ptr(), views[1].Ptr(), views[2].Ptr(), views[3].Ptr(), views[4].Ptr(), views[5].Ptr(), stat.Ptr()}
	args := attentionResidualArguments(ptrs)
	var samples [2][]int64
	launches := 0
	for iteration := 0; iteration < 7; iteration++ {
		for index, x := range consumers {
			for i, b := range owners {
				if i < 3 {
					continue
				}
				h := make([]float32, b.Len())
				for j := range h {
					h[j] = attentionResidualPoison
				}
				h[0], h[len(h)-1] = attentionResidualGuard, attentionResidualGuard
				if err := b.CopyFromHost(h); err != nil {
					t.Fatal(err)
				}
			}
			sv := make([]int32, status.Len())
			for i := range sv {
				sv[i] = -1
			}
			sv[0], sv[len(sv)-1] = 0x13579, 0x13579
			if err := status.CopyFromHost(sv); err != nil {
				t.Fatal(err)
			}
			if err := stream.Synchronize(context.Background()); err != nil {
				t.Fatal(err)
			}
			duration, err := timer.Measure(func() error {
				for _, k := range x.kernels {
					if err := stream.WithHandle(func(raw driver.Stream) error { return k.Launch([3]uint32{2, 1, 1}, [3]uint32{32, 1, 1}, 0, raw, args) }); err != nil {
						return err
					}
					launches++
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if duration <= 0 {
				t.Fatal("GPU event interval unresolved")
			}
			if iteration >= 2 {
				samples[index] = append(samples[index], duration.Nanoseconds())
			}
			if err := status.CopyToHost(sv); err != nil {
				t.Fatal(err)
			}
			if sv[0] != 0x13579 || sv[len(sv)-1] != 0x13579 || attentionResidualStatus(sv[1:len(sv)-1]) != nil {
				t.Fatal("invalid timed kernel statuses")
			}
			out := make([]float32, views[5].Len())
			if err := views[5].CopyToHost(out); err != nil {
				t.Fatal(err)
			}
			if err := attentionResidualCompare(out, expected); err != nil {
				t.Fatal(err)
			}
			for _, b := range owners {
				h := make([]float32, b.Len())
				if err := b.CopyToHost(h); err != nil {
					t.Fatal(err)
				}
				if h[0] != attentionResidualGuard || h[len(h)-1] != attentionResidualGuard {
					t.Fatal("timed guard changed")
				}
			}
			attentionResidualJSON(t, outDir, fmt.Sprintf("iteration%d-path%d.json", iteration, index), struct {
				Config        AttentionResidualConfig
				Path          AttentionResidualPath
				DurationNanos int64
				Output        []uint32
				Expected      []float64
				Images        []string
			}{c, x.path, duration.Nanoseconds(), attentionResidualBits(out), expected, prep.Cases[8+index].ImageDigests})
		}
	}
	if launches != 35 {
		t.Fatal("performance launch bound differs")
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary := attentionResidualRead(t, self)
	attentionResidualWrite(t, outDir, "executed.test", binary)
	attentionResidualJSON(t, outDir, "timing.json", struct {
		Config                                   AttentionResidualConfig
		Samples                                  [2][]int64
		Launches                                 int
		RequestedBytes                           uintptr
		Interval, Setup, CUPTI, ExecutableSHA256 string
	}{c, samples, launches, requested, "GPU events; includes host launch gaps and timer joins", "allocation, compilation, transfers and preparation outside intervals; no wall-time comparison", "unmeasured", fmt.Sprintf("%x", sha256.Sum256(binary))})
}
