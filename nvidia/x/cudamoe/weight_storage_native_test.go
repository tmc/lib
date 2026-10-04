//go:build native

package cudamoe

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func weightPacketSources() map[string]string {
	view, _, _ := MoEWeightSource(MoEWeightViewConsumer)
	slab, _, _ := MoEWeightSource(MoEWeightSlabConsumer)
	return map[string]string{
		"view":               view,
		"slab":               slab,
		"scale-half-split":   strings.ReplaceAll(strings.ReplaceAll(slab, "scales[gate_row]", "scales[base+n]"), "scales[up_row]", "scales[base+intermediate+n]"),
		"gate-up-swap":       strings.Replace(slab, "gate_row=base+(n/8)*16+n%8, up_row=gate_row+8", "up_row=base+(n/8)*16+n%8, gate_row=up_row+8", 1),
		"wrong-global-layer": strings.Replace(slab, "((unsigned long long)layer*experts+expert)", "((unsigned long long)0*experts+expert)", 1),
		"no-op":              strings.Replace(slab, "output[n]=", "if (false) output[n]=", 1),
	}
}

func weightPacketDigest(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }

func weightPacketJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}

func weightPacketRead(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	s, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !s.Mode().IsRegular() || s.Size() < 0 || s.Size() > limit {
		return nil, fmt.Errorf("invalid or oversized packet file")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("packet grew past bound")
	}
	return b, nil
}

type weightPacketCompiler struct {
	Path, SHA256 string
	Major, Minor int32
}

// TestBuildMoEWeightPacket compiles six ordinary images without initializing a
// CUDA driver or querying a device. Compilation is not native qualification.
func TestBuildMoEWeightPacket(t *testing.T) {
	if os.Getenv("MOE_WEIGHT_BUILD") != "1" {
		t.Skip("offline packet preparation not requested")
	}
	dir, path, target := os.Getenv("MOE_WEIGHT_PACKET"), os.Getenv("NVRTC_LIB_PATH"), os.Getenv("MOE_WEIGHT_TARGET")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(path) || target != "sm_121" {
		t.Fatal("absolute packet/compiler paths and sm_121 required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := nvrtc.Load(path); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	compiler, err := weightPacketRead(nvrtc.Path(), 512<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := weightPacketJSON(filepath.Join(dir, "compiler.json"), weightPacketCompiler{nvrtc.Path(), weightPacketDigest(compiler), major, minor}); err != nil {
		t.Fatal(err)
	}
	schema, err := MoEWeightSchema()
	if err != nil {
		t.Fatal(err)
	}
	for name, source := range weightPacketSources() {
		entry := "moe_weight_slab"
		if name == "view" {
			entry = "moe_weight_view"
		}
		cm, err := nvrtcx.Compile(source, nvrtcx.WithName("cudamoe_weight_storage.cu"), nvrtcx.WithPhysicalArchTarget(target), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"), nvrtcx.WithNoCache())
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		a, err := cm.ToArtifact(entry, schema, target)
		if err != nil {
			t.Fatal(err)
		}
		a.Toolchain = fmt.Sprintf("nvrtc %d.%d", major, minor)
		a.CompilerOpts = []string{"--gpu-architecture=" + target, "--std=c++17", "--fmad=false", "--ftz=false"}
		a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: "cudamoe_weight_storage.cu", Digest: weightPacketDigest([]byte(source)), Tracked: true})
		if err := weightPacketJSON(filepath.Join(dir, name+".artifact.json"), a); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".cu"), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".cubin"), a.Raw, 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("COMPILED %s source=%s image=%s target=%s", name, weightPacketDigest([]byte(source)), a.Digest, target)
	}
}

func weightPacketArtifact(dir, name string) (*cudakernel.Artifact, error) {
	source, ok := weightPacketSources()[name]
	if !ok {
		return nil, fmt.Errorf("unknown packet arm")
	}
	data, err := weightPacketRead(filepath.Join(dir, name+".artifact.json"), 16<<20)
	if err != nil {
		return nil, err
	}
	var a cudakernel.Artifact
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&a); err != nil {
		return nil, err
	}
	var extra any
	if dec.Decode(&extra) != io.EOF {
		return nil, fmt.Errorf("trailing artifact JSON")
	}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	cu, err := weightPacketRead(filepath.Join(dir, name+".cu"), 1<<20)
	if err != nil {
		return nil, err
	}
	image, err := weightPacketRead(filepath.Join(dir, name+".cubin"), 16<<20)
	if err != nil {
		return nil, err
	}
	schema, err := MoEWeightSchema()
	if err != nil {
		return nil, err
	}
	entry := "moe_weight_slab"
	if name == "view" {
		entry = "moe_weight_view"
	}
	if string(cu) != source || !bytes.Equal(image, a.Raw) || a.Kind != cudakernel.KindCUBIN || a.Target != "sm_121" || a.EntryPoint != entry || !reflect.DeepEqual(a.Schema, schema) || !reflect.DeepEqual(a.CompilerOpts, []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}) {
		return nil, fmt.Errorf("packet source/image/ABI/options differ")
	}
	n := 0
	for _, d := range a.Dependencies {
		if d.Path == "cudamoe_weight_storage.cu" {
			n++
			if !d.Tracked || d.ErrorMsg != "" || d.Digest != weightPacketDigest(cu) {
				return nil, fmt.Errorf("packet source binding differs")
			}
		}
	}
	if n != 1 {
		return nil, fmt.Errorf("packet source binding absent or repeated")
	}
	return &a, nil
}

// This reference works from logical half-split rows, not the production row
// permutation or decoder. IEEE FP32 construction decodes finite normal values;
// subnormal FP8 values are exact integer multiples of 2^-9.
func weightPacketReference(raw []byte, scales, input []float32, intermediate int) []float32 {
	decode := func(b byte) float64 {
		e, m := uint32(b&127)>>3, uint32(b&7)
		var v float32
		if e == 0 {
			v = float32(m) / 512
		} else {
			v = math.Float32frombits(((e + 120) << 23) | (m << 20))
		}
		if b&128 != 0 {
			v = -v
		}
		return float64(v)
	}
	out := make([]float32, intermediate)
	for n := range out {
		var g, u float64
		for c, x := range input {
			g += decode(raw[n*len(input)+c]) * float64(x)
			u += decode(raw[(intermediate+n)*len(input)+c]) * float64(x)
		}
		g *= float64(scales[n])
		u *= float64(scales[intermediate+n])
		out[n] = float32(g / (1 + math.Exp(-g)) * u)
	}
	return out
}

func weightPacketLock() error {
	if os.Getenv("GPU_LOCK_DISABLED") != "" || os.Getenv("CUDATEST_NO_LOCK") != "" {
		return fmt.Errorf("disabled lock cannot qualify native work")
	}
	pid, err := strconv.Atoi(os.Getenv("GPU_LOCK_HELD"))
	if err != nil || pid <= 0 {
		return fmt.Errorf("real inherited gpu-lock pid required")
	}
	return nil
}

func TestMoEWeightPacketReference(t *testing.T) {
	for _, width := range []int{17, 32} {
		raw := make([]byte, 16*width)
		for i := range raw {
			raw[i] = 0x38
			if i >= 8*width {
				raw[i] = 0x40
			}
		}
		scales := make([]float32, 16)
		for i := range scales {
			scales[i] = 0.125
		}
		input := make([]float32, width)
		for i := range input {
			input[i] = 1
		}
		out := weightPacketReference(raw, scales, input, 8)
		gate := float64(width) / 8
		want := float32(gate / (1 + math.Exp(-gate)) * float64(width) / 4)
		for i, v := range out {
			if v != want {
				t.Fatalf("width%d neuron%d=%g want%g", width, i, v, want)
			}
		}
	}
	sources := weightPacketSources()
	if len(sources) != 6 {
		t.Fatal("incorrect bounded image count")
	}
	for name, source := range sources {
		if name != "view" && name != "slab" && source == sources["slab"] {
			t.Fatalf("vacuous %s mutant", name)
		}
	}
	if weightPacketLock() == nil {
		t.Fatal("host test unexpectedly inherits a GPU lease")
	}
}

func weightNativePayload(intermediate, hidden, layer, expert int) ([]byte, []float32) {
	raw := make([]byte, 2*intermediate*hidden)
	choices := []byte{0x38, 0x3c, 0x40, 0xb8, 0xbc, 0x01, 0x80, 0x00}
	for i := range raw {
		raw[i] = choices[(i+i/hidden+layer*3+expert)%len(choices)]
	}
	scales := make([]float32, 2*intermediate)
	for i := range scales {
		scales[i] = float32(1+(i*3+layer*5+expert*7)%11) / 64
	}
	return raw, scales
}

func weightNativeScaleBytes(scales []float32) []byte {
	b := make([]byte, 4*len(scales))
	for i, v := range scales {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func weightNativeMutant(dev *cuda.Device, a *cudakernel.Artifact) (*MoEWeightConsumer, error) {
	// Deliberate compiler-bound source variants must not pass the public loader's
	// baseline binding. This private test owner loads their exact retained image.
	c := &MoEWeightConsumer{device: dev, kind: MoEWeightSlabConsumer}
	var err error
	c.raw, err = a.Load(dev)
	if err != nil {
		return c, err
	}
	c.module, err = cuda.OwnModule(dev, c.raw)
	if err != nil {
		return c, err
	}
	c.raw = 0
	c.kernel, err = c.module.Function(a.EntryPoint)
	return c, err
}

func TestNativeSharedMoEWeightConsumers(t *testing.T) {
	if os.Getenv("MOE_WEIGHT_NATIVE") != "1" {
		t.Skip("scoped native packet not requested")
	}
	if err := weightPacketLock(); err != nil {
		t.Fatal(err)
	}
	dir, out := os.Getenv("MOE_WEIGHT_PACKET"), os.Getenv("MOE_WEIGHT_RESULTS")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(out) {
		t.Fatal("absolute prepared/result directories required")
	}
	manifestPath := os.Getenv("MOE_WEIGHT_SOURCE_MANIFEST")
	if !filepath.IsAbs(manifestPath) {
		t.Fatal("absolute source manifest required")
	}
	sourceManifest, err := weightPacketRead(manifestPath, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var compiler weightPacketCompiler
	compilerJSON, err := weightPacketRead(filepath.Join(dir, "compiler.json"), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(bytes.NewReader(compilerJSON))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&compiler); err != nil {
		t.Fatal(err)
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		t.Fatal("trailing compiler JSON")
	}
	if !filepath.IsAbs(compiler.Path) || len(compiler.SHA256) != 64 || strings.Trim(compiler.SHA256, "0123456789abcdef") != "" || compiler.Major <= 0 || compiler.Minor < 0 {
		t.Fatal("invalid prepared compiler identity")
	}
	actualCompiler, err := weightPacketRead(compiler.Path, 512<<20)
	if err != nil {
		t.Fatal(err)
	}
	if weightPacketDigest(actualCompiler) != compiler.SHA256 {
		t.Fatal("prepared compiler bytes changed")
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "source-manifest.json"), sourceManifest, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "compiler.json"), compilerJSON, 0600); err != nil {
		t.Fatal(err)
	}
	artifacts := make(map[string]*cudakernel.Artifact)
	for name := range weightPacketSources() {
		a, err := weightPacketArtifact(dir, name)
		if err != nil {
			t.Fatal(err)
		}
		if a.Toolchain != fmt.Sprintf("nvrtc %d.%d", compiler.Major, compiler.Minor) {
			t.Fatal("compiler toolchain binding differs")
		}
		artifacts[name] = a
	}
	if err := driver.Load(""); err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil || major != 12 || minor != 1 {
		t.Fatalf("packet requires SM121: %d.%d %v", major, minor, err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	}()
	ctx := context.Background()
	launches, values := 0, 0
	for _, intermediate := range []int{8, 16} {
		for _, hidden := range []int{17, 32} {
			t.Run(fmt.Sprintf("I%d-H%d", intermediate, hidden), func(t *testing.T) {
				var storage *MoEWeightStorage
				var ownedConsumers []*MoEWeightConsumer
				var ownedViews []*MoEWeightView
				var roots []cudamem.Slice[float32]
				defer func() {
					// No allocation cleanup is permitted before every queued
					// consumer has observed an actual successful drain.
					for _, c := range ownedConsumers {
						if err := c.Close(); err != nil {
							t.Error(err)
							return
						}
					}
					for _, v := range ownedViews {
						if err := v.Close(); err != nil {
							t.Error(err)
							return
						}
					}
					for _, root := range roots {
						if err := root.Free(); err != nil {
							t.Error(err)
							return
						}
					}
					if err := storage.Close(); err != nil {
						t.Error(err)
					}
				}()
				storage, err = NewMoEWeightStorage(dev, MoEWeightShape{3, 2, intermediate, hidden, []int{0, 2}})
				if err != nil {
					t.Fatal(err)
				}
				if _, err := storage.View(1, 0); err == nil {
					t.Fatal("unreadable global hole published")
				}
				copies := [2]int{}
				for index, b := range []*moeWeightBuffer{storage.weights, storage.scales} {
					original := b.write
					b.write = func(offset int, payload []byte) error { copies[index]++; return original(offset, payload) }
				}
				type allocationFact struct {
					Base                driver.DevicePtr
					Requested, Reported uint64
				}
				allocationFacts := make([]allocationFact, 2)
				err = dev.WithContext(func() error {
					for index, base := range []driver.DevicePtr{storage.weights.ptr(), storage.scales.ptr()} {
						actual, size, e := base.MemGetAddressRange()
						if e != nil {
							return e
						}
						requested := uint64(storage.layout.weightBytes)
						if index == 1 {
							requested = uint64(storage.layout.scaleRows * 4)
						}
						if actual != base || uint64(size) < requested {
							return fmt.Errorf("owned allocation is outside reported driver range")
						}
						allocationFacts[index] = allocationFact{actual, requested, uint64(size)}
					}
					return nil
				})
				if err != nil {
					t.Fatal(err)
				}
				rawByRange := map[[2]int][]byte{}
				scaleByRange := map[[2]int][]float32{}
				for _, layer := range []int{0, 2} {
					for expert := 0; expert < 2; expert++ {
						raw, scales := weightNativePayload(intermediate, hidden, layer, expert)
						if err := storage.Load(layer, expert, bytes.NewReader(raw), bytes.NewReader(weightNativeScaleBytes(scales))); err != nil {
							t.Fatal(err)
						}
						rawByRange[[2]int{layer, expert}] = raw
						scaleByRange[[2]int{layer, expert}] = scales
					}
				}
				expected := uint64(3 * 2 * 2 * intermediate * (hidden + 4))
				stats := storage.Stats()
				if stats.BaseAllocations != 2 || stats.AllocatedBytes != expected || stats.RequestedBytes != expected {
					t.Fatalf("allocation ledger: %+v", stats)
				}
				if intermediate == 16 && hidden == 17 {
					raw, scale := rawByRange[[2]int{0, 0}], scaleByRange[[2]int{0, 0}]
					write := storage.scales.write
					injected := errors.New("injected failure after partial actual scale copy")
					storage.scales.write = func(offset int, p []byte) error {
						if err := write(offset, p[:len(p)/2]); err != nil {
							return err
						}
						return injected
					}
					loadErr := storage.Load(0, 0, bytes.NewReader(raw), bytes.NewReader(weightNativeScaleBytes(scale)))
					storage.scales.write = write
					if !errors.Is(loadErr, injected) {
						t.Fatalf("partial second-copy control: %v", loadErr)
					}
					if _, err := storage.View(0, 0); err == nil {
						t.Fatal("partial scale load published ready view")
					}
					if err := storage.Load(0, 0, bytes.NewReader(raw), bytes.NewReader(weightNativeScaleBytes(scale))); err != nil {
						t.Fatal(err)
					}
					// These two explicit loading attempts are distinct from
					// session creation and are retained in the copy ledger.
					if copies != [2]int{6, 6} {
						t.Fatalf("partial/recovery copy counts %v", copies)
					}
					if err := weightPacketJSON(filepath.Join(out, "partial-load.json"), struct {
						InjectedAfterActualPartialCopy, ReadinessRefused, Recovered bool
						CopyAttempts                                                [2]int
					}{true, true, true, copies}); err != nil {
						t.Fatal(err)
					}
				}
				sessionCopyBaseline := copies
				consumers := map[string]*MoEWeightConsumer{}
				for _, name := range []string{"view", "slab"} {
					kind := MoEWeightViewConsumer
					if name == "slab" {
						kind = MoEWeightSlabConsumer
					}
					c, err := LoadMoEWeightConsumer(dev, kind, artifacts[name])
					if c != nil {
						ownedConsumers = append(ownedConsumers, c)
					}
					if err != nil {
						t.Fatal(err)
					}
					consumers[name] = c
				}
				x := make([]float32, hidden)
				for i := range x {
					x[i] = float32((i*3)%9-4) / 8
				}
				inRoot, err := cudamem.Alloc[float32](dev, hidden+2)
				if err != nil {
					t.Fatal(err)
				}
				roots = append(roots, inRoot)
				outRoot, err := cudamem.Alloc[float32](dev, intermediate+2)
				if err != nil {
					t.Fatal(err)
				}
				roots = append(roots, outRoot)
				input := inRoot.Slice(1, hidden+1)
				output := outRoot.Slice(1, intermediate+1)
				guarded := append([]float32{90123}, x...)
				guarded = append(guarded, -90123)
				if err := inRoot.CopyFromHost(guarded); err != nil {
					t.Fatal(err)
				}
				run := func(name string, c *MoEWeightConsumer, v *MoEWeightView, raw []byte, scales []float32, mutant bool) {
					t.Helper()
					poison := make([]float32, intermediate+2)
					poison[0] = 90123
					poison[len(poison)-1] = -90123
					for i := 1; i <= intermediate; i++ {
						poison[i] = -12345
					}
					if err := outRoot.CopyFromHost(poison); err != nil {
						t.Fatal(err)
					}
					if err := c.Execute(ctx, stream, v, input, output); err != nil {
						t.Fatalf("%s execution failed, not a mutant kill: %v", name, err)
					}
					launches++
					got := make([]float32, intermediate+2)
					if err := outRoot.CopyToHost(got); err != nil {
						t.Fatal(err)
					}
					observedInput := make([]float32, hidden+2)
					if err := inRoot.CopyToHost(observedInput); err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(observedInput, guarded) || got[0] != 90123 || got[len(got)-1] != -90123 {
						t.Fatal("input or guard changed")
					}
					want := weightPacketReference(raw, scales, x, intermediate)
					source := weightPacketSources()[name]
					if source == "" {
						source = weightPacketSources()["slab"]
					}
					d, err := v.Descriptor()
					if err != nil {
						t.Fatal(err)
					}
					identity := append(append(append([]byte(nil), raw...), weightNativeScaleBytes(scales)...), weightNativeScaleBytes(x)...)
					policy := cudatest.CUDAOutputPolicy{Target: "sm_121", DType: "fp32", MathMode: "manual-e4m3fn-fp32-silu-rn", SourceDigest: weightPacketDigest([]byte(source)), InputDigest: weightPacketDigest(identity), Elements: intermediate, Absolute: 1e-5, Relative: 5e-5, NaN: "reject", Infinity: "reject", Zero: "equal"}
					mismatch := cudatest.CompareCUDAOutputFP32(got[1:len(got)-1], want, policy)
					if mutant {
						var numerical *cudatest.CUDAOutputMismatch
						if !errors.As(mismatch, &numerical) {
							t.Fatalf("%s survived or invalid checker: %v", name, mismatch)
						}
					} else if mismatch != nil {
						t.Fatal(mismatch)
					}
					if name == "no-op" {
						for _, v := range got[1 : len(got)-1] {
							if v != -12345 {
								t.Fatal("no-op fixture wrote output")
							}
						}
					}
					values += intermediate
					record := struct {
						Name                            string
						Shape                           MoEWeightShape
						Descriptor                      MoEWeightDescriptor
						Raw                             []byte
						Scales, Input, Actual, Expected []float32
						Policy                          cudatest.CUDAOutputPolicy
						Image                           string
						Mutant                          bool
					}{name, storage.layout.shape, d, raw, scales, x, got[1 : len(got)-1], want, policy, artifacts["slab"].Digest, mutant}
					if a := artifacts[name]; a != nil {
						record.Image = a.Digest
					}
					path := filepath.Join(out, fmt.Sprintf("I%d-H%d-L%d-E%d-%s.json", intermediate, hidden, d.Layer, d.Expert, name))
					if err := weightPacketJSON(path, record); err != nil {
						t.Fatal(err)
					}
				}
				for _, layer := range []int{0, 2} {
					for expert := 0; expert < 2; expert++ {
						v, err := storage.View(layer, expert)
						if err != nil {
							t.Fatal(err)
						}
						ownedViews = append(ownedViews, v)
						d, err := v.Descriptor()
						if err != nil {
							t.Fatal(err)
						}
						if uint64(d.WeightBase) != uint64(storage.weights.ptr()) || uint64(d.ScaleBase) != uint64(storage.scales.ptr()) || d.WeightOffsetBytes != uint64((layer*2+expert)*2*intermediate*hidden) || d.ScaleOffsetBytes != uint64((layer*2+expert)*2*intermediate*4) {
							t.Fatal("view is not shared global offset")
						}
						raw, scales := rawByRange[[2]int{layer, expert}], scaleByRange[[2]int{layer, expert}]
						run("view", consumers["view"], v, raw, scales, false)
						run("slab", consumers["slab"], v, raw, scales, false)
						if err := v.Close(); err != nil {
							t.Fatal(err)
						}
					}
				}
				if err := consumers["view"].Close(); err != nil {
					t.Fatal(err)
				}
				v, err := storage.View(2, 1)
				if err != nil {
					t.Fatal(err)
				}
				ownedViews = append(ownedViews, v)
				raw, scales := rawByRange[[2]int{2, 1}], scaleByRange[[2]int{2, 1}]
				run("slab-after-view-close", consumers["slab"], v, raw, scales, false)
				for session := 0; session < 20; session++ {
					other, err := storage.View(0, 1)
					if err != nil {
						t.Fatal(err)
					}
					if err := other.Close(); err != nil {
						t.Fatal(err)
					}
				}
				if storage.Stats() != stats {
					t.Fatal("new sessions allocate base weights")
				}
				if copies != sessionCopyBaseline {
					t.Fatalf("new sessions reupload base weights: %v", copies)
				}
				if err := weightPacketJSON(filepath.Join(out, fmt.Sprintf("I%d-H%d-storage.json", intermediate, hidden)), struct {
					Stats        MoEWeightStorageStats
					DriverRanges []allocationFact
					CopyAttempts [2]int
					Sessions     int
				}{stats, allocationFacts, copies, 20}); err != nil {
					t.Fatal(err)
				}

				if intermediate == 16 && hidden == 17 {
					for _, name := range []string{"scale-half-split", "gate-up-swap", "wrong-global-layer", "no-op"} {
						c, err := weightNativeMutant(dev, artifacts[name])
						if c != nil {
							ownedConsumers = append(ownedConsumers, c)
						}
						if err != nil {
							t.Fatal(err)
						}
						run(name, c, v, raw, scales, true)
					}
					// Queue a real healthy read, then adopt its leases and force a canceled
					// host observation. Real stream synchronization is the only recovery.
					d, release, err := v.retain(dev)
					if err != nil {
						t.Fatal(err)
					}
					inRelease, err := input.Borrow(dev)
					if err != nil {
						release()
						t.Fatal(err)
					}
					outRelease, err := output.Borrow(dev)
					if err != nil {
						release()
						inRelease()
						t.Fatal(err)
					}
					completion, err := cudaplan.NewStreamCompletion(stream, release, inRelease, outRelease)
					if err != nil {
						release()
						inRelease()
						outRelease()
						t.Fatal(err)
					}
					c := consumers["slab"]
					c.pending = completion
					w, s, in, outPtr := d.WeightBase, d.ScaleBase, input.Ptr(), output.Ptr()
					layer, expert, experts, i, h := int32(2), int32(1), int32(2), int32(intermediate), int32(hidden)
					args := []unsafe.Pointer{unsafe.Pointer(&w), unsafe.Pointer(&s), unsafe.Pointer(&in), unsafe.Pointer(&outPtr), unsafe.Pointer(&layer), unsafe.Pointer(&expert), unsafe.Pointer(&experts), unsafe.Pointer(&i), unsafe.Pointer(&h)}
					c.failure = stream.WithHandle(func(handle driver.Stream) error {
						return c.kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{32, 1, 1}, 0, handle, args)
					})
					launches++
					canceled, cancel := context.WithCancel(ctx)
					cancel()
					if !errors.Is(c.Wait(canceled), context.Canceled) {
						t.Fatal("canceled observation accepted")
					}
					if err := v.Close(); err != nil {
						t.Fatal(err)
					}
					if !errors.Is(storage.Close(), cudamem.ErrInUse) || !errors.Is(inRoot.Free(), cudamem.ErrInUse) || !errors.Is(outRoot.Free(), cudamem.ErrInUse) {
						t.Fatal("incomplete work lost allocation retention")
					}
					if !errors.Is(storage.Load(2, 1, bytes.NewReader(raw), bytes.NewReader(weightNativeScaleBytes(scales))), cudamem.ErrInUse) {
						t.Fatal("retained range rewritten")
					}
					if err := c.Wait(ctx); err != nil {
						t.Fatal(err)
					}
					got := make([]float32, intermediate)
					if err := output.CopyToHost(got); err != nil {
						t.Fatal(err)
					}
					want := weightPacketReference(raw, scales, x, intermediate)
					for i := range got {
						if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Abs(float64(got[i]-want[i])) > 1e-5+5e-5*math.Max(math.Abs(float64(got[i])), math.Abs(float64(want[i]))) {
							t.Fatal("recovered queued read disagrees")
						}
					}
					values += intermediate
					if err := weightPacketJSON(filepath.Join(out, "lifetime.json"), struct {
						Canceled, Retained, Recovered bool
						Actual, Expected              []float32
					}{true, true, true, got, want}); err != nil {
						t.Fatal(err)
					}
				}
			})
		}
	}
	if launches != 41 || values != 512 {
		t.Fatalf("packet counts launches=%d values=%d", launches, values)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	self, err := weightPacketRead(executable, 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "executed.test"), self, 0600); err != nil {
		t.Fatal(err)
	}
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
	if err := weightPacketJSON(filepath.Join(out, "execution.json"), struct {
		Device                                 string
		UUID                                   driver.Uuid
		DriverVersion, Launches, CheckedValues int
		ExecutableSHA256                       string
	}{name, uuid, version, launches, values, weightPacketDigest(self)}); err != nil {
		t.Fatal(err)
	}
}
