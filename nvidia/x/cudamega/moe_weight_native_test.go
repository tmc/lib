//go:build native

package cudamega

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

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var moePacketConfig = MoEWeightConfig{3, 2, 16, 17}
var moePacketOptions = []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}

func moePacketSources() map[string]string {
	s, _ := MoEWeightSource(moePacketConfig, "sm_121")
	ordinary, _, _ := cudamoe.MoEWeightSource(cudamoe.MoEWeightSlabConsumer)
	return map[string]string{
		"worker": s, "ordinary": ordinary,
		"scale-half-split":   strings.ReplaceAll(strings.ReplaceAll(s, "scales[gate_row]", "scales[base+neuron]"), "scales[up_row]", "scales[base+intermediate+neuron]"),
		"gate-up-swap":       strings.Replace(s, "gate_row=base+(neuron/8)*16+neuron%8, up_row=gate_row+8", "up_row=base+(neuron/8)*16+neuron%8, gate_row=up_row+8", 1),
		"wrong-global-layer": strings.Replace(s, "(unsigned long long)layer*2+expert", "(unsigned long long)0*2+expert", 1),
		"no-op":              strings.Replace(s, "output[neuron]=", "if(false) output[neuron]=", 1),
		"unpublished":        strings.Replace(s, "atomicAdd(completed + 0, 1u)", "if(false) atomicAdd(completed + 0, 1u)", 1),
	}
}

func moePacketHash(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }
func moePacketRead(path string, limit int64) ([]byte, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	s, e := f.Stat()
	if e != nil {
		return nil, e
	}
	if !s.Mode().IsRegular() || s.Size() <= 0 || s.Size() > limit {
		return nil, errors.New("invalid packet size or kind")
	}
	b, e := io.ReadAll(io.LimitReader(f, limit+1))
	if e != nil {
		return nil, e
	}
	if len(b) == 0 || int64(len(b)) > limit {
		return nil, errors.New("packet exceeds bound")
	}
	return b, nil
}
func moePacketWrite(path string, v any) error {
	b, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}
func moePacketToolHash(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	s, e := f.Stat()
	if e != nil {
		return "", e
	}
	if !s.Mode().IsRegular() || s.Size() <= 0 || s.Size() > 512<<20 {
		return "", errors.New("invalid compiler file")
	}
	h := sha256.New()
	n, e := io.Copy(h, io.LimitReader(f, (512<<20)+1))
	if e != nil {
		return "", e
	}
	if n > 512<<20 {
		return "", errors.New("compiler file exceeds bound")
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

type moePacketCompiler struct {
	Path, SHA256, BuiltinsPath, BuiltinsSHA256 string
	Major, Minor                               int32
}

// TestBuildMoEMegaWeightPacket compiles the finite packet without driver init.
// It is preparation only; no image load or GPU execution occurs here.
func TestBuildMoEMegaWeightPacket(t *testing.T) {
	if os.Getenv("MOE_MEGA_BUILD") != "1" {
		t.Skip("offline packet not requested")
	}
	dir, path, builtins := os.Getenv("MOE_MEGA_PACKET"), os.Getenv("NVRTC_LIB_PATH"), os.Getenv("MOE_MEGA_BUILTINS")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(path) || !filepath.IsAbs(builtins) {
		t.Fatal("absolute packet/compiler/builtins paths required")
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
	hash, err := moePacketToolHash(nvrtc.Path())
	if err != nil {
		t.Fatal(err)
	}
	bh, err := moePacketToolHash(builtins)
	if err != nil {
		t.Fatal(err)
	}
	if err := moePacketWrite(filepath.Join(dir, "compiler.json"), moePacketCompiler{nvrtc.Path(), hash, builtins, bh, major, minor}); err != nil {
		t.Fatal(err)
	}
	for name, source := range moePacketSources() {
		entry, path := "moe_weight_worker", "cudamega_moe_weight.cu"
		schema, _ := MoEWeightSchema()
		if name == "ordinary" {
			entry, path = "moe_weight_slab", "cudamoe_weight_storage.cu"
			schema, _ = cudamoe.MoEWeightSchema()
		}
		cm, err := nvrtcx.Compile(source, nvrtcx.WithName(path), nvrtcx.WithPhysicalArchTarget("sm_121"), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"), nvrtcx.WithNoCache())
		if err != nil {
			var pending io.Closer
			if errors.As(err, &pending) {
				err = errors.Join(err, pending.Close())
			}
			t.Fatal(name, err)
		}
		a, err := cm.ToArtifact(entry, schema, "sm_121")
		if err != nil {
			t.Fatal(err)
		}
		a.Toolchain = fmt.Sprintf("nvrtc %d.%d", major, minor)
		a.CompilerOpts = append([]string(nil), moePacketOptions...)
		a.Dependencies = append(a.Dependencies, cudakernel.Dependency{Path: path, Tracked: true, Digest: moePacketHash([]byte(source))})
		if err := moePacketWrite(filepath.Join(dir, name+".artifact.json"), a); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".cu"), []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+".cubin"), a.Raw, 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("compiled %s source=%s image=%s", name, moePacketHash([]byte(source)), a.Digest)
	}
}

func moePacketArtifact(dir, name string) (*cudakernel.Artifact, error) {
	source, ok := moePacketSources()[name]
	if !ok {
		return nil, errors.New("unknown packet image")
	}
	b, err := moePacketRead(filepath.Join(dir, name+".artifact.json"), 16<<20)
	if err != nil {
		return nil, err
	}
	var a cudakernel.Artifact
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&a); err != nil {
		return nil, err
	}
	var extra any
	if dec.Decode(&extra) != io.EOF {
		return nil, errors.New("trailing packet JSON")
	}
	if err := a.Validate(); err != nil {
		return nil, err
	}
	cu, err := moePacketRead(filepath.Join(dir, name+".cu"), 1<<20)
	if err != nil {
		return nil, err
	}
	image, err := moePacketRead(filepath.Join(dir, name+".cubin"), 16<<20)
	if err != nil {
		return nil, err
	}
	schema, _ := MoEWeightSchema()
	entry, path := "moe_weight_worker", "cudamega_moe_weight.cu"
	if name == "ordinary" {
		schema, _ = cudamoe.MoEWeightSchema()
		entry, path = "moe_weight_slab", "cudamoe_weight_storage.cu"
	}
	if string(cu) != source || !bytes.Equal(image, a.Raw) || a.Kind != cudakernel.KindCUBIN || a.Target != "sm_121" || a.EntryPoint != entry || !reflect.DeepEqual(a.Schema, schema) || !reflect.DeepEqual(a.CompilerOpts, moePacketOptions) {
		return nil, errors.New("packet source/image/ABI/options differ")
	}
	count := 0
	for _, d := range a.Dependencies {
		if d.Path == path {
			count++
			if !d.Tracked || d.ErrorMsg != "" || d.Digest != moePacketHash(cu) {
				return nil, errors.New("source dependency differs")
			}
		}
	}
	if count != 1 {
		return nil, errors.New("missing or repeated source dependency")
	}
	return &a, nil
}

func moeNativePayload(layer, expert int) ([]byte, []float32, []float32) {
	const I, H = 16, 17
	choices := []byte{0x38, 0x3c, 0x40, 0xb8, 0xbc, 1, 0x80, 0}
	raw := make([]byte, 2*I*H)
	for i := range raw {
		raw[i] = choices[(i+i/H+layer*3+expert)%len(choices)]
	}
	s := make([]float32, 2*I)
	for i := range s {
		s[i] = float32(1+(i*3+layer*5+expert*7)%11) / 64
	}
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i*3)%9-4) / 8
	}
	return raw, s, x
}
func moeNativeScales(s []float32) []byte {
	b := make([]byte, 4*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// Reference enumerates logical half-split rows in FP64, independently of the
// worker's interleaved addresses and FP32 recurrence. No host device substitute.
func moeNativeReference(raw []byte, s, x []float32) []float32 {
	decode := func(b byte) float64 {
		e, m := int(b&127)>>3, int(b&7)
		v := float64(m) / 512
		if e != 0 {
			v = math.Ldexp(1+float64(m)/8, e-7)
		}
		if b&128 != 0 {
			v = -v
		}
		return v
	}
	out := make([]float32, 16)
	for n := range out {
		g, u := 0.0, 0.0
		for j, v := range x {
			g += decode(raw[n*len(x)+j]) * float64(v)
			u += decode(raw[(16+n)*len(x)+j]) * float64(v)
		}
		g *= float64(s[n])
		u *= float64(s[16+n])
		out[n] = float32(g / (1 + math.Exp(-g)) * u)
	}
	return out
}

func TestMoEMegaWeightPacketSources(t *testing.T) {
	s := moePacketSources()
	if len(s) != 7 {
		t.Fatal("wrong image count")
	}
	for n, v := range s {
		if n != "worker" && n != "ordinary" && v == s["worker"] {
			t.Fatal("vacuous compiled mutant", n)
		}
	}
	raw, sc, x := moeNativePayload(2, 1)
	got := moeNativeReference(raw, sc, x)
	for _, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v == -12345 {
			t.Fatal("vacuous reference")
		}
	}
}

func moeNativeRead[T any](t *testing.T, s cudamem.Slice[T]) []T {
	t.Helper()
	v := make([]T, s.Len())
	if err := s.CopyToHost(v); err != nil {
		t.Fatal(err)
	}
	return v
}

func TestNativeMoEMegaSharedWeights(t *testing.T) {
	if os.Getenv("MOE_MEGA_NATIVE") != "1" {
		t.Skip("scoped native packet not requested")
	}
	pid, e := strconv.Atoi(os.Getenv("GPU_LOCK_HELD"))
	if e != nil || pid <= 0 || os.Getenv("GPU_LOCK_DISABLED") != "" || os.Getenv("CUDATEST_NO_LOCK") != "" {
		t.Fatal("real inherited gpu-lock required")
	}
	dir, out := os.Getenv("MOE_MEGA_PACKET"), os.Getenv("MOE_MEGA_RESULTS")
	if !filepath.IsAbs(dir) || !filepath.IsAbs(out) {
		t.Fatal("absolute packet/results required")
	}
	if err := os.MkdirAll(out, 0700); err != nil {
		t.Fatal(err)
	}
	manifest, err := moePacketRead(os.Getenv("MOE_MEGA_SOURCE_MANIFEST"), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "source-manifest.json"), manifest, 0600); err != nil {
		t.Fatal(err)
	}
	cb, err := moePacketRead(filepath.Join(dir, "compiler.json"), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var compiler moePacketCompiler
	dec := json.NewDecoder(bytes.NewReader(cb))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&compiler); err != nil {
		t.Fatal(err)
	}
	var extra any
	if dec.Decode(&extra) != io.EOF {
		t.Fatal("trailing compiler JSON")
	}
	for _, v := range []struct{ path, hash string }{{compiler.Path, compiler.SHA256}, {compiler.BuiltinsPath, compiler.BuiltinsSHA256}} {
		if !filepath.IsAbs(v.path) {
			t.Fatal("relative compiler")
		}
		h, err := moePacketToolHash(v.path)
		if err != nil || h != v.hash {
			t.Fatal("compiler bytes differ", err)
		}
	}
	if compiler.Major != 13 || compiler.Minor != 0 {
		t.Fatal("packet compiler differs")
	}
	if err := os.WriteFile(filepath.Join(out, "compiler.json"), cb, 0600); err != nil {
		t.Fatal(err)
	}
	images := map[string]*cudakernel.Artifact{}
	for name := range moePacketSources() {
		a, err := moePacketArtifact(dir, name)
		if err != nil {
			t.Fatal(name, err)
		}
		if a.Toolchain != "nvrtc 13.0" {
			t.Fatal("artifact compiler differs")
		}
		images[name] = a
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
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	}()
	storage, err := cudamoe.NewMoEWeightStorage(dev, cudamoe.MoEWeightShape{Layers: 3, Experts: 2, Intermediate: 16, Hidden: 17, ActiveLayers: []int{0, 2}})
	var owners []*MoEWeightConsumer
	var ordinary *cudamoe.MoEWeightConsumer
	var views []*cudamoe.MoEWeightView
	var frees []func() error
	defer func() {
		// Drain consumers before releasing their public views and allocations.
		// A failed drain leaves the independent pins intact.
		for _, c := range owners {
			if err := c.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if ordinary != nil {
			if err := ordinary.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		for _, v := range views {
			if err := v.Close(); err != nil {
				t.Error(err)
			}
		}
		for _, free := range frees {
			if err := free(); err != nil {
				t.Error(err)
			}
		}
		if storage != nil {
			if err := storage.Close(); err != nil {
				t.Error(err)
			}
		}
	}()
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err = cudamoe.LoadMoEWeightConsumer(dev, cudamoe.MoEWeightSlabConsumer, images["ordinary"])
	if err != nil {
		t.Fatal(err)
	}
	worker, err := LoadMoEWeightConsumer(dev, moePacketConfig, images["worker"])
	if worker != nil {
		owners = append(owners, worker)
	}
	if err != nil {
		t.Fatal(err)
	}
	inputRoot, err := cudamem.Alloc[float32](dev, 19)
	if err != nil {
		t.Fatal(err)
	}
	frees = append(frees, inputRoot.Free)
	outputRoot, err := cudamem.Alloc[float32](dev, 18)
	if err != nil {
		t.Fatal(err)
	}
	frees = append(frees, outputRoot.Free)
	completed, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	frees = append(frees, completed.Free)
	abort, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	frees = append(frees, abort.Free)
	order, err := cudamem.Alloc[uint64](dev, 2)
	if err != nil {
		t.Fatal(err)
	}
	frees = append(frees, order.Free)
	b := MoEWeightBuffers{Input: inputRoot.Slice(1, 18), Output: outputRoot.Slice(1, 17), Completed: completed, Abort: abort, StartOrder: order}
	ctx := context.Background()
	launches, values := 0, 0
	stats := storage.Stats()
	if stats.BaseAllocations != 2 || stats.AllocatedBytes != 4032 {
		t.Fatal("shared allocation ledger differs")
	}
	var last *cudamoe.MoEWeightView
	var lastRaw []byte
	var lastScales, lastInput []float32
	epoch := uint64(1)
	poison := func() {
		v := make([]float32, 18)
		v[0] = 90123
		v[17] = -90123
		for i := 1; i < 17; i++ {
			v[i] = -12345
		}
		if err := outputRoot.CopyFromHost(v); err != nil {
			t.Fatal(err)
		}
	}
	record := func(name string, v *cudamoe.MoEWeightView, raw []byte, sc, x []float32, image *cudakernel.Artifact, mutant bool, publication bool) {
		t.Helper()
		got := moeNativeRead(t, outputRoot)
		in := moeNativeRead(t, inputRoot)
		if got[0] != 90123 || got[17] != -90123 || in[0] != 90123 || in[18] != -90123 || !reflect.DeepEqual(in[1:18], x) {
			t.Fatal("guard/input changed")
		}
		expected := moeNativeReference(raw, sc, x)
		policy := cudatest.CUDAOutputPolicy{Target: "sm_121", DType: "fp32", MathMode: "manual-e4m3fn-fp32-silu-rn", SourceDigest: "", InputDigest: moePacketHash(append(append(append([]byte(nil), raw...), moeNativeScales(sc)...), moeNativeScales(x)...)), Elements: 16, Absolute: 1e-5, Relative: 5e-5, NaN: "reject", Infinity: "reject", Zero: "equal"}
		for _, d := range image.Dependencies {
			if d.Path == "cudamega_moe_weight.cu" || d.Path == "cudamoe_weight_storage.cu" {
				policy.SourceDigest = d.Digest
			}
		}
		mismatch := cudatest.CompareCUDAOutputFP32(got[1:17], expected, policy)
		if mutant {
			var numerical *cudatest.CUDAOutputMismatch
			if !errors.As(mismatch, &numerical) {
				t.Fatal("mutant survived or invalid checker", name, mismatch)
			}
		} else if mismatch != nil {
			t.Fatal(name, mismatch)
		}
		desc, err := v.Descriptor()
		if err != nil {
			t.Fatal(err)
		}
		values += 16
		data := struct {
			Name                                           string
			Descriptor                                     cudamoe.MoEWeightDescriptor
			Raw                                            []byte
			Scales, Input, Actual, Expected, GuardedOutput []float32
			Completed, Abort                               []uint32
			StartOrder                                     []uint64
			Image                                          string
			Policy                                         cudatest.CUDAOutputPolicy
			Mutant, Publication                            bool
		}{name, desc, raw, sc, x, got[1:17], expected, got, moeNativeRead(t, completed), moeNativeRead(t, abort), moeNativeRead(t, order), image.Digest, policy, mutant, publication}
		if err := moePacketWrite(filepath.Join(out, name+".json"), data); err != nil {
			t.Fatal(err)
		}
	}
	for _, layer := range []int{0, 2} {
		for expert := 0; expert < 2; expert++ {
			raw, sc, x := moeNativePayload(layer, expert)
			if err := storage.Load(layer, expert, bytes.NewReader(raw), bytes.NewReader(moeNativeScales(sc))); err != nil {
				t.Fatal(err)
			}
			v, err := storage.View(layer, expert)
			if err != nil {
				t.Fatal(err)
			}
			views = append(views, v)
			guarded := append(append([]float32{90123}, x...), -90123)
			if err := inputRoot.CopyFromHost(guarded); err != nil {
				t.Fatal(err)
			}
			poison()
			if err := ordinary.Execute(ctx, stream, v, b.Input, b.Output); err != nil {
				t.Fatal(err)
			}
			launches++
			record(fmt.Sprintf("L%d-E%d-ordinary", layer, expert), v, raw, sc, x, images["ordinary"], false, false)
			poison()
			if err := worker.Execute(ctx, stream, v, b, epoch); err != nil {
				t.Fatal(err)
			}
			epoch++
			launches++
			record(fmt.Sprintf("L%d-E%d-worker", layer, expert), v, raw, sc, x, images["worker"], false, true)
			last, lastRaw, lastScales, lastInput = v, raw, sc, x
		}
	}
	if err := ordinary.Close(); err != nil {
		t.Fatal(err)
	}
	poison()
	if err := worker.Execute(ctx, stream, last, b, epoch); err != nil {
		t.Fatal(err)
	}
	epoch++
	launches++
	record("worker-after-ordinary-close", last, lastRaw, lastScales, lastInput, images["worker"], false, true)
	for _, name := range []string{"scale-half-split", "gate-up-swap", "wrong-global-layer", "no-op", "unpublished"} {
		if _, err := checkedMoEArtifact(moePacketConfig, images[name]); err == nil {
			t.Fatal("public loader admitted mutant")
		}
		c, err := loadMoEWeightImage(dev, moePacketConfig, images[name])
		if c != nil {
			owners = append(owners, c)
		}
		if err != nil {
			t.Fatal(err)
		}
		poison()
		err = c.Execute(ctx, stream, last, b, epoch)
		epoch++
		launches++
		if name == "unpublished" {
			if err == nil || !strings.Contains(err.Error(), "publication incomplete") {
				t.Fatal("missing publication accepted or unrelated failure", err)
			}
		} else if err != nil {
			t.Fatal("mutant execution failed, not numerical disagreement", err)
		}
		record(name, last, lastRaw, lastScales, lastInput, images[name], name != "unpublished", name != "unpublished")
	}
	poison()
	if err := worker.Submit(stream, last, b, epoch); err != nil {
		t.Fatal(err)
	}
	launches++
	if !errors.Is(worker.Submit(stream, last, b, epoch), cudamem.ErrInUse) {
		t.Fatal("pending reuse accepted")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if !errors.Is(worker.Wait(cancelled), context.Canceled) {
		t.Fatal("canceled wait completed")
	}
	if err := last.Close(); err != nil {
		t.Fatal(err)
	}
	if err := storage.Load(2, 1, bytes.NewReader(lastRaw), bytes.NewReader(moeNativeScales(lastScales))); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatal("closed public range lost independent pending retention", err)
	}
	if !errors.Is(storage.Close(), cudamem.ErrInUse) || !errors.Is(inputRoot.Free(), cudamem.ErrInUse) || !errors.Is(outputRoot.Free(), cudamem.ErrInUse) {
		t.Fatal("pending pins lost")
	}
	if err := worker.Wait(ctx); err != nil {
		t.Fatal(err)
	}
	if err := checkMoEPublication(b); err != nil {
		t.Fatal(err)
	}
	// Reacquire the ready descriptor only after the independently retained
	// queued consumer has drained. The closed public view remains closed.
	last, err = storage.View(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	views = append(views, last)
	record("recovered-canceled-read", last, lastRaw, lastScales, lastInput, images["worker"], false, true)
	if err := last.Close(); err != nil {
		t.Fatal(err)
	}
	if err := storage.Load(2, 1, bytes.NewReader(lastRaw), bytes.NewReader(moeNativeScales(lastScales))); err != nil {
		t.Fatal("drained range did not permit reload", err)
	}
	if storage.Stats() != stats {
		t.Fatal("Mega consumer restacked shared weights")
	}
	if launches != 15 || values != 240 {
		t.Fatal("packet count differs", launches, values)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	binary, err := moePacketRead(self, 32<<20)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "executed.test"), binary, 0600); err != nil {
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
	if err := moePacketWrite(filepath.Join(out, "execution.json"), struct {
		Device                                 string
		UUID                                   driver.Uuid
		DriverVersion, Launches, CheckedValues int
		BlocksPerSM                            int32
		ExecutableSHA256                       string
		Storage                                cudamoe.MoEWeightStorageStats
	}{name, uuid, version, launches, values, worker.BlocksPerSM(), moePacketHash(binary), stats}); err != nil {
		t.Fatal(err)
	}
}
