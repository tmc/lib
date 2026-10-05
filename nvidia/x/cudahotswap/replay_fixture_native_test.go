//go:build native

package cudahotswap

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
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const replayPacketCompiler = "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc.so.13.0.88"
const replayPacketBuiltins = "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvrtc-builtins.so.13.0.88"
const replayPacketCompilerHash = "64db9c3fc128f4a57a22c314fda08831a8e3a725d439a70076a0fb98f7bea1b7"
const replayPacketBuiltinsHash = "84f1c5031bfe1146375b367b2c7fb8d31ea00430be7172177eeb5d04ab813b69"
const replayPacketGuard uint32 = 0x4b7e6319

var replayPacketFlags = []string{"--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false"}

type replayPacketCase struct {
	Name, Baseline, Candidate, Outcome string
	Launches                           int
}

func replayPacketCases() []replayPacketCase {
	return []replayPacketCase{
		{"correct", "baseline", "correct", "pass", 13},
		{"wrong_math", "baseline", "wrong", "wrong_answer", 13},
		{"nonfinite", "baseline", "nonfinite", "wrong_answer", 13},
		{"inplace", "inplace_baseline", "inplace_correct", "pass", 13},
		{"noop", "noop", "", "unchanged_baseline", 2},
		{"noise", "noisy", "", "baseline_noise", 2},
		{"stable_write", "baseline", "stable_write", "stable_write_refusal", 5},
		{"cleanup_retry", "baseline", "correct", "injected_release_retry", 13},
	}
}

func replayPacketSources() map[string]string {
	bodies := map[string]string{
		"baseline":         "y[i] = x[i] + w[i]; x[i] += 100.0f;",
		"correct":          "y[i] = w[i] + x[i]; x[i] = 100.0f + x[i];",
		"wrong":            "y[i] = x[i] + w[i] + 1.0f; x[i] += 100.0f;",
		"nonfinite":        "y[i] = i == 3 ? __int_as_float(0x7f800000) : x[i] + w[i]; x[i] += 100.0f;",
		"inplace_baseline": "y[i] = y[i] + w[i]; x[i] += 100.0f;",
		"inplace_correct":  "y[i] = w[i] + y[i]; x[i] = 100.0f + x[i];",
		"noop":             "",
		"stable_write":     "y[i] = x[i] + w[i]; x[i] += 100.0f; const_cast<float*>(w)[i] += 7.0f;",
	}
	out := make(map[string]string)
	for name, body := range bodies {
		out[name] = "extern \"C\" __global__ void replay(float* x, float* y, const float* w, unsigned n) { unsigned i = threadIdx.x; if (i < n) { " + body + " } }\n"
	}
	// A separate negative-control module deliberately has persistent state.
	// One block and its barrier make successive launch counters deterministic.
	out["noisy"] = "__device__ unsigned replay_calls = 0;\nextern \"C\" __global__ void replay(float* x, float* y, const float* w, unsigned n) { __shared__ unsigned step; if (threadIdx.x == 0) step = atomicAdd(&replay_calls, 1u); __syncthreads(); unsigned i = threadIdx.x; if (i < n) { y[i] = x[i] + w[i] + float(step); x[i] += 100.0f; } }\n"
	return out
}

type replayPacketImage struct {
	Name, Source, PTX, CUBIN       string
	SourceHash, PTXHash, ImageHash string
	Signature                      cudalaunch.Signature
	CompilerKey                    string
}

type replayPacketManifest struct {
	Version, Elements, Rounds, Launches                                   int
	Target, Entry, CompilerPath, CompilerHash, BuiltinsPath, BuiltinsHash string
	CompilerMajor, CompilerMinor                                          int
	Flags                                                                 []string
	SourcePins                                                            map[string]string
	Cases                                                                 []replayPacketCase
	Images                                                                []replayPacketImage
}

func replayPacketDigest(b []byte) string { h := sha256.Sum256(b); return fmt.Sprintf("%x", h[:]) }

func replayPacketFile(path string, limit int64) ([]byte, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, fmt.Errorf("packet path must be canonical and absolute")
	}
	for parent := filepath.Dir(path); ; parent = filepath.Dir(parent) {
		info, err := os.Lstat(parent)
		if err != nil {
			return nil, err
		}
		if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("packet parent is not a real directory")
		}
		if parent == filepath.Dir(parent) {
			break
		}
	}
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > limit {
		return nil, fmt.Errorf("packet file kind or size refused")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !os.SameFile(info, opened) || !opened.Mode().IsRegular() {
		return nil, fmt.Errorf("packet opened object changed")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if len(b) == 0 || int64(len(b)) > limit {
		return nil, fmt.Errorf("packet grew beyond bound")
	}
	return b, nil
}

func replayPacketSourcePins(root string) (map[string]string, error) {
	if !filepath.IsAbs(root) {
		return nil, fmt.Errorf("explicit source root required")
	}
	pins := make(map[string]string)
	for _, name := range []string{"x/cudahotswap/replay_fixture.go", "x/cudahotswap/replay_fixture_test.go", "x/cudahotswap/replay_owner_test.go", "x/cudahotswap/replay_fixture_native_test.go", "x/cudahotswap/loader.go", "x/cudahotswap/replay_compare.go"} {
		b, err := replayPacketFile(filepath.Join(root, name), 1<<20)
		if err != nil {
			return nil, err
		}
		pins[name] = replayPacketDigest(b)
	}
	return pins, nil
}

func replayPacketABI(s cudalaunch.Signature) error {
	if s.KernelName != "replay" || len(s.Params) != 4 {
		return fmt.Errorf("packet entry or argument count refused")
	}
	for i, p := range s.Params {
		width := 8
		if i == 3 {
			width = 4
		}
		if p.Size != width || p.IsPointer || (p.Type != "u64" && i < 3) || (p.Type != "u32" && i == 3) {
			return fmt.Errorf("packet actual PTX ABI refused")
		}
	}
	return nil
}

func TestBuildHotSwapReplayPacket(t *testing.T) {
	if os.Getenv("HOTSWAP_REPLAY_BUILD_ONLY") != "1" {
		t.Skip("offline preparation not requested")
	}
	dir, root := os.Getenv("HOTSWAP_REPLAY_PACKET_DIR"), os.Getenv("HOTSWAP_REPLAY_SOURCE_ROOT")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute packet directory required")
	}
	pins, err := replayPacketSourcePins(root)
	if err != nil {
		t.Fatal(err)
	}
	for path, want := range map[string]string{replayPacketCompiler: replayPacketCompilerHash, replayPacketBuiltins: replayPacketBuiltinsHash} {
		b, err := replayPacketFile(path, 128<<20)
		if err != nil || replayPacketDigest(b) != want {
			t.Fatalf("compiler pin refused: %s %v", path, err)
		}
	}
	if err := nvrtc.Load(replayPacketCompiler); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil || major != 13 || minor != 0 {
		t.Fatalf("NVRTC13.0 required: %d.%d %v", major, minor, err)
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	m := replayPacketManifest{Version: 1, Elements: 8, Rounds: 5, Launches: 74, Target: "sm_121", Entry: "replay", CompilerPath: replayPacketCompiler, CompilerHash: replayPacketCompilerHash, BuiltinsPath: replayPacketBuiltins, BuiltinsHash: replayPacketBuiltinsHash, CompilerMajor: int(major), CompilerMinor: int(minor), Flags: append([]string(nil), replayPacketFlags...), SourcePins: pins, Cases: replayPacketCases()}
	sources := replayPacketSources()
	for _, name := range []string{"baseline", "correct", "wrong", "nonfinite", "noop", "inplace_baseline", "inplace_correct", "noisy", "stable_write"} {
		source := sources[name]
		cm, err := nvrtcx.Compile(source, nvrtcx.WithName(name+".cu"), nvrtcx.WithPhysicalArchTarget("sm_121"), nvrtcx.WithFlag("--std=c++17"), nvrtcx.WithFlag("--fmad=false"), nvrtcx.WithFlag("--ftz=false"), nvrtcx.WithNoLineInfo(), nvrtcx.WithNoCache())
		if err != nil {
			t.Fatal(err)
		}
		ptx, image := []byte(cm.PTX()), cm.CUBIN()
		sig, err := cudalaunch.SignatureFromPTX(string(ptx), "replay")
		if err != nil {
			t.Fatal(err)
		}
		if err := replayPacketABI(sig); err != nil {
			t.Fatal(err)
		}
		if len(image) < 64 || !bytes.HasPrefix(image, []byte{0x7f, 'E', 'L', 'F'}) {
			t.Fatal("compiler did not produce CUBIN")
		}
		entry := replayPacketImage{Name: name, Source: name + ".cu", PTX: name + ".ptx", CUBIN: name + ".cubin", SourceHash: replayPacketDigest([]byte(source)), PTXHash: replayPacketDigest(ptx), ImageHash: replayPacketDigest(image), Signature: sig, CompilerKey: cm.Key()}
		for suffix, data := range map[string][]byte{".cu": []byte(source), ".ptx": ptx, ".cubin": image} {
			if err := os.WriteFile(filepath.Join(dir, name+suffix), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		m.Images = append(m.Images, entry)
		t.Logf("prepared name=%s source=%s image=%s key=%s", name, entry.SourceHash, entry.ImageHash, entry.CompilerKey)
	}
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func replayReadPacket(dir, root string, tools bool) (replayPacketManifest, map[string][]byte, error) {
	var m replayPacketManifest
	b, err := replayPacketFile(filepath.Join(dir, "manifest.json"), 1<<20)
	if err != nil {
		return m, nil, err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err = d.Decode(&m); err != nil {
		return m, nil, err
	}
	if d.Decode(new(any)) != io.EOF {
		return m, nil, fmt.Errorf("trailing packet JSON")
	}
	if m.Version != 1 || m.Elements != 8 || m.Rounds != 5 || m.Launches != 74 || m.Target != "sm_121" || m.Entry != "replay" || m.CompilerMajor != 13 || m.CompilerMinor != 0 || m.CompilerPath != replayPacketCompiler || m.CompilerHash != replayPacketCompilerHash || m.BuiltinsPath != replayPacketBuiltins || m.BuiltinsHash != replayPacketBuiltinsHash || !reflect.DeepEqual(m.Flags, replayPacketFlags) || !reflect.DeepEqual(m.Cases, replayPacketCases()) {
		return m, nil, fmt.Errorf("packet policy/compiler identity disagrees")
	}
	pins, err := replayPacketSourcePins(root)
	if err != nil {
		return m, nil, err
	}
	if !reflect.DeepEqual(pins, m.SourcePins) {
		return m, nil, fmt.Errorf("packet actual source pins disagree")
	}
	if tools {
		for path, want := range map[string]string{m.CompilerPath: m.CompilerHash, m.BuiltinsPath: m.BuiltinsHash} {
			b, err := replayPacketFile(path, 128<<20)
			if err != nil || replayPacketDigest(b) != want {
				return m, nil, fmt.Errorf("packet actual compiler library differs: %s %v", path, err)
			}
		}
	}
	sources := replayPacketSources()
	if len(m.Images) != len(sources) {
		return m, nil, fmt.Errorf("packet image inventory differs")
	}
	images := make(map[string][]byte)
	expected := map[string]bool{"manifest.json": true}
	for _, entry := range m.Images {
		source, ok := sources[entry.Name]
		if !ok || images[entry.Name] != nil || entry.Source != entry.Name+".cu" || entry.PTX != entry.Name+".ptx" || entry.CUBIN != entry.Name+".cubin" || entry.SourceHash != replayPacketDigest([]byte(source)) || len(entry.CompilerKey) != 64 {
			return m, nil, fmt.Errorf("packet image source/identity refused")
		}
		for _, name := range []string{entry.Source, entry.PTX, entry.CUBIN} {
			expected[name] = true
		}
		cu, err := replayPacketFile(filepath.Join(dir, entry.Source), 1<<20)
		if err != nil || !bytes.Equal(cu, []byte(source)) {
			return m, nil, fmt.Errorf("packet source bytes disagree: %v", err)
		}
		ptx, err := replayPacketFile(filepath.Join(dir, entry.PTX), 4<<20)
		if err != nil || replayPacketDigest(ptx) != entry.PTXHash {
			return m, nil, fmt.Errorf("packet PTX pin disagrees: %v", err)
		}
		sig, err := cudalaunch.SignatureFromPTX(string(ptx), "replay")
		if err != nil || replayPacketABI(sig) != nil || !reflect.DeepEqual(sig, entry.Signature) {
			return m, nil, fmt.Errorf("packet PTX schema disagrees")
		}
		image, err := replayPacketFile(filepath.Join(dir, entry.CUBIN), 4<<20)
		if err != nil || len(image) < 64 || !bytes.HasPrefix(image, []byte{0x7f, 'E', 'L', 'F'}) || replayPacketDigest(image) != entry.ImageHash {
			return m, nil, fmt.Errorf("packet CUBIN pin disagrees: %v", err)
		}
		images[entry.Name] = image
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		return m, nil, err
	}
	if len(files) != len(expected) {
		return m, nil, fmt.Errorf("packet file inventory differs")
	}
	for _, f := range files {
		if !expected[f.Name()] || f.IsDir() {
			return m, nil, fmt.Errorf("packet unexpected file")
		}
	}
	return m, images, nil
}

// The raw partial module and its owned context wrapper are adopted before every
// error check. Failed Close leaves the actual remaining owner reachable.
type replayPacketOwner struct {
	dev    *cuda.Device
	raw    driver.Module
	module *cuda.Module
	kernel *nativeKernel
}

func replayLoadPacketOwner(dev *cuda.Device, image []byte, sig cudalaunch.Signature) (*replayPacketOwner, error) {
	o := &replayPacketOwner{dev: dev}
	err := dev.WithContext(func() error { var err error; o.raw, err = driver.ModuleLoadData(image); return err })
	if err != nil {
		return o, err
	}
	module, err := cuda.OwnModule(dev, o.raw)
	if module != nil {
		o.module = module
		o.raw = 0
	}
	if err != nil {
		return o, err
	}
	o.kernel = &nativeKernel{dev: dev, module: module, signature: sig}
	o.kernel.function, err = module.Function("replay")
	return o, err
}

func (o *replayPacketOwner) Close() error {
	if o.module != nil {
		if err := o.module.Close(); err != nil {
			return err
		}
		o.module = nil
	}
	if o.raw != 0 {
		return o.dev.WithContext(func() error { return o.raw.Close() })
	}
	return nil
}

type replayNativeRun struct {
	Kernel        string
	Timed         bool
	EventNanos    int64
	Function      uint64
	Args          []uint64
	Buffers       map[int][]uint32
	CallerGuarded map[int][]uint32
}

type replayNativeObserver struct {
	*cudaReplayBackend
	guards map[int]cudamem.Slice[byte]
	names  map[Kernel]string
	runs   []replayNativeRun
	emit   func(any) error
}

func replayNativeBits(data []byte) []uint32 {
	out := make([]uint32, len(data)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(data[4*i:])
	}
	return out
}

func (o *replayNativeObserver) Run(ctx context.Context, arm replayArm, spec replaySpec, timed bool) (time.Duration, error) {
	kernel, ok := arm.Kernel.(Kernel)
	if !ok {
		return 0, fmt.Errorf("native kernel observer requires a kernel callable")
	}
	if o.emit != nil {
		if err := o.emit(map[string]any{"kind": "attempt", "index": len(o.runs), "kernel": o.names[kernel], "timed": timed}); err != nil {
			return 0, err
		}
	}
	d, err := o.cudaReplayBackend.Run(ctx, arm, spec, timed)
	if err != nil {
		if o.emit != nil {
			_ = o.emit(map[string]any{"kind": "submission_or_completion_error", "index": len(o.runs), "kernel": o.names[kernel], "error": err.Error(), "queued": "unknown"})
		}
		return d, err
	}
	r := replayNativeRun{Kernel: o.names[kernel], Timed: timed, EventNanos: -1, Function: uint64(kernel.Raw()), Buffers: make(map[int][]uint32), CallerGuarded: make(map[int][]uint32)}
	if timed {
		r.EventNanos = d.Nanoseconds()
	}
	for _, decl := range spec.Buffers {
		data, err := o.Read(decl.Arg)
		if err != nil {
			return d, err
		}
		r.Buffers[decl.Arg] = replayNativeBits(data)
		s := o.bindings[decl.Arg]
		if !decl.Stable {
			s = o.scratch[decl.Arg]
		}
		r.Args = append(r.Args, uint64(s.Ptr()))
		guarded := o.guards[decl.Arg]
		whole := make([]byte, guarded.Len())
		if err := guarded.CopyToHost(whole); err != nil {
			return d, err
		}
		bits := replayNativeBits(whole)
		r.CallerGuarded[decl.Arg] = bits
		if len(bits) != 10 || bits[0] != replayPacketGuard || bits[9] != replayPacketGuard {
			return d, fmt.Errorf("native caller redzone changed")
		}
	}
	// Scalar and geometry are fixed in the packet; pointers are actual runtime
	// bindings. Stable Arg2 must retain the caller's original device address.
	r.Args = append(r.Args, uint64(spec.Args[3].(uint32)))
	o.runs = append(o.runs, r)
	if o.emit != nil {
		if err := o.emit(map[string]any{"kind": "arm", "index": len(o.runs) - 1, "run": r}); err != nil {
			return d, err
		}
	}
	if err := replayVerifyNativeRun(r, len(o.runs)-1); err != nil {
		return d, err
	}
	return d, nil
}

// FP64 equations over separately specified integer inputs validate every
// actual element. They do not reuse the baseline's recorded comparison vector.
func replayVerifyNativeRun(run replayNativeRun, index int) error {
	for arg := 0; arg < 3; arg++ {
		if len(run.Buffers[arg]) != 8 {
			return fmt.Errorf("native full readback extent differs")
		}
	}
	for i := 0; i < 8; i++ {
		x, w, y := float64(i+2), float64(5+2*i), float64(-8-i)
		wantX, wantW, wantY := x+100, w, x+w
		poison := false
		switch run.Kernel {
		case "noop":
			wantX, wantY = x, y
		case "wrong":
			wantY++
		case "nonfinite":
			if i == 3 {
				wantY = math.Inf(1)
			}
		case "inplace_baseline", "inplace_correct":
			wantY = y + w
			poison = !run.Timed && index == 2
		case "noisy":
			wantY += float64(index)
		case "stable_write":
			wantW += 7
		}
		if run.Buffers[0][i] != math.Float32bits(float32(wantX)) || run.Buffers[2][i] != math.Float32bits(float32(wantW)) {
			return fmt.Errorf("native input restoration/stable effect differs at element %d", i)
		}
		if poison {
			if !math.IsNaN(float64(math.Float32frombits(run.Buffers[1][i]))) {
				return fmt.Errorf("native in-place poison did not propagate")
			}
		} else if run.Buffers[1][i] != math.Float32bits(float32(wantY)) {
			return fmt.Errorf("native output differs from FP64 equation at element %d", i)
		}
		for arg, bits := range run.CallerGuarded {
			value := float64(i + 2)
			if arg == 1 {
				value = y
			}
			if arg == 2 {
				value = wantW
			}
			if len(bits) != 10 || bits[i+1] != math.Float32bits(float32(value)) {
				return fmt.Errorf("native original caller bytes differ for argument %d element %d", arg, i)
			}
		}
	}
	return nil
}

type replayNativeResources struct {
	dev      *cuda.Device
	lease    *cudalock.Handle
	fixtures []*replayFixture
	modules  []*replayPacketOwner
	buffers  []cudamem.Slice[byte]
	streams  []*cuda.Stream
}

func (r *replayNativeResources) Close(ctx context.Context) error {
	for _, f := range r.fixtures {
		if err := f.Close(ctx); err != nil {
			return err
		}
	}
	for _, m := range r.modules {
		if err := m.Close(); err != nil {
			return err
		}
	}
	for _, s := range r.buffers {
		if err := s.Free(); err != nil {
			return err
		}
	}
	for _, s := range r.streams {
		if err := s.Close(); err != nil {
			return err
		}
	}
	if r.dev != nil {
		if err := r.dev.Close(); err != nil {
			return err
		}
		r.dev = nil
	}
	if r.lease != nil {
		if err := r.lease.Release(); err != nil {
			return err
		}
		r.lease = nil
	}
	return nil
}

func replayNativeSpec(sig cudalaunch.Signature, stream *cuda.Stream) replaySpec {
	return replaySpec{Signature: sig, Args: []any{nil, nil, nil, uint32(8)}, Buffers: []replayBuffer{{Arg: 0, Bytes: 32}, {Arg: 1, Bytes: 32}, {Arg: 2, Bytes: 32, Stable: true}}, Outputs: []replayOutput{{Arg: 1, Elem: "f32"}}, Config: cudalaunch.Config{Grid: cudashape.Grid3D{X: 1, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: stream}}
}

func TestNativeHotSwapReplay(t *testing.T) {
	if os.Getenv("HOTSWAP_REPLAY_NATIVE") != "1" {
		t.Skip("physical native qualification not requested")
	}
	dir, root := os.Getenv("HOTSWAP_REPLAY_PACKET_DIR"), os.Getenv("HOTSWAP_REPLAY_SOURCE_ROOT")
	manifest, images, err := replayReadPacket(dir, root, true)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	r := &replayNativeResources{}
	t.Cleanup(func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		if err := r.Close(cleanupCtx); err != nil {
			t.Errorf("unresolved native owners/lease retained: %v", err)
		}
	})
	r.lease, err = cudalock.Acquire(ctx, cudalock.WithLabel("lane61-replay-native"))
	if err != nil {
		t.Fatal(err)
	}
	r.dev, err = cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := r.dev.ComputeCapability()
	if err != nil || major != 12 || minor != 1 {
		t.Fatalf("sm121 device required: %d.%d %v", major, minor, err)
	}
	uuid, err := r.dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	identity := map[string]any{"kind": "device", "uuid": uuid, "driver_version": version, "driver_path": driver.Path(), "capability": []int{major, minor}, "elements": 8, "rounds": 5, "declared_launches": 74, "guard_scope": "caller allocations only; engine scratch exact extent"}
	encoded, _ := json.Marshal(identity)
	t.Logf("REPLAY_NATIVE_JSON %s", encoded)
	imageMetadata := make(map[string]replayPacketImage)
	for _, image := range manifest.Images {
		imageMetadata[image.Name] = image
	}
	launches := 0
	for _, tc := range manifest.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			stream, err := r.dev.NewStream()
			if stream != nil {
				r.streams = append(r.streams, stream)
			}
			if err != nil {
				t.Fatal(err)
			}
			guarded := make(map[int]cudamem.Slice[byte])
			bindings := make([]replayDeviceBinding, 0, 3)
			initial := make(map[int][]uint32)
			for arg := 0; arg < 3; arg++ {
				whole, err := cudamem.Alloc[byte](r.dev, 40)
				r.buffers = append(r.buffers, whole)
				if err != nil {
					t.Fatal(err)
				}
				data := make([]byte, 40)
				binary.LittleEndian.PutUint32(data, replayPacketGuard)
				binary.LittleEndian.PutUint32(data[36:], replayPacketGuard)
				for i := 0; i < 8; i++ {
					v := float32(i + 2)
					if arg == 1 {
						v = float32(-8 - i)
					}
					if arg == 2 {
						v = float32(5 + 2*i)
					}
					binary.LittleEndian.PutUint32(data[4+4*i:], math.Float32bits(v))
				}
				if err := whole.CopyFromHost(data); err != nil {
					t.Fatal(err)
				}
				initial[arg] = replayNativeBits(data[4:36])
				guarded[arg] = whole
				bindings = append(bindings, replayDeviceBinding{Arg: arg, Slice: whole.Slice(4, 36)})
			}
			load := func(name string) *replayPacketOwner {
				owner, err := replayLoadPacketOwner(r.dev, images[name], imageMetadata[name].Signature)
				r.modules = append(r.modules, owner)
				if err != nil {
					t.Fatal(err)
				}
				attrs := make(map[string]int)
				for name, attr := range map[string]driver.FunctionAttribute{"registers": driver.FuncAttributeNumRegs, "shared_bytes": driver.FuncAttributeSharedSizeBytes, "local_bytes": driver.FuncAttributeLocalSizeBytes, "max_threads": driver.FuncAttributeMaxThreadsPerBlock} {
					value, err := owner.kernel.function.Attribute(attr)
					if err != nil {
						t.Fatal(err)
					}
					attrs[name] = value
				}
				meta := map[string]any{"kind": "loaded", "case": tc.Name, "image": name, "image_sha256": imageMetadata[name].ImageHash, "source_sha256": imageMetadata[name].SourceHash, "function": uint64(owner.kernel.Raw()), "resources": attrs}
				encoded, _ := json.Marshal(meta)
				t.Logf("REPLAY_NATIVE_JSON %s", encoded)
				return owner
			}
			base := load(tc.Baseline)
			spec := replayNativeSpec(base.kernel.signature, stream)
			policy := replayPolicy{Rounds: 5, Exact: true}
			backend, err := newCUDAReplayBackend(r.dev, spec, policy, bindings)
			if err != nil {
				t.Fatal(err)
			}
			observer := &replayNativeObserver{cudaReplayBackend: backend, guards: guarded, names: map[Kernel]string{base.kernel: tc.Baseline}, emit: func(value any) error {
				row := map[string]any{"case": tc.Name, "receipt": value}
				b, err := json.Marshal(row)
				if err == nil {
					t.Logf("REPLAY_NATIVE_JSON %s", b)
				}
				return err
			}}
			// Count retained arm observations even if later admission or final
			// ownership assertions fail; case summaries are a separate gate.
			defer func() {
				launches += len(observer.runs)
				row := map[string]any{"kind": "case-arm-count", "case": tc.Name, "completed_arms": len(observer.runs), "declared_arms": tc.Launches, "count_semantics": "retained completed arm observations; not CUPTI counter"}
				encoded, _ := json.Marshal(row)
				t.Logf("REPLAY_NATIVE_JSON %s", encoded)
			}()
			fixture, recordErr := recordReplay(ctx, observer, replayArm{Kernel: base.kernel, Release: base.Close}, spec, policy)
			if fixture != nil {
				r.fixtures = append(r.fixtures, fixture)
			}
			var result replayResult
			var checkErr error
			retryAttempts := 0
			if tc.Candidate == "" {
				if recordErr == nil {
					t.Fatal("negative baseline was accepted")
				}
				needle := "unchanged"
				if tc.Outcome == "baseline_noise" {
					needle = "noise tolerance"
				}
				if !strings.Contains(recordErr.Error(), needle) {
					t.Fatalf("wrong refusal: %v", recordErr)
				}
			} else {
				if recordErr != nil {
					t.Fatal(recordErr)
				}
				// Real borrowed allocation refuses Free while the fixture holds it.
				for _, binding := range bindings {
					if err := binding.Slice.Free(); !errors.Is(err, cudamem.ErrInUse) {
						t.Fatalf("borrowed Free not refused: %v", err)
					}
				}
				candidate := load(tc.Candidate)
				observer.names[candidate.kernel] = tc.Candidate
				release := candidate.Close
				if tc.Outcome == "injected_release_retry" {
					release = func() error {
						retryAttempts++
						if retryAttempts == 1 {
							return fmt.Errorf("injected retained pin release")
						}
						return candidate.Close()
					}
				}
				result, checkErr = fixture.Check(ctx, replayArm{Kernel: candidate.kernel, Release: release})
				switch tc.Outcome {
				case "pass":
					if checkErr != nil || !result.Pass {
						t.Fatalf("positive failed: %+v %v", result, checkErr)
					}
				case "wrong_answer":
					if checkErr != nil || result.Pass || len(result.Outputs) != 1 || result.Outputs[0].Pass {
						t.Fatalf("wrong candidate admitted: %+v %v", result, checkErr)
					}
				case "stable_write_refusal":
					if checkErr == nil || !strings.Contains(checkErr.Error(), "stable argument") {
						t.Fatalf("stable write not refused: %v", checkErr)
					}
				case "injected_release_retry":
					if checkErr == nil || !strings.Contains(checkErr.Error(), "injected retained") {
						t.Fatalf("cleanup fault not retained: %v", checkErr)
					}
					if backend.lease == nil || retryAttempts != 1 {
						t.Fatal("lease/pin lost before retry")
					}
					if _, err := candidate.kernel.function.Attribute(driver.FuncAttributeNumRegs); err != nil {
						t.Fatalf("actual retained module no longer live: %v", err)
					}
				}
				if err := replayAdmitNativeTrials(result, tc.Outcome, checkErr); err != nil {
					t.Fatal(err)
				}
				if tc.Outcome == "pass" && (!result.Native || result.Timing != "cuda_event_interval" || len(result.Trials) != 10 || result.Baseline <= 0 || result.Candidate <= 0) {
					t.Fatal("positive missing actual event measurements")
				}
				if tc.Outcome == "pass" && fixture.writeOnly[1] != (tc.Name != "inplace") {
					t.Fatal("actual poison write-only observation disagrees")
				}
			}
			// Retain full arrays after each real completed arm, including stable
			// mutation before recovery. Pointer identity proves weights stayed in place.
			for _, run := range observer.runs {
				if run.Args[2] != uint64(bindings[2].Slice.Ptr()) {
					t.Fatal("stable weights were retargeted")
				}
				if run.Args[0] == uint64(bindings[0].Slice.Ptr()) || run.Args[1] == uint64(bindings[1].Slice.Ptr()) {
					t.Fatal("ordinary input/output did not use scratch")
				}
			}
			stable := make([]byte, 32)
			if err := bindings[2].Slice.CopyToHost(stable); err != nil {
				t.Fatal(err)
			}
			if err := replayAdmitStableRecovery(replayNativeBits(stable), initial[2]); err != nil {
				t.Fatal(err)
			}
			if fixture != nil {
				if err := fixture.Close(ctx); err != nil {
					t.Fatal(err)
				}
			}
			if tc.Outcome == "injected_release_retry" && retryAttempts != 2 {
				t.Fatal("retained pin was not retried once")
			}
			if len(observer.runs) != tc.Launches {
				t.Fatalf("actual completed invocation count=%d declared=%d", len(observer.runs), tc.Launches)
			}
			for _, binding := range bindings {
				original := make([]byte, 32)
				if err := binding.Slice.CopyToHost(original); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(replayNativeBits(original), initial[binding.Arg]) {
					t.Fatal("caller allocation changed")
				}
				if err := binding.Slice.Free(); err != nil {
					t.Fatalf("borrow retained after fixture Close: %v", err)
				}
			}
			trialErrors := make([]string, len(result.Trials))
			for i, trial := range result.Trials {
				if trial.Err != nil {
					trialErrors[i] = trial.Err.Error()
				}
			}
			row := map[string]any{"kind": "case", "case": tc.Name, "outcome": tc.Outcome, "record_error": fmt.Sprint(recordErr), "check_error": fmt.Sprint(checkErr), "trial_errors": trialErrors, "initial_bits": initial, "stable_after_bits": replayNativeBits(stable), "runs": observer.runs, "result": result, "retry_attempts": retryAttempts, "count_semantics": "successful checked native launch/completion calls; not CUPTI counter"}
			encoded, err := json.Marshal(row)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("REPLAY_NATIVE_JSON %s", encoded)
		})
	}
	encoded, _ = json.Marshal(map[string]any{"kind": "total-arm-count", "completed_arms": launches, "declared_arms": manifest.Launches, "count_semantics": "retained completed arm observations independent of case summaries; not CUPTI counter"})
	t.Logf("REPLAY_NATIVE_JSON %s", encoded)
	if launches != manifest.Launches {
		t.Fatalf("total successful completed invocations=%d want=%d", launches, manifest.Launches)
	}
}

// A completed device arm can still be an incomplete comparison trial. The
// stable-write control fails before success-only event receipt stamping; its
// raw observed device interval remains in the arm record, not this trial.
func replayAdmitNativeTrials(result replayResult, outcome string, checkErr error) error {
	if !result.Native || result.Timing != "cuda_event_interval" {
		return fmt.Errorf("native event provenance missing")
	}
	complete := result.Trials
	if outcome == "stable_write_refusal" {
		const reason = "replay stable argument 2 was modified"
		if checkErr == nil || checkErr.Error() != `cudatrial: arm "candidate" round 1: `+reason || len(result.Trials) != 2 {
			return fmt.Errorf("stable refusal missing exact error or two-trial prefix")
		}
		failed := result.Trials[1]
		if failed.Name != "candidate" || failed.Round != 1 || failed.Err == nil || failed.Err.Error() != reason || failed.Complete || failed.Digest != "" || failed.TimingKind != cudatrial.TimingWallElapsed || len(failed.Timings) != 0 || failed.HasTiming("cuda_event_interval") || failed.DeviceNs != 0 || failed.QueueNs != 0 || failed.GpuIntervalUnionNs != 0 || failed.DeviceDuration != 0 || failed.SummedKernel != 0 || failed.GPUBusy != 0 || failed.APICall != 0 {
			return fmt.Errorf("stable refusal trial fabricates completion or timing")
		}
		if result.Baseline != 0 || result.Candidate != 0 || len(result.Outputs) != 0 {
			return fmt.Errorf("stable refusal fabricates a complete comparison")
		}
		complete = complete[:1]
		if complete[0].Name != "baseline" || complete[0].Round != 1 {
			return fmt.Errorf("stable refusal baseline prefix missing")
		}
	} else if len(complete) != 10 {
		return fmt.Errorf("native comparison trial count differs")
	}
	for _, trial := range complete {
		if trial.Err != nil || !trial.Complete || !trial.HasTiming("cuda_event_interval") || trial.DeviceNs != cudatrial.Unmeasured {
			return fmt.Errorf("native event receipt mislabelled")
		}
		duration, err := trial.DurationFor("cuda_event_interval")
		if err != nil || duration <= 0 {
			return fmt.Errorf("native event receipt interval unresolved")
		}
	}
	return nil
}

func replayAdmitStableRecovery(got, want []uint32) error {
	if !reflect.DeepEqual(got, want) {
		return fmt.Errorf("stable bytes not restored after check")
	}
	return nil
}

// This host-only backend uses fake durations to reproduce the actual engine's
// native receipt bookkeeping. It does not supply physical timing evidence.
type replayReceiptTestBackend struct{ *fixtureTestBackend }

func (*replayReceiptTestBackend) Timing() (cudatrial.TimingKind, bool) {
	return "cuda_event_interval", true
}

func TestHotSwapReplayNativeStableRefusalAdmission(t *testing.T) {
	b := &replayReceiptTestBackend{fixtureBackend()}
	initial := replayNativeBits(b.data[2])
	base := fixtureKernel("baseline", false)
	f, err := recordReplay(context.Background(), b, replayArm{Kernel: base, Release: func() error { return nil }}, fixtureSpec(), fixturePolicy())
	if err != nil {
		t.Fatal(err)
	}
	candidate := fixtureKernel("candidate", false)
	ordinary := candidate.run
	candidate.run = func(config cudalaunch.Config, args []any) error {
		if err := ordinary(config, args); err != nil {
			return err
		}
		copy(args[2].([]byte), fixtureBytes(12, 14))
		return nil
	}
	result, checkErr := f.Check(context.Background(), replayArm{Kernel: candidate, Release: func() error { return nil }})
	if err := replayAdmitNativeTrials(result, "stable_write_refusal", checkErr); err != nil {
		t.Fatalf("actual expected incomplete trial refused: %v; trial=%+v", err, result.Trials)
	}
	if err := replayAdmitStableRecovery(replayNativeBits(b.data[2]), initial); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"missing-error", "wrong-error", "false-complete", "invented-timing", "summed-kernel", "gpu-busy", "api-call", "missing-baseline-timing", "missing-trial", "invented-comparison"} {
		t.Run(name, func(t *testing.T) {
			changed := result
			changed.Trials = append([]cudatrial.Receipt(nil), result.Trials...)
			failure := checkErr
			switch name {
			case "missing-error":
				changed.Trials[1].Err = nil
			case "wrong-error":
				changed.Trials[1].Err = errors.New("unrelated failure")
				failure = changed.Trials[1].Err
			case "false-complete":
				changed.Trials[1].Complete = true
			case "invented-timing":
				if err := changed.Trials[1].RecordTiming("cuda_event_interval", time.Nanosecond); err != nil {
					t.Fatal(err)
				}
			case "summed-kernel":
				changed.Trials[1].SummedKernel = time.Nanosecond
			case "gpu-busy":
				changed.Trials[1].GPUBusy = time.Nanosecond
			case "api-call":
				changed.Trials[1].APICall = time.Nanosecond
			case "missing-baseline-timing":
				changed.Trials[0].Timings = nil
			case "missing-trial":
				changed.Trials = changed.Trials[:1]
			case "invented-comparison":
				changed.Candidate = time.Nanosecond
			}
			if err := replayAdmitNativeTrials(changed, "stable_write_refusal", failure); err == nil {
				t.Fatal("invalid incomplete native trial admitted")
			}
		})
	}
	changed := append([]uint32(nil), initial...)
	changed[1] ^= 1
	if err := replayAdmitStableRecovery(changed, initial); err == nil {
		t.Fatal("unrestored stable bytes admitted")
	}
	if err := f.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestHotSwapReplayNativeCompleteAdmission(t *testing.T) {
	b := &replayReceiptTestBackend{fixtureBackend()}
	policy := fixturePolicy()
	policy.Rounds = 5
	f, err := recordReplay(context.Background(), b, replayArm{Kernel: fixtureKernel("baseline", false), Release: func() error { return nil }}, fixtureSpec(), policy)
	if err != nil {
		t.Fatal(err)
	}
	result, err := f.Check(context.Background(), replayArm{Kernel: fixtureKernel("candidate", false), Release: func() error { return nil }})
	if err != nil || !result.Pass {
		t.Fatalf("complete host comparison failed: %v", err)
	}
	if err := replayAdmitNativeTrials(result, "pass", nil); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"incomplete", "missing-event", "unmeasured-event"} {
		t.Run(name, func(t *testing.T) {
			changed := result
			changed.Trials = append([]cudatrial.Receipt(nil), result.Trials...)
			switch name {
			case "incomplete":
				changed.Trials[4].Complete = false
			case "missing-event":
				changed.Trials[4].Timings = nil
			case "unmeasured-event":
				changed.Trials[4].Timings = map[cudatrial.TimingKind]time.Duration{"cuda_event_interval": 0}
			}
			if err := replayAdmitNativeTrials(changed, "pass", nil); err == nil {
				t.Fatal("invalid complete native trial admitted")
			}
		})
	}
	if err := f.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestHotSwapReplayPreparedPacket(t *testing.T) {
	dir := os.Getenv("HOTSWAP_REPLAY_HOST_PACKET_DIR")
	if dir == "" {
		t.Skip("host prepared-packet audit not requested")
	}
	if _, _, err := replayReadPacket(dir, os.Getenv("HOTSWAP_REPLAY_SOURCE_ROOT"), false); err != nil {
		t.Fatal(err)
	}
}

func TestHotSwapReplayNativeEquationControls(t *testing.T) {
	for _, name := range []string{"baseline", "correct", "wrong", "nonfinite", "noop", "inplace_baseline", "inplace_correct", "noisy", "stable_write"} {
		run := replayNativeRun{Kernel: name, Buffers: make(map[int][]uint32), CallerGuarded: make(map[int][]uint32)}
		for arg := 0; arg < 3; arg++ {
			run.Buffers[arg] = make([]uint32, 8)
			run.CallerGuarded[arg] = make([]uint32, 10)
			run.CallerGuarded[arg][0], run.CallerGuarded[arg][9] = replayPacketGuard, replayPacketGuard
		}
		for i := 0; i < 8; i++ {
			x, w, y := float32(i+2), float32(5+2*i), float32(-8-i)
			out, changedX, changedW := x+w, x+100, w
			switch name {
			case "noop":
				out, changedX = y, x
			case "wrong":
				out++
			case "nonfinite":
				if i == 3 {
					out = float32(math.Inf(1))
				}
			case "inplace_baseline", "inplace_correct":
				out = y + w
			case "stable_write":
				changedW += 7
			}
			run.Buffers[0][i], run.Buffers[1][i], run.Buffers[2][i] = math.Float32bits(changedX), math.Float32bits(out), math.Float32bits(changedW)
			run.CallerGuarded[0][i+1], run.CallerGuarded[1][i+1], run.CallerGuarded[2][i+1] = math.Float32bits(x), math.Float32bits(y), math.Float32bits(changedW)
		}
		if err := replayVerifyNativeRun(run, 0); err != nil {
			t.Fatalf("host oracle positive %s: %v", name, err)
		}
		for arg := 0; arg < 3; arg++ {
			run.Buffers[arg][7] ^= 1
			if err := replayVerifyNativeRun(run, 0); err == nil {
				t.Fatalf("host oracle accepted changed argument %d element %s", arg, name)
			}
			run.Buffers[arg][7] ^= 1
			run.CallerGuarded[arg][8] ^= 1
			if err := replayVerifyNativeRun(run, 0); err == nil {
				t.Fatalf("host oracle accepted changed original caller argument %d %s", arg, name)
			}
			run.CallerGuarded[arg][8] ^= 1
		}
	}
}

func TestHotSwapReplayPreparedPacketControls(t *testing.T) {
	dir := os.Getenv("HOTSWAP_REPLAY_HOST_PACKET_DIR")
	if dir == "" {
		t.Skip("retained prepared packet controls not requested")
	}
	root := os.Getenv("HOTSWAP_REPLAY_SOURCE_ROOT")
	m, _, err := replayReadPacket(dir, root, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"wrong-source", "wrong-image", "duplicate-image", "changed-policy", "extra-file", "trailing-json", "symlink", "mismatched-schema", "wrong-tool", "missing-image"} {
		t.Run(name, func(t *testing.T) {
			copyDir, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			files, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range files {
				b, err := replayPacketFile(filepath.Join(dir, f.Name()), 4<<20)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(copyDir, f.Name()), b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, _, err := replayReadPacket(copyDir, root, false); err != nil {
				t.Fatalf("copied positive control failed before mutation: %v", err)
			}
			encoded, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			var modified replayPacketManifest
			if err := json.Unmarshal(encoded, &modified); err != nil {
				t.Fatal(err)
			}
			switch name {
			case "wrong-source":
				if err := os.WriteFile(filepath.Join(copyDir, "baseline.cu"), []byte("wrong"), 0600); err != nil {
					t.Fatal(err)
				}
			case "wrong-image":
				if err := os.WriteFile(filepath.Join(copyDir, "baseline.cubin"), []byte("wrong"), 0600); err != nil {
					t.Fatal(err)
				}
			case "duplicate-image":
				modified.Images[1] = modified.Images[0]
			case "changed-policy":
				modified.Launches++
			case "extra-file":
				if err := os.WriteFile(filepath.Join(copyDir, "extra"), []byte("extra"), 0600); err != nil {
					t.Fatal(err)
				}
			case "trailing-json":
			case "symlink":
				if err := os.Remove(filepath.Join(copyDir, "baseline.cubin")); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(filepath.Join(dir, "baseline.cubin"), filepath.Join(copyDir, "baseline.cubin")); err != nil {
					t.Fatal(err)
				}
			case "mismatched-schema":
				modified.Images[0].Signature.Params[0].Size = 4
			case "wrong-tool":
				modified.CompilerHash = "wrong"
			case "missing-image":
				if err := os.Remove(filepath.Join(copyDir, "baseline.cubin")); err != nil {
					t.Fatal(err)
				}
			}
			encoded, err = json.Marshal(modified)
			if err != nil {
				t.Fatal(err)
			}
			if name == "trailing-json" {
				encoded = append(encoded, []byte(" {}")...)
			}
			if err := os.WriteFile(filepath.Join(copyDir, "manifest.json"), encoded, 0600); err != nil {
				t.Fatal(err)
			}
			_, _, err = replayReadPacket(copyDir, root, false)
			if err == nil {
				t.Fatal("mutated prepared packet accepted")
			}
			needle := map[string]string{"wrong-source": "source bytes", "wrong-image": "CUBIN", "duplicate-image": "image source/identity", "changed-policy": "policy/compiler", "extra-file": "inventory", "trailing-json": "trailing", "symlink": "CUBIN", "mismatched-schema": "PTX schema", "wrong-tool": "policy/compiler", "missing-image": "CUBIN"}[name]
			if !strings.Contains(err.Error(), needle) {
				t.Fatalf("wrong refusal branch: %v want %s", err, needle)
			}
		})
	}
}
