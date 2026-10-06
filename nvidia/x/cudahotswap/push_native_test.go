//go:build linux && native

package cudahotswap

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// This producer must be source/packet/operation reviewed before a coordinator
// dispatches it under gpu-lock. Its output alone is not a qualification seal.
func publicNativeRequire(t *testing.T) string {
	t.Helper()
	if os.Getenv("HOTSWAP_PUBLIC_NATIVE") != "1" {
		t.Skip("HOTSWAP_PUBLIC_NATIVE=1 requires reviewed native dispatch")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err) // Before the first CUDA call.
	}
	dir := os.Getenv("HOTSWAP_PUBLIC_ARTIFACTS")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute HOTSWAP_PUBLIC_ARTIFACTS is required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	return dir
}

type publicRecordingLoader struct {
	base     *NVRTCLoader
	dir      string
	next     int
	receipt  *nativereceipt.Session
	compiled map[*Compilation]string
}

func (l *publicRecordingLoader) Compile(ctx context.Context, source Source) (*Compilation, error) {
	sum := fmt.Sprintf("%x", sha256.Sum256([]byte(source.Code)))
	if err := l.receipt.Capture("public-before-compile", sum, ""); err != nil {
		return nil, err
	}
	c, err := l.base.Compile(ctx, source)
	if err != nil {
		return c, errors.Join(err, l.receipt.Capture("public-compile-error", sum, ""))
	}
	if err := l.receipt.Capture("public-after-compile", sum, fmt.Sprintf("%x", sha256.Sum256([]byte(c.PTX)))); err != nil {
		return c, err
	}
	l.next++
	dir := filepath.Join(l.dir, fmt.Sprintf("compile-%03d", l.next))
	if err := os.Mkdir(dir, 0700); err != nil {
		return nil, err
	}
	for name, data := range map[string][]byte{"source.cu": []byte(source.Code), "module.ptx": []byte(c.PTX), "module.cubin": c.native.CUBIN(), "compile.log": []byte(c.native.Log())} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			return nil, err
		}
	}
	if err := publicWriteJSON(filepath.Join(dir, "identity.json"), struct {
		Source       Source
		Key          string
		Dependencies any
	}{source, c.Key, c.Dependencies}); err != nil {
		return nil, err
	}
	if l.compiled == nil {
		l.compiled = make(map[*Compilation]string)
	}
	l.compiled[c] = dir
	return c, nil
}

func (l *publicRecordingLoader) Load(ctx context.Context, c *Compilation) (Kernel, error) {
	kernel, err := l.base.Load(ctx, c)
	if err != nil {
		return kernel, err
	}
	dir := l.compiled[c]
	if dir == "" || kernel == nil || kernel.Raw() == 0 {
		return kernel, errors.New("public load receipt: actual compilation or function absent")
	}
	image, err := c.CUBIN()
	if err != nil {
		return kernel, err
	}
	var nativeContext driver.Context
	if err := l.base.dev.WithContext(func() error { var err error; nativeContext, err = driver.CtxGetCurrent(); return err }); err != nil {
		return kernel, err
	}
	if err := l.receipt.Capture("public-after-load", fmt.Sprintf("%x", sha256.Sum256([]byte(c.PTX))), fmt.Sprintf("%x", sha256.Sum256(image))); err != nil {
		return kernel, err
	}
	if err := publicWriteJSON(filepath.Join(dir, "loaded.json"), struct {
		Key, Entry, PTXSHA256, CUBINSHA256 string
		Function, Context                  uint64
	}{c.Key, c.Entry, fmt.Sprintf("%x", sha256.Sum256([]byte(c.PTX))), fmt.Sprintf("%x", sha256.Sum256(image)), uint64(kernel.Raw()), uint64(nativeContext)}); err != nil {
		return kernel, err
	}
	return kernel, nil
}

func publicWriteJSON(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0600)
}

type publicProcessCapture struct{ bytes.Buffer }

func (b *publicProcessCapture) Write(data []byte) (int, error) {
	if len(data) > (1<<20)-b.Len() {
		return 0, errors.New("cold process output exceeds 1 MiB budget")
	}
	return b.Buffer.Write(data)
}

type publicNativeFrame struct {
	dev       *cuda.Device
	service   *Service
	stream    *cuda.Stream
	buffers   []cudamem.Slice[byte]
	pins      []*Pin
	raw       []*cudagraph.Exec
	rawGraphs []*cudagraph.Graph
	execs     []*cudagraph.Exec
	graphs    []*cudagraph.Graph
	n         int
	process   nativereceipt.Process
}

func (f *publicNativeFrame) close() error {
	// Raw control graphs are caller-owned, so establish completion before
	// releasing their original-generation pins. Stop on any unresolved owner.
	if f.stream != nil {
		if err := f.stream.Synchronize(context.Background()); err != nil {
			return err
		}
	}
	for len(f.raw) > 0 {
		last := len(f.raw) - 1
		if err := f.raw[last].Destroy(); err != nil {
			return err
		}
		f.raw = f.raw[:last]
	}
	for len(f.rawGraphs) > 0 {
		last := len(f.rawGraphs) - 1
		if err := f.rawGraphs[last].Destroy(); err != nil {
			return err
		}
		f.rawGraphs = f.rawGraphs[:last]
	}
	for len(f.pins) > 0 {
		last := len(f.pins) - 1
		if err := f.pins[last].Release(context.Background()); err != nil {
			return err
		}
		f.pins = f.pins[:last]
	}
	if f.service != nil {
		if err := f.service.Close(context.Background()); err != nil {
			return err
		}
		f.service = nil
	}
	for len(f.execs) > 0 {
		last := len(f.execs) - 1
		if err := f.execs[last].Destroy(); err != nil {
			return err
		}
		f.execs = f.execs[:last]
	}
	for len(f.graphs) > 0 {
		last := len(f.graphs) - 1
		if err := f.graphs[last].Destroy(); err != nil {
			return err
		}
		f.graphs = f.graphs[:last]
	}
	if f.stream != nil {
		if err := f.stream.Close(); err != nil {
			return err
		}
		f.stream = nil
	}
	for len(f.buffers) > 0 {
		last := len(f.buffers) - 1
		if err := f.buffers[last].Free(); err != nil {
			return err
		}
		f.buffers = f.buffers[:last]
	}
	if f.dev != nil {
		if err := f.dev.Close(); err != nil {
			return err
		}
		f.dev = nil
	}
	return nil
}

func publicNativeFrameNew(t *testing.T, dir string, n int) *publicNativeFrame {
	t.Helper()
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	f := &publicNativeFrame{n: n}
	t.Cleanup(func() {
		if err := f.close(); err != nil {
			t.Errorf("public native cleanup: %v", err)
			if err := f.close(); err != nil {
				t.Errorf("public native cleanup retry: %v", err)
			}
		}
	})
	var err error
	f.dev, err = cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	f.stream, err = f.dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		buffer, err := cudamem.Alloc[byte](f.dev, 4*n)
		if buffer.Ptr() != 0 {
			f.buffers = append(f.buffers, buffer)
		}
		if err != nil {
			t.Fatal(err)
		}
		view, err := cudamem.Reinterpret[float32](buffer)
		if err != nil {
			t.Fatal(err)
		}
		data := make([]float32, n)
		for j := range data {
			data[j] = float32((i + 1) * (j%31 + 1))
		}
		if err := view.CopyFromHost(data); err != nil {
			t.Fatal(err)
		}
	}
	loader, err := NewNVRTCLoader(f.dev)
	if err != nil {
		t.Fatal(err)
	}
	f.service, err = NewService(loader)
	if err != nil {
		t.Fatal(err)
	}
	// Only a transparent recorder replaces the Loader after NewService wired
	// its actual native replay backend to this exact native loader/device.
	receipt, err := nativereceipt.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := errors.Join(receipt.Write(filepath.Join(dir, "native-libraries.json")), receipt.Close()); err != nil {
			t.Errorf("native library receipt: %v", err)
		}
	})
	f.service.loader = &publicRecordingLoader{base: loader, dir: dir, receipt: receipt}
	major, minor, err := f.dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	nvMajor, nvMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	data, err := replayPacketFile(binary, 128<<20)
	if err != nil {
		t.Fatal(err)
	}
	f.process, err = nativereceipt.Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	f.process.ExecutableSHA256 = fmt.Sprintf("%x", sha256.Sum256(data))
	if f.process.Executable != binary {
		t.Fatal("public receipt: actual executable path differs")
	}
	build, _ := debug.ReadBuildInfo()
	if err := publicWriteJSON(filepath.Join(dir, "runtime.json"), struct {
		Target                [2]int
		NVRTC                 [2]int32
		Binary, SHA256, Clock string
		Build                 *debug.BuildInfo
	}{[2]int{major, minor}, [2]int32{nvMajor, nvMinor}, binary, fmt.Sprintf("%x", sha256.Sum256(data)), "Push wall latency; replay native CUDA events", build}); err != nil {
		t.Fatal(err)
	}
	return f
}

func publicAXPBY(wrong bool, index int) string {
	body := "out[i] = alpha*a[i] + b[i];"
	if index >= 0 {
		body = "out[i] = b[i] + alpha*a[i];"
	}
	if wrong {
		body = "out[i] = alpha*a[i] + b[i] + 1.0f;"
	}
	return fmt.Sprintf("// candidate %d\nextern \"C\" __global__ void axpby(float *out, const float *a, const float *b, float alpha, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; int stride=blockDim.x*gridDim.x; for (; i<n; i+=stride) { %s } }", index, body)
}

func (f *publicNativeFrame) recordAXPBY(t *testing.T) {
	t.Helper()
	if _, err := f.service.Register(context.Background(), "kernel", Source{Code: publicAXPBY(false, -1), Filename: "baseline.cu", Entry: "axpby"}, "baseline"); err != nil {
		t.Fatal(err)
	}
	config := FixtureConfig{Args: []any{nil, nil, nil, float32(2.5), int32(f.n)}, Outputs: []FixtureOutput{{Arg: 0, Elem: "f32"}}, Policy: DefaultReplayPolicy(), Launch: cudalaunch.Config{Grid: cudashape.Grid3D{X: uint32((f.n + 31) / 32), Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: f.stream}}
	for i, buffer := range f.buffers {
		config.Buffers = append(config.Buffers, FixtureBuffer{Arg: i, Bytes: 4 * f.n, Buffer: buffer, Stable: i == 2})
	}
	fixture, err := f.service.RecordFixture(context.Background(), "kernel", config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.service.SetCheck("kernel", CheckConfig{Fixture: fixture}, "replay"); err != nil {
		t.Fatal(err)
	}
}

func TestNativeServicePushReplayAndGraphs(t *testing.T) {
	dir := publicNativeRequire(t)
	f := publicNativeFrameNew(t, filepath.Join(dir, "public"), 32)
	f.recordAXPBY(t)
	for _, wrong := range []bool{false, true} {
		result, err := f.service.Push(context.Background(), "kernel", Source{Code: publicAXPBY(wrong, 1), Filename: "candidate.cu", Entry: "axpby"}, "public replay")
		if err := publicWriteJSON(filepath.Join(dir, fmt.Sprintf("replay-wrong-%t.json", wrong)), struct {
			Result *PushResult
			Error  string
		}{result, fmt.Sprint(err)}); err != nil {
			t.Fatal(err)
		}
		if result == nil || result.Published == wrong || (err != nil) != wrong || result.Report.Replay == nil || !result.Report.Replay.Native || (result.Report.Outcome == OutcomeOK) == wrong {
			t.Fatalf("actual public replay: %+v, %v", result, err)
		}
	}
	abiResult, abiErr := f.service.Push(context.Background(), "kernel", Source{Code: "extern \"C\" __global__ void axpby(int n) {}", Filename: "bad-abi.cu", Entry: "axpby"}, "bad ABI")
	if abiErr == nil {
		t.Fatal("actual public ABI mismatch was admitted")
	}
	if err := publicWriteJSON(filepath.Join(dir, "bad-abi.json"), struct {
		Result *PushResult
		Error  string
	}{abiResult, abiErr.Error()}); err != nil {
		t.Fatal(err)
	}
	reverted, err := f.service.Revert(context.Background(), "kernel", "baseline")
	if err != nil {
		t.Fatal(err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "kernel-revert.json"), reverted); err != nil {
		t.Fatal(err)
	}
	if err := publicKernelRevertReadback(f, dir, reverted); err != nil {
		t.Fatal("kernel revert readback: ", err)
	}
	if err := publicConcurrentPinReadback(f, dir); err != nil {
		t.Fatal("concurrent pin readback: ", err)
	}
	inplace := func(rewrite bool) Source {
		body := "out[i] = out[i] + a[i];"
		if rewrite {
			body = "out[i] = a[i] + out[i];"
		}
		return Source{Code: "extern \"C\" __global__ void inplace(float *out, const float *a, unsigned n) { unsigned i=threadIdx.x; if(i<n) { " + body + " } }", Filename: "inplace.cu", Entry: "inplace"}
	}
	if _, err := f.service.Register(context.Background(), "inplace", inplace(false), "inplace baseline"); err != nil {
		t.Fatal(err)
	}
	inplaceConfig := FixtureConfig{Args: []any{nil, nil, uint32(32)}, Buffers: []FixtureBuffer{{Arg: 0, Bytes: 128, Buffer: f.buffers[0]}, {Arg: 1, Bytes: 128, Buffer: f.buffers[1], Stable: true}}, Outputs: []FixtureOutput{{Arg: 0, Elem: "f32"}}, Policy: DefaultReplayPolicy(), Launch: cudalaunch.Config{Grid: cudashape.Grid3D{X: 1, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: f.stream}}
	inplaceFixture, err := f.service.RecordFixture(context.Background(), "inplace", inplaceConfig)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.service.SetCheck("inplace", CheckConfig{Fixture: inplaceFixture}, "restored arms"); err != nil {
		t.Fatal(err)
	}
	inplaceResult, err := f.service.Push(context.Background(), "inplace", inplace(true), "inplace rewrite")
	if err != nil || inplaceResult == nil || !inplaceResult.Published || !inplaceResult.Report.Replay.Passed {
		t.Fatalf("public in-place restore: %+v %v", inplaceResult, err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "inplace.json"), inplaceResult); err != nil {
		t.Fatal(err)
	}
	if _, err := f.service.Register(context.Background(), "noop", Source{Code: "extern \"C\" __global__ void noop(float *out, const float *a, unsigned n) {}", Filename: "noop.cu", Entry: "noop"}, "no-op baseline"); err != nil {
		t.Fatal(err)
	}
	noopFixture, err := f.service.RecordFixture(context.Background(), "noop", inplaceConfig)
	if err == nil {
		t.Fatal("actual public RecordFixture accepted a no-op baseline")
	}
	if noopFixture != nil {
		if closeErr := noopFixture.Close(context.Background()); closeErr != nil {
			t.Fatal(closeErr)
		}
	}
	if err := publicWriteJSON(filepath.Join(dir, "noop.json"), struct {
		Refused bool
		Reason  string
	}{true, err.Error()}); err != nil {
		t.Fatal(err)
	}
	graphCode := func(value int) string {
		return fmt.Sprintf("extern \"C\" __global__ void graph_value(float *out) { out[threadIdx.x] = %d.0f; }", value)
	}
	if _, err := f.service.Register(context.Background(), "graph", Source{Code: graphCode(7), Filename: "graph-a.cu", Entry: "graph_value"}, "graph A"); err != nil {
		t.Fatal(err)
	}
	pin, err := f.service.Pin("graph")
	if pin != nil {
		f.pins = append(f.pins, pin)
	}
	if err != nil {
		t.Fatal(err)
	}
	var bound *cudagraph.Exec
	for _, register := range []bool{true, false} {
		g, err := cudagraph.New(f.dev)
		if g != nil {
			if register {
				f.graphs = append(f.graphs, g)
			} else {
				f.rawGraphs = append(f.rawGraphs, g)
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		desc, err := cudalaunch.Kernel(pin.Function()).Grid(1, 1, 1).Block(32, 1, 1).Describe()
		if err != nil {
			t.Fatal(err)
		}
		node, err := g.AddLaunch(desc, []any{f.buffers[0]})
		if err != nil {
			t.Fatal(err)
		}
		e, err := g.Instantiate()
		if e != nil {
			if register {
				f.execs = append(f.execs, e)
			} else {
				f.raw = append(f.raw, e)
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		if register {
			bound = e
			if _, err := f.service.RegisterGraph(context.Background(), "graph", e, node, f.stream, driver.Dims(1, 1, 1), f.buffers[0]); err != nil {
				t.Fatal(err)
			}
		}
	}
	readIndex := 0
	read := func(arm string, exec *cudagraph.Exec, retained *Pin, want float32) error {
		view, err := cudamem.Reinterpret[float32](f.buffers[0])
		if err != nil {
			return err
		}
		out := make([]float32, 32)
		if err := view.CopyToHost(out); err != nil {
			return err
		}
		graphExec, _, err := exec.KernelOrder()
		if err != nil {
			return err
		}
		var state State
		var function driver.Function
		if retained != nil {
			state, function = retained.state, retained.Function()
		} else {
			f.service.mu.Lock()
			state = f.service.states["graph"]
			f.service.mu.Unlock()
			f.service.graphs.mu.Lock()
			for _, binding := range f.service.graphs.bindings {
				if binding.exec == exec && binding.pin != nil && binding.pin.gen != nil {
					function = binding.pin.gen.function
					if state.Version != binding.pin.gen.version || state.SourceDigest != binding.pin.gen.digest {
						f.service.graphs.mu.Unlock()
						return errors.New("graph receipt: selected state differs from retained graph generation")
					}
				}
			}
			f.service.graphs.mu.Unlock()
		}
		var nativeContext driver.Context
		if err := f.dev.WithContext(func() error { var err error; nativeContext, err = driver.CtxGetCurrent(); return err }); err != nil {
			return err
		}
		var stream driver.Stream
		if err := f.stream.WithHandle(func(h driver.Stream) error { stream = h; return nil }); err != nil {
			return err
		}
		if function == 0 || graphExec == 0 || nativeContext == 0 || stream == 0 {
			return errors.New("graph receipt: missing actual native owner")
		}
		if err := publicWriteJSON(filepath.Join(dir, fmt.Sprintf("graph-readback-%03d.json", readIndex)), struct {
			Process                              nativereceipt.Process
			Arm                                  string
			State                                State
			Function, Context, Stream, GraphExec uint64
			Output                               []float32
			Expected                             float32
			Completed                            bool
		}{f.process, arm, state, uint64(function), uint64(nativeContext), uint64(stream), uint64(graphExec), out, want, true}); err != nil {
			return err
		}
		readIndex++
		for _, value := range out {
			if value != want {
				return fmt.Errorf("graph output %v, want all %g", out, want)
			}
		}
		return nil
	}
	if _, err := f.service.SetCheck("graph", CheckConfig{Checker: func(ctx context.Context, _ State) error {
		work, err := f.service.LaunchGraph(ctx, bound)
		if work != nil {
			err = errors.Join(err, work.Wait(ctx))
		}
		if err != nil {
			return err
		}
		return read("checker", bound, nil, 11)
	}}, "graph output checker"); err != nil {
		t.Fatal(err)
	}
	result, err := f.service.Push(context.Background(), "graph", Source{Code: graphCode(11), Filename: "graph-b.cu", Entry: "graph_value"}, "graph B")
	if err != nil || result == nil || !result.Published {
		t.Fatalf("actual public graph B: %+v %v", result, err)
	}
	rejected, err := f.service.Push(context.Background(), "graph", Source{Code: graphCode(13), Filename: "graph-wrong.cu", Entry: "graph_value"}, "graph callback refuses")
	if err == nil || rejected == nil || !rejected.Published || !rejected.RolledBack || rejected.Report.Outcome != OutcomeWrongAnswer {
		t.Fatalf("actual public graph callback rollback: %+v %v", rejected, err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "graph-rollback.json"), rejected); err != nil {
		t.Fatal(err)
	}
	checked, err := f.service.LaunchGraph(context.Background(), bound)
	if checked != nil {
		err = errors.Join(err, checked.Wait(context.Background()))
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := read("rollback", bound, nil, 11); err != nil {
		t.Fatal("graph callback rollback output: ", err)
	}
	if err := f.raw[0].Launch(f.stream); err != nil {
		t.Fatal(err)
	}
	if err := f.stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := read("raw-skipped-patch", f.raw[0], pin, 7); err != nil {
		t.Fatal("skipped-patch raw graph control: ", err)
	}
	if _, err := f.service.Revert(context.Background(), "graph", "graph A"); err != nil {
		t.Fatal(err)
	}
	work, err := f.service.LaunchGraph(context.Background(), bound)
	if work != nil {
		err = errors.Join(err, work.Wait(context.Background()))
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := read("reverted", bound, nil, 7); err != nil {
		t.Fatal("public Revert graph output: ", err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "graph.json"), struct {
		Patched, Skipped, Reverted int
		Push                       *PushResult
	}{11, 7, 7, result}); err != nil {
		t.Fatal(err)
	}
	if err := f.close(); err != nil {
		t.Fatal("public native owner completion: ", err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "public-cleanup.json"), struct {
		Process                                                     nativereceipt.Process
		DeviceClosed, ServiceClosed, StreamClosed                   bool
		Pins, RawExecs, BoundExecs, RawGraphs, BoundGraphs, Buffers int
	}{f.process, f.dev == nil, f.service == nil, f.stream == nil, len(f.pins), len(f.raw), len(f.execs), len(f.rawGraphs), len(f.graphs), len(f.buffers)}); err != nil {
		t.Fatal(err)
	}
}

func TestNativeServiceTwentyPushPrebuiltLatency(t *testing.T) {
	dir := publicNativeRequire(t)
	if os.Getenv("HOTSWAP_PUBLIC_COLD") != "1" {
		t.Skip("HOTSWAP_PUBLIC_COLD=1 requires twenty reviewed cold-process controls")
	}
	binary := os.Getenv("HOTSWAP_PUBLIC_COLD_BINARY")
	expected := os.Getenv("HOTSWAP_PUBLIC_COLD_SHA256")
	if !filepath.IsAbs(binary) || len(expected) != 64 {
		t.Fatal("absolute prepared cold harness and SHA256 required")
	}
	data, err := replayPacketFile(binary, 16<<20)
	if err != nil || fmt.Sprintf("%x", sha256.Sum256(data)) != expected {
		t.Fatalf("prepared cold harness identity: %v", err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "cold-harness.json"), struct {
		Binary, SHA256, Scope string
	}{binary, expected, "prepared executable startup; Go build excluded"}); err != nil {
		t.Fatal(err)
	}
	f := publicNativeFrameNew(t, filepath.Join(dir, "latency"), 1<<20)
	f.recordAXPBY(t)
	parentIdentity, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		wrong := i%2 != 0
		source := publicAXPBY(wrong, i)
		file := filepath.Join(dir, fmt.Sprintf("latency-%02d.cu", i))
		if err := os.WriteFile(file, []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		result, pushErr := f.service.Push(context.Background(), "kernel", Source{Code: source, Filename: file, Entry: "axpby"}, fmt.Sprintf("scripted %02d", i))
		pushWall := time.Since(start).Nanoseconds()
		if result == nil || result.Published == wrong || (pushErr != nil) != wrong || result.Report.Replay == nil || !result.Report.Replay.Native {
			t.Fatalf("scripted public Push %d: %+v %v", i, result, pushErr)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		command := publicColdCommand(ctx, binary, file)
		receiptPath := filepath.Join(dir, fmt.Sprintf("cold-%02d.native-libraries.json", i))
		command.Args = append(command.Args, "-native-receipt", receiptPath)
		start = time.Now()
		var out, diagnostics publicProcessCapture
		command.Stdout, command.Stderr = &out, &diagnostics
		if err := command.Start(); err != nil {
			cancel()
			t.Fatal(err)
		}
		identity, identityErr := nativereceipt.Identity(command.Process.Pid)
		if identityErr != nil {
			cancel()
			_ = command.Wait()
			t.Fatal(identityErr)
		}
		identity.ExecutableSHA256 = expected
		coldErr := command.Wait()
		stdout, stderr := out.Bytes(), diagnostics.Bytes()
		coldWall := time.Since(start).Nanoseconds()
		cancel()
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("cold-%02d.stdout", i)), stdout, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("cold-%02d.stderr", i)), stderr, 0600); err != nil {
			t.Fatal(err)
		}
		var cold Report
		if err := json.Unmarshal(stdout, &cold); err != nil {
			t.Fatalf("cold harness JSON %d: %v, process=%v", i, err, coldErr)
		}
		if (cold.Outcome == OutcomeOK) == wrong || (coldErr != nil) != wrong || wrong && cold.Outcome != OutcomeWrongAnswer {
			t.Fatalf("cold harness verdict %d: %+v, %v", i, cold, coldErr)
		}
		proof, err := nativereceipt.Read(receiptPath)
		if err != nil {
			t.Fatal(err)
		}
		if cold.Compile == nil || identity.Parent != os.Getpid() || identity.Group != parentIdentity.Group || identity.Executable != binary {
			t.Fatal("cold receipt: actual child identity or compilation absent")
		}
		if err := proof.Check(identity, fmt.Sprintf("%x", sha256.Sum256([]byte(source))), cold.Compile.PTXSHA256); err != nil {
			t.Fatal(err)
		}
		if err := publicWriteJSON(filepath.Join(dir, fmt.Sprintf("latency-%02d.json", i)), struct {
			Index                               int
			PushWallNanos, ColdProcessWallNanos int64
			Push                                *PushResult
			PushError                           string
			Cold                                Report
			Scope                               string
		}{i, pushWall, coldWall, result, fmt.Sprint(pushErr), cold, "request/process wall latency includes receipt capture and cold descriptor hashing; prepared harness startup excludes Go build; retained replay versus default harness; compiler cache reuse, no kernel throughput claim"}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestNativeServiceTwentyPushLatency(t *testing.T) {
	dir := publicNativeRequire(t)
	if os.Getenv("HOTSWAP_PUBLIC_COLD") != "1" {
		t.Skip("HOTSWAP_PUBLIC_COLD=1 requires twenty reviewed cold-process controls")
	}
	bindingPath := os.Getenv("HOTSWAP_PUBLIC_GO_RUN_BINDING")
	if !filepath.IsAbs(bindingPath) {
		t.Fatal("absolute reviewed Go run binding required")
	}
	data, err := replayPacketFile(bindingPath, 2<<20)
	if err != nil {
		t.Fatal(err)
	}
	binding, err := publicReadGoRunBinding(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if err := binding.check(); err != nil {
		t.Fatal(err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "cold-go-run-binding.json"), binding); err != nil {
		t.Fatal(err)
	}
	f := publicNativeFrameNew(t, filepath.Join(dir, "latency"), 1<<20)
	f.recordAXPBY(t)
	parentIdentity, err := nativereceipt.Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	publicColdEdits(t, func(t *testing.T, i int) {
		wrong := i%2 != 0
		source := publicAXPBY(wrong, i)
		file := filepath.Join(dir, fmt.Sprintf("latency-%02d.cu", i))
		if err := os.WriteFile(file, []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
		prior, err := f.service.Status("kernel")
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		result, pushErr := f.service.Push(context.Background(), "kernel", Source{Code: source, Filename: file, Entry: "axpby"}, fmt.Sprintf("scripted %02d", i))
		pushWall := time.Since(start).Nanoseconds()
		current, stateErr := f.service.Status("kernel")
		if stateErr != nil {
			t.Fatal(stateErr)
		}
		if err := publicColdPushCheck(prior, current, result, pushErr, source, wrong); err != nil {
			t.Fatalf("scripted public Push %d: %+v %v: %v", i, result, pushErr, err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		receiptPath := filepath.Join(dir, fmt.Sprintf("cold-%02d.native-libraries.json", i))
		command, err := publicGoRunCommand(ctx, binding, file, receiptPath)
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		start = time.Now()
		var out, diagnostics publicProcessCapture
		command.Stdout, command.Stderr = &out, &diagnostics
		if err := command.Start(); err != nil {
			cancel()
			t.Fatal(err)
		}
		identity, identityErr := nativereceipt.Identity(command.Process.Pid)
		if identityErr != nil {
			cancel()
			_ = command.Wait()
			t.Fatal(identityErr)
		}
		if identity.Executable != binding.GoBinary {
			cancel()
			_ = command.Wait()
			t.Fatal("Go launcher executable differs")
		}
		identity.ExecutableSHA256 = binding.GoSHA256
		if err := publicWriteJSON(filepath.Join(dir, fmt.Sprintf("cold-%02d.launcher.json", i)), identity); err != nil {
			cancel()
			_ = command.Wait()
			t.Fatal(err)
		}
		coldErr := command.Wait()
		stdout, stderr := out.Bytes(), diagnostics.Bytes()
		coldWall := time.Since(start).Nanoseconds()
		requestErr := ctx.Err()
		cancel()
		if requestErr != nil {
			_ = publicWriteJSON(filepath.Join(dir, fmt.Sprintf("cold-%02d.custody-refusal.json", i)), struct {
				Launcher                                       nativereceipt.Process
				Cause                                          string
				HarnessRetirementProven, OuterGuardianRequired bool
			}{identity, requestErr.Error(), false, true})
			t.Fatalf("Go run cancelled; descendant retirement remains owned by outer guardian: %v", requestErr)
		}
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("cold-%02d.stdout", i)), stdout, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("cold-%02d.stderr", i)), stderr, 0600); err != nil {
			t.Fatal(err)
		}
		var cold Report
		if err := json.Unmarshal(stdout, &cold); err != nil {
			t.Fatalf("cold harness JSON %d: %v, process=%v", i, err, coldErr)
		}
		if (cold.Outcome == OutcomeOK) == wrong || (coldErr != nil) != wrong || wrong && cold.Outcome != OutcomeWrongAnswer {
			t.Fatalf("cold harness verdict %d: %+v, %v", i, cold, coldErr)
		}
		proof, err := nativereceipt.Read(receiptPath)
		if err != nil {
			t.Fatal(err)
		}
		if cold.Compile == nil || publicGoRunProducer(identity, parentIdentity, proof.Process) != nil {
			t.Fatal("cold receipt: actual child identity or compilation absent")
		}
		if err := proof.Check(proof.Process, fmt.Sprintf("%x", sha256.Sum256([]byte(source))), cold.Compile.PTXSHA256); err != nil {
			t.Fatal(err)
		}
		if err := binding.check(); err != nil {
			t.Fatal(err)
		}
		if _, err := nativereceipt.Identity(proof.Process.PID); !os.IsNotExist(err) {
			t.Fatal("Go harness remains live; outer guardian retirement required")
		}
		if err := publicWriteJSON(filepath.Join(dir, fmt.Sprintf("cold-%02d.processes.json", i)), struct{ Launcher, Harness nativereceipt.Process }{identity, proof.Process}); err != nil {
			t.Fatal(err)
		}
		if err := publicWriteJSON(filepath.Join(dir, fmt.Sprintf("latency-%02d.json", i)), struct {
			Index                               int
			PushWallNanos, ColdProcessWallNanos int64
			Push                                *PushResult
			PushError                           string
			Cold                                Report
			Scope                               string
		}{i, pushWall, coldWall, result, fmt.Sprint(pushErr), cold, "request/process wall latency includes native mapped-receipt hashing; Go source/tool binding checks are outside cold wall interval; actual go run includes Go build and harness execution; no cache eviction or fresh-cache claim; retained replay versus default harness; compiler cache reuse, no kernel throughput claim"}); err != nil {
			t.Fatal(err)
		}
	})
}

// publicReadback32 refuses a state-only claim: every actual output is required.
func publicReadback32(got []float32, want func(int) float32) error {
	if len(got) != 32 || want == nil {
		return errors.New("public readback requires exactly 32 values and an oracle")
	}
	for i, value := range got {
		if value != want(i) {
			return fmt.Errorf("public readback element %d = %g, want %g", i, value, want(i))
		}
	}
	return nil
}

func publicKernelRevertReadback(f *publicNativeFrame, dir string, reverted State) error {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cfg := cudalaunch.Config{Grid: cudashape.Grid3D{X: 1, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: f.stream}
	work, err := f.service.Launch(ctx, "kernel", cfg, f.buffers[0], f.buffers[1], f.buffers[2], float32(2.5), int32(32))
	if work != nil {
		err = errors.Join(err, work.Wait(ctx))
	}
	if err != nil || work == nil {
		return errors.Join(err, errors.New("kernel revert missing completed submission"))
	}
	view, err := cudamem.Reinterpret[float32](f.buffers[0])
	if err != nil {
		return err
	}
	got := make([]float32, 32)
	if err := view.CopyToHost(got); err != nil {
		return err
	}
	state := work.State()
	if err := publicWriteJSON(filepath.Join(dir, "kernel-revert-readback.json"), struct {
		Process              nativereceipt.Process
		Reverted, Submission State
		Output               []float32
		Completed            bool
	}{f.process, reverted, state, got, true}); err != nil {
		return err
	}
	if state != reverted {
		return errors.New("kernel revert submission state differs")
	}
	return publicReadback32(got, func(i int) float32 { return float32(8 * (i%31 + 1)) })
}

func publicConcurrentPinReadback(f *publicNativeFrame, dir string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	code := func(value int) Source {
		return Source{Code: fmt.Sprintf("extern \"C\" __global__ void pin_value(float *out) { out[threadIdx.x] = %d.0f; }", value), Filename: "pin.cu", Entry: "pin_value"}
	}
	if _, err := f.service.Register(ctx, "concurrent", code(7), "retained original"); err != nil {
		return err
	}
	pin, err := f.service.Pin("concurrent")
	if pin != nil {
		f.pins = append(f.pins, pin)
	}
	if err != nil {
		return err
	}
	var buffers [2]cudamem.Slice[byte]
	for i := range buffers {
		buffer, err := cudamem.Alloc[byte](f.dev, 128)
		if buffer.Ptr() != 0 {
			f.buffers = append(f.buffers, buffer)
		}
		if err != nil {
			return err
		}
		buffers[i] = buffer
		view, err := cudamem.Reinterpret[float32](buffer)
		if err != nil {
			return err
		}
		poison := make([]float32, 32)
		for j := range poison {
			poison[j] = -123
		}
		if err := view.CopyFromHost(poison); err != nil {
			return err
		}
	}
	type readback struct {
		Output []float32
		Err    error
	}
	start, done := make(chan struct{}), make(chan readback, 1)
	finished := make(chan struct{})
	defer func() { cancel(); <-finished }()
	go func() {
		defer close(finished)
		select {
		case <-ctx.Done():
			done <- readback{Err: ctx.Err()}
			return
		case <-start:
		}
		err := f.dev.WithContext(func() error {
			return cudalaunch.Kernel(pin.Function()).Grid(1, 1, 1).Block(32, 1, 1).On(f.stream).Launch(buffers[0])
		})
		// Even a nonnil enqueue error can follow submission. Establish completion
		// before reporting or letting the caller release the original pin.
		err = errors.Join(err, f.stream.Synchronize(ctx))
		var got []float32
		if err == nil {
			view, viewErr := cudamem.Reinterpret[float32](buffers[0])
			err = viewErr
			got = make([]float32, 32)
			if err == nil {
				err = view.CopyToHost(got)
			}
		}
		done <- readback{got, err}
	}()
	var original readback
	var checkerState State
	if _, err := f.service.SetCheck("concurrent", CheckConfig{Checker: func(checkCtx context.Context, selected State) error {
		checkerState = selected
		close(start)
		select {
		case original = <-done:
		case <-checkCtx.Done():
			return checkCtx.Err()
		case <-ctx.Done():
			return ctx.Err()
		}
		return errors.Join(original.Err, publicReadback32(original.Output, func(int) float32 { return 7 }))
	}}, "launch original during published candidate check"); err != nil {
		cancel()
		<-done
		return err
	}
	result, err := f.service.Push(ctx, "concurrent", code(11), "candidate while original pinned")
	if err != nil || result == nil || !result.Published {
		cancel()
		return errors.Join(err, errors.New("concurrent candidate was not published"))
	}
	cfg := cudalaunch.Config{Grid: cudashape.Grid3D{X: 1, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: f.stream}
	work, err := f.service.Launch(ctx, "concurrent", cfg, buffers[1])
	if work != nil {
		err = errors.Join(err, work.Wait(ctx))
	}
	if err != nil || work == nil {
		return errors.Join(err, errors.New("concurrent selected submission missing"))
	}
	view, err := cudamem.Reinterpret[float32](buffers[1])
	if err != nil {
		return err
	}
	got := make([]float32, 32)
	if err := view.CopyToHost(got); err != nil {
		return err
	}
	if err := publicWriteJSON(filepath.Join(dir, "concurrent-pin-readback.json"), struct {
		Process                                 nativereceipt.Process
		Original, Selected, Submission, Checker State
		OriginalFunction                        uint64
		OriginalOutput, SelectedOutput          []float32
		OriginalDuringChecker, Completed        bool
	}{f.process, pin.State(), result.Selected, work.State(), checkerState, uint64(pin.Function()), original.Output, got, true, true}); err != nil {
		return err
	}
	if pin.State().Version == result.Selected.Version || work.State() != result.Selected || checkerState != result.State {
		return errors.New("concurrent retained and selected generation relation differs")
	}
	return publicReadback32(got, func(int) float32 { return 11 })
}

func TestPublicReadback32CPU(t *testing.T) {
	got := make([]float32, 32)
	for i := range got {
		got[i] = float32(i + 1)
	}
	want := func(i int) float32 { return float32(i + 1) }
	if err := publicReadback32(got, want); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		got  []float32
		want func(int) float32
	}{
		{"missing", nil, want}, {"short", got[:31], want}, {"oracle", got, nil}, {"wrong", got, func(int) float32 { return 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if publicReadback32(tc.got, tc.want) == nil {
				t.Fatal("invalid readback admitted")
			}
		})
	}
	got[31] = -123
	if publicReadback32(got, want) == nil {
		t.Fatal("last element corruption admitted")
	}
}

func TestPublicNativeCatalogueCPU(t *testing.T) {
	path := os.Getenv("HOTSWAP_PUBLIC_CATALOG")
	if path == "" {
		t.Skip("catalog export not requested")
	}
	if !filepath.IsAbs(path) {
		t.Fatal("absolute catalog path required")
	}
	catalog := []Source{
		{Code: publicAXPBY(false, -1), Filename: "baseline.cu", Entry: "axpby"},
		{Code: publicAXPBY(false, 1), Filename: "candidate.cu", Entry: "axpby"},
		{Code: publicAXPBY(true, 1), Filename: "candidate.cu", Entry: "axpby"},
		{Code: "extern \"C\" __global__ void axpby(int n) {}", Filename: "bad-abi.cu", Entry: "axpby"},
	}
	for _, v := range []int{7, 11} {
		catalog = append(catalog, Source{Code: fmt.Sprintf("extern \"C\" __global__ void pin_value(float *out) { out[threadIdx.x] = %d.0f; }", v), Filename: "pin.cu", Entry: "pin_value"})
	}
	for _, body := range []string{"out[i] = out[i] + a[i];", "out[i] = a[i] + out[i];"} {
		catalog = append(catalog, Source{Code: "extern \"C\" __global__ void inplace(float *out, const float *a, unsigned n) { unsigned i=threadIdx.x; if(i<n) { " + body + " } }", Filename: "inplace.cu", Entry: "inplace"})
	}
	catalog = append(catalog, Source{Code: "extern \"C\" __global__ void noop(float *out, const float *a, unsigned n) {}", Filename: "noop.cu", Entry: "noop"})
	for i, v := range []int{7, 11, 13} {
		catalog = append(catalog, Source{Code: fmt.Sprintf("extern \"C\" __global__ void graph_value(float *out) { out[threadIdx.x] = %d.0f; }", v), Filename: []string{"graph-a.cu", "graph-b.cu", "graph-wrong.cu"}[i], Entry: "graph_value"})
	}
	if err := publicWriteJSON(path, catalog); err != nil {
		t.Fatal(err)
	}
}
