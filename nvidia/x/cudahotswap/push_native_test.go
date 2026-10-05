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
	base    *NVRTCLoader
	dir     string
	next    int
	receipt *nativereceipt.Session
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
	return c, nil
}

func (l *publicRecordingLoader) Load(ctx context.Context, c *Compilation) (Kernel, error) {
	return l.base.Load(ctx, c)
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
	if _, err := f.service.Push(context.Background(), "kernel", Source{Code: "extern \"C\" __global__ void axpby(int n) {}", Filename: "bad-abi.cu", Entry: "axpby"}, "bad ABI"); err == nil {
		t.Fatal("actual public ABI mismatch was admitted")
	}
	if _, err := f.service.Revert(context.Background(), "kernel", "baseline"); err != nil {
		t.Fatal(err)
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
	read := func(want float32) error {
		view, err := cudamem.Reinterpret[float32](f.buffers[0])
		if err != nil {
			return err
		}
		out := make([]float32, 32)
		if err := view.CopyToHost(out); err != nil {
			return err
		}
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
		return read(11)
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
	if err := read(11); err != nil {
		t.Fatal("graph callback rollback output: ", err)
	}
	if err := f.raw[0].Launch(f.stream); err != nil {
		t.Fatal(err)
	}
	if err := f.stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := read(7); err != nil {
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
	if err := read(7); err != nil {
		t.Fatal("public Revert graph output: ", err)
	}
	if err := publicWriteJSON(filepath.Join(dir, "graph.json"), struct {
		Patched, Skipped, Reverted int
		Push                       *PushResult
	}{11, 7, 7, result}); err != nil {
		t.Fatal(err)
	}
}

func TestNativeServiceTwentyPushLatency(t *testing.T) {
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
