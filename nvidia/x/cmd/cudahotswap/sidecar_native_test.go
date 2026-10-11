//go:build linux && native

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type sidecarNativeWatch struct {
	bytes.Buffer
	path, dir       string
	codes           []string
	index, received int
	cancel          context.CancelFunc
	proof           *nativereceipt.Session
	reports         []*cudahotswap.PushResult
	validate        func(int, *cudahotswap.PushResult, string) error
	recheck         func(cudahotswap.State) error
	pin             *cudahotswap.Pin
}

// Preparation compiles for an explicit target without CUDA device admission.
// Its output is a recorded compiler fixture, never a native execution receipt.
func TestSidecarPrepareSourceCPU(t *testing.T) {
	if os.Getenv("SIDECAR_PREPARE_ONLY") != "1" {
		t.Skip("SIDECAR_PREPARE_ONLY=1 enables CPU compiler fixture preparation")
	}
	dir := os.Getenv("SIDECAR_PREPARE_DIR")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute fresh SIDECAR_PREPARE_DIR required")
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	fixtures, err := sidecarCompileFixtures()
	if err != nil {
		t.Fatal(err)
	}
	for _, fixture := range fixtures {
		source, inputs, name := fixture.source, fixture.inputs, fixture.name
		path := filepath.Join(dir, name+".cu")
		if err := source.WriteFile(path); err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(inputs)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name+"-inputs.json"), data, 0600); err != nil {
			t.Fatal(err)
		}
		for _, physical := range []bool{false, true} {
			arch := nvrtcx.WithArch(12, 1)
			kind := nvrtcx.WithOutput(nvrtcx.OutputPTX)
			suffix := ".ptx"
			if physical {
				arch = nvrtcx.WithPhysicalArch(12, 1)
				kind = nvrtcx.WithOutput(nvrtcx.OutputCUBIN)
				suffix = ".cubin"
			}
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			module, err := nvrtcx.CompileContext(ctx, source.CUDA, nvrtcx.WithName(path), arch, kind, nvrtcx.WithLineInfo())
			cancel()
			if err != nil {
				t.Fatal(err)
			}
			body := []byte(module.PTX())
			if physical {
				body = module.CUBIN()
			}
			if len(body) == 0 {
				t.Fatal("compiler returned empty fixture")
			}
			if err := os.WriteFile(filepath.Join(dir, name+suffix), body, 0600); err != nil {
				t.Fatal(err)
			}
			deps, err := json.Marshal(module.Dependencies())
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, name+suffix+".dependencies.json"), deps, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func (w *sidecarNativeWatch) Write(data []byte) (int, error) {
	var report struct {
		Kernel string
		Result *cudahotswap.PushResult
		Error  string
	}
	// Retain raw bytes before validation so refusal evidence is not discarded.
	if err := os.WriteFile(filepath.Join(w.dir, fmt.Sprintf("raw-report-%d.json", w.received)), data, 0600); err != nil {
		return 0, err
	}
	w.received++
	w.Buffer.Write(data)
	if err := json.Unmarshal(data, &report); err != nil {
		return 0, err
	}
	if w.index >= 3 || report.Kernel != "kernel" || report.Result == nil {
		return 0, errors.New("sidecar watch lacks bounded checked report")
	}
	if err := w.validate(w.index, report.Result, report.Error); err != nil {
		return 0, err
	}
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(w.codes[w.index])))
	if err := w.proof.Capture(fmt.Sprintf("sidecar-report-%d", w.index), hash, report.Result.Report.Compile.PTXSHA256); err != nil {
		return 0, err
	}
	if err := os.WriteFile(filepath.Join(w.dir, fmt.Sprintf("report-%d.json", w.index)), data, 0600); err != nil {
		return 0, err
	}
	w.reports = append(w.reports, report.Result)
	if w.index == 1 {
		if err := sidecarValidateRestored(w.reports[0].State, report.Result.State, report.Result.Selected); err != nil {
			return 0, err
		}
		if err := w.recheck(report.Result.Selected); err != nil {
			return 0, err
		}
	}
	w.index++
	if w.index == 3 {
		w.cancel()
		return len(data), nil
	}
	temp := w.path + ".save"
	if err := os.WriteFile(temp, []byte(w.codes[w.index]), 0600); err != nil {
		return 0, err
	}
	if err := os.Rename(temp, w.path); err != nil {
		return 0, err
	}
	return len(data), nil
}

func (w *sidecarNativeWatch) releasePin(ctx context.Context) error {
	if w.pin == nil {
		return nil
	}
	if err := w.pin.Release(ctx); err != nil {
		return err
	}
	w.pin = nil
	return nil
}

func TestNativeSidecarCLIBarAndFused(t *testing.T) {
	if os.Getenv("HOTSWAP_SIDECAR_NATIVE") != "1" {
		t.Skip("HOTSWAP_SIDECAR_NATIVE=1 requires independent source/packet review and native dispatch")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	root := os.Getenv("HOTSWAP_SIDECAR_ARTIFACTS")
	if !filepath.IsAbs(root) {
		t.Fatal("absolute fresh HOTSWAP_SIDECAR_ARTIFACTS required")
	}
	if err := os.Mkdir(root, 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CACHE_HOME", filepath.Join(root, "compiler-cache"))
	sidecarNativeCases(t, func(t *testing.T, fused bool) {
		dir := filepath.Join(root, fmt.Sprintf("fused-%t", fused))
		if err := os.Mkdir(dir, 0700); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		if err := nativeSidecarCase(ctx, cancel, fused, dir); err != nil {
			t.Fatal(err)
		}
	})
}

func nativeSidecarCase(ctx context.Context, cancel context.CancelFunc, fused bool, dir string) (err error) {
	source, inputs, err := sidecarExpressionFixture(fused)
	if err != nil {
		return err
	}
	path := filepath.Join(dir, "chain.cu")
	if err = source.WriteFile(path); err != nil {
		return err
	}
	if err = os.WriteFile(filepath.Join(dir, "reference.cu"), []byte(source.CUDA), 0600); err != nil {
		return err
	}
	inputPath := filepath.Join(dir, "inputs.json")
	data, err := json.Marshal(inputs)
	if err != nil {
		return err
	}
	if err = os.WriteFile(inputPath, data, 0600); err != nil {
		return err
	}
	ref, err := loadSidecarReference(path, filepath.Join(dir, "chain.egraph.json"), inputPath)
	if err != nil {
		return err
	}
	metadata, err := sidecarReadFile(filepath.Join(dir, "chain.egraph.json"))
	if err != nil {
		return err
	}
	wrong := strings.Replace(source.CUDA, "eg_output[i] = eg_result[i];", "eg_output[i] = 0.0f;", 1)
	if wrong == source.CUDA {
		return errors.New("actual emitter store control missing")
	}
	codes := []string{"// watcher first edit\n" + source.CUDA, wrong, "// editor repair\n" + source.CUDA}
	for i, code := range codes {
		if err = os.WriteFile(filepath.Join(dir, fmt.Sprintf("candidate-%d.cu", i)), []byte(code), 0600); err != nil {
			return err
		}
	}
	proof, err := nativereceipt.New()
	if err != nil {
		return err
	}
	output := &sidecarNativeWatch{path: path, dir: dir, codes: codes, cancel: cancel, proof: proof}
	var stderr bytes.Buffer
	var rows []sidecarNativeReadback
	phase := "setup-unbound"
	var initial cudahotswap.State
	var cleanup func() error
	// This terminal path also runs after refusals. Missing library evidence is
	// retained as an explicit error, never turned into a successful receipt.
	defer func() {
		closeCtx, stop := context.WithTimeout(context.Background(), 10*time.Second)
		defer stop()
		err = errors.Join(err, output.releasePin(closeCtx))
		if cleanup != nil {
			err = errors.Join(err, cleanup())
		}
		var receiptErr error
		err = errors.Join(err, sidecarFinishNativeReceipt(
			func() error { return proof.Capture("terminal", "", "") },
			func() error { receiptErr = proof.Write(filepath.Join(dir, "native-libraries.json")); return receiptErr },
			proof.Close))
		if receiptErr != nil {
			err = errors.Join(err, os.WriteFile(filepath.Join(dir, "library-receipt-error.txt"), []byte(receiptErr.Error()+"\n"), 0600))
		}
		err = errors.Join(err, os.WriteFile(filepath.Join(dir, "stdout.jsonl"), output.Bytes(), 0600), os.WriteFile(filepath.Join(dir, "stderr.log"), stderr.Bytes(), 0600))
		if err != nil {
			err = errors.Join(err, os.WriteFile(filepath.Join(dir, "terminal-error.txt"), []byte(err.Error()+"\n"), 0600))
		}
	}()
	output.validate = func(index int, result *cudahotswap.PushResult, text string) error {
		if len(rows) == 0 {
			return errors.New("sidecar report has no actual readback")
		}
		return sidecarValidateNativeReport(index, result, text, codes[index], rows[len(rows)-1])
	}
	create := func(ctx context.Context, c config) (*cudahotswap.Service, func() error, error) {
		var app *sidecarApplication
		var boundService *cudahotswap.Service
		var images *sidecarImageLoader
		observe := func(state cudahotswap.State, out []float32) error {
			bits := make([]uint32, len(out))
			for i, value := range out {
				bits[i] = math.Float32bits(value)
			}
			row := sidecarNativeReadback{Phase: phase, State: state, OutputBits: bits}
			data, e := json.Marshal(row)
			if e != nil {
				return e
			}
			if e = os.WriteFile(filepath.Join(dir, fmt.Sprintf("actual-readback-%03d.json", len(rows))), data, 0600); e != nil {
				return e
			}
			rows = append(rows, row)
			if state.Generation != 0 {
				return sidecarRetainImage(ctx, dir, fmt.Sprintf("loaded-image-%03d", len(rows)-1), boundService, images, state)
			}
			return nil
		}
		service, close, e := newSidecarSessionLoader(ctx, c, observe, func(a *sidecarApplication) { app = a }, func(l cudahotswap.Loader) cudahotswap.Loader {
			images = newSidecarImageLoader(l, source.CUDA, metadata)
			images.attribute = func(ctx context.Context, dir, name string, record sidecarImageRecord, body []byte, compiled sidecarImageSource) error {
				return sidecarStaticAttribution(ctx, dir, name, record, body, compiled, metadata)
			}
			return images
		})
		boundService = service
		close = sidecarImageCleanup(close, images)
		cleanup = close
		if e != nil {
			return service, close, e
		}
		if app == nil {
			return service, close, errors.New("native sidecar setup did not publish ready hook")
		}
		initial, e = service.Status("kernel")
		if e != nil {
			return service, close, e
		}
		phase = "initial-verified"
		if e = app.check(ctx, initial); e != nil {
			return service, close, e
		}
		phase = "candidate"
		output.recheck = func(selected cudahotswap.State) (e error) {
			actual, e := service.Status("kernel")
			if e != nil {
				return e
			}
			if actual != selected {
				return errors.New("native rollback status differs from selected report")
			}
			pin, e := service.Pin("kernel")
			if pin != nil {
				output.pin = pin
			}
			if e != nil {
				return e
			}
			defer func() { e = errors.Join(e, output.releasePin(ctx)) }()
			if pin.State() != selected || pin.Function() == 0 {
				return errors.New("native restored pin differs")
			}
			phase = "restored"
			defer func() { phase = "candidate" }()
			if e = app.check(ctx, actual); e != nil {
				return e
			}
			after, e := service.Status("kernel")
			if e != nil {
				return e
			}
			if after != selected {
				return errors.New("native restored selection changed during actual check")
			}
			if len(rows) == 0 || rows[len(rows)-1].State != selected || rows[len(rows)-1].Phase != "restored" {
				return errors.New("native restored launch lacks actual stamped readback")
			}
			return nil
		}
		// Registration and its explicit initial check used the original source.
		// Change the actual watched file before Watch.Ready so the first scan
		// observes a real edit instead of waiting forever on unchanged bytes.
		e = sidecarFirstWatchEdit(path, codes[0])
		return service, close, e
	}
	err = run(ctx, []string{"-watch", path, "-entry", "chain", "-sidecar", filepath.Join(dir, "chain.egraph.json"), "-inputs", inputPath}, output, &stderr, create)
	if err != nil && !onlyCancellation(err) {
		return err
	}
	err = nil
	if output.index != 3 {
		return errors.New("incomplete actual consumer report sequence")
	}
	if err = sidecarValidateNativeRows(rows, output.reports, initial, ref.want); err != nil {
		return err
	}
	if fused {
		actual := make([]float32, len(rows[5].OutputBits))
		for i, bits := range rows[5].OutputBits {
			actual[i] = math.Float32frombits(bits)
		}
		compareCtx, stop := context.WithTimeout(context.Background(), 30*time.Second)
		defer stop()
		if err = nativeSidecarTemplateComparison(compareCtx, source, inputs, actual, dir, proof); err != nil {
			return err
		}
	}
	return nil
}

// Compare the real existing parallel template on the same explicit inputs.
// This uses its declared scalar dimensions/epsilon, not the structural ABI.
func nativeSidecarTemplateComparison(ctx context.Context, source egraph.Source, inputs map[string][]float32, emitted []float32, dir string, proof *nativereceipt.Session) (err error) {
	metadata, err := sidecarReadFile(filepath.Join(dir, "chain.egraph.json"))
	if err != nil {
		return err
	}
	ref, err := readSidecarReference(bytes.NewReader(metadata), source.CUDA, inputs)
	if err != nil {
		return err
	}
	code, err := egraph.GenerateCUDA(source.Expression, egraph.CodegenOptions{KernelName: "template_reference", Precision: "float", UseWarpSum: true})
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "template-reference.cu"), []byte(code), 0600); err != nil {
		return err
	}
	a := &sidecarApplication{owners: &sessionOwners{acquire: acquireSessionLock}, ref: ref}
	o := a.owners
	defer func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		err = errors.Join(err, a.close(cleanup))
	}()
	o.lease, err = o.acquire(ctx, "cudahotswap: template comparison setup")
	if err != nil {
		return err
	}
	if o.lease == nil {
		return errors.New("template setup missing lease")
	}
	o.dev, err = cuda.NewDevice(0)
	if err != nil {
		return err
	}
	o.stream, err = o.dev.NewStream()
	if err != nil {
		return err
	}
	values := [][]float32{inputs["a"], inputs["weight"], inputs["matrix"], make([]float32, len(ref.want))}
	for _, v := range values {
		b, e := cudamem.Alloc[byte](o.dev, 4*len(v))
		if b.Ptr() != 0 {
			o.buffers = append(o.buffers, b)
		}
		if e != nil {
			return e
		}
		view, e := cudamem.Reinterpret[float32](b)
		if e != nil {
			return e
		}
		if e = view.CopyFromHost(v); e != nil {
			return e
		}
	}
	if err = o.lease.Close(); err != nil {
		return err
	}
	o.lease = nil
	loader, err := cudahotswap.NewNVRTCLoader(o.dev)
	if err != nil {
		return err
	}
	images := newSidecarImageLoader(loader, source.CUDA, metadata)
	o.service, err = cudahotswap.NewService(images)
	if o.service != nil {
		o.serviceClose = func() error {
			if e := images.releasePins(context.Background()); e != nil {
				return e
			}
			return o.service.Close(context.Background())
		}
	}
	if err != nil {
		return err
	}
	state, registerErr := o.service.Register(ctx, "kernel", cudahotswap.Source{Code: code, Filename: filepath.Join(dir, "template-reference.cu"), Entry: "template_reference"}, "same-input legacy template")
	if registerErr != nil {
		err = registerErr
		return err
	}
	if err = sidecarRetainImage(ctx, dir, "template-loaded-image", o.service, images, state); err != nil {
		return err
	}
	output := o.buffers[3]
	a.write = func(v []float32) error {
		view, e := cudamem.Reinterpret[float32](output)
		if e != nil {
			return e
		}
		return view.CopyFromHost(v)
	}
	a.read = func(v []float32) error {
		view, e := cudamem.Reinterpret[float32](output)
		if e != nil {
			return e
		}
		return view.CopyToHost(v)
	}
	launch := cudalaunch.Config{Grid: cudashape.Grid3D{X: uint32(len(ref.want)), Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), Stream: o.stream}
	a.submit = func(ctx context.Context) (sidecarPending, error) {
		p, e := o.service.Launch(ctx, "kernel", launch, o.buffers[0], o.buffers[1], o.buffers[2], o.buffers[3], int32(len(inputs["a"])), int32(len(ref.want)), float32(1e-5))
		if p == nil {
			return nil, e
		}
		return p, e
	}
	a.observe = func(state cudahotswap.State, out []float32) error {
		bits := make([]uint32, len(out))
		for i, value := range out {
			bits[i] = math.Float32bits(value)
		}
		data, e := json.Marshal(sidecarNativeReadback{Phase: "template", State: state, OutputBits: bits})
		if e != nil {
			return e
		}
		if e = os.WriteFile(filepath.Join(dir, "template-actual-readback.json"), data, 0600); e != nil {
			return e
		}
		if e = compareSidecar(emitted, out); e != nil {
			return fmt.Errorf("emitted versus template (abs1e-5+rel1e-5): %w", e)
		}
		return nil
	}
	if err = a.check(ctx, state); err != nil {
		return err
	}
	return proof.Capture("same-input-template-complete", fmt.Sprintf("%x", sha256.Sum256([]byte(code))), "")
}

func TestSidecarPrepareTemplateCPU(t *testing.T) {
	if os.Getenv("SIDECAR_TEMPLATE_PREPARE_ONLY") != "1" {
		t.Skip("CPU compiler preparation only")
	}
	dir := os.Getenv("SIDECAR_PREPARE_DIR")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute fresh SIDECAR_PREPARE_DIR required")
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	source, _, err := sidecarExpressionFixture(true)
	if err != nil {
		t.Fatal(err)
	}
	code, err := egraph.GenerateCUDA(source.Expression, egraph.CodegenOptions{KernelName: "template_reference", Precision: "float", UseWarpSum: true})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "template-reference.cu")
	if err := os.WriteFile(path, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	for _, physical := range []bool{false, true} {
		arch, kind, suffix := nvrtcx.WithArch(12, 1), nvrtcx.WithOutput(nvrtcx.OutputPTX), ".ptx"
		if physical {
			arch, kind, suffix = nvrtcx.WithPhysicalArch(12, 1), nvrtcx.WithOutput(nvrtcx.OutputCUBIN), ".cubin"
		}
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		module, err := nvrtcx.CompileContext(ctx, code, nvrtcx.WithName(path), arch, kind, nvrtcx.WithLineInfo())
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		body := []byte(module.PTX())
		if physical {
			body = module.CUBIN()
		}
		if len(body) == 0 {
			t.Fatal("compiler returned empty template")
		}
		if err := os.WriteFile(filepath.Join(dir, "template-reference"+suffix), body, 0600); err != nil {
			t.Fatal(err)
		}
		deps, err := json.Marshal(module.Dependencies())
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "template-reference"+suffix+".dependencies.json"), deps, 0600); err != nil {
			t.Fatal(err)
		}
	}
}
