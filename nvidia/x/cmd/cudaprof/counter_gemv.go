package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"runtime"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof/counters"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

type counterGEMVOptions struct {
	operation                               string
	rows, cols, iterations, device          int
	implementation, selection, arm, metrics string
	passes                                  uint
	token                                   int
}

func runCounterGEMVCommand(args []string) error { return runCounterGEMV(args, os.Stdout) }
func runCounterGEMV(args []string, out io.Writer) (err error) {
	if len(args) == 0 || args[0] == "-h" || args[0] == "-help" || args[0] == "--help" {
		fmt.Fprintln(out, "Usage: cudaprof counter-gemv workload|collect [options]")
		return nil
	}
	o, err := parseCounterGEMV(args[0], args[1:], out)
	if err != nil || o == nil {
		return err
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		return errors.New("counter-gemv: GPU_LOCK_HELD is unset; use one existing gpu-lock hold")
	}
	var selection counterSelectionReport
	if o.operation == "collect" {
		data, _, err := readCounterJSONFile(o.selection)
		if err != nil {
			return err
		}
		if err := decodeCounterJSON(data, &selection); err != nil {
			return err
		}
		if selection.Version != 1 || selection.Selection == nil || selection.Failure != "" || len(selection.Source.Arms) != 2 {
			return errors.New("counter-gemv: qualified selection receipt required")
		}
		if !reflect.DeepEqual(selection.Selection.Identity, counterGEMVIdentity(*o, o.token)) {
			return errors.New("counter-gemv: workload identity differs from historical selection")
		}
		arm := 0
		if o.arm == "target" {
			arm = 1
		}
		if !selection.Source.Shape.Pass || !counterReceiptSHA(selection.Source.Arms[arm].SHA256) {
			return errors.New("counter-gemv: shape qualification or capture identity unavailable")
		}
		exact, err := counters.Select(selection.Source.Shape, selection.Selection.Identity)
		if err != nil {
			return err
		}
		if !reflect.DeepEqual(exact, *selection.Selection) {
			return errors.New("counter-gemv: selection differs from qualified source receipt")
		}
		if err := revalidateCounterGEMVSelection(selection); err != nil {
			return err
		}
	}
	dev, err := cuda.NewDevice(o.device)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, dev.Close()) }()
	workload, err := newCounterGEMV(dev, *o)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, workload.close()) }()
	if o.operation == "workload" {
		for token := 0; token < o.iterations; token++ {
			if err := workload.prepare(); err != nil {
				return err
			}
			if _, err := workload.run(counterGEMVIdentity(*o, token)); err != nil {
				return err
			}
		}
		fmt.Fprintf(out, "gemv-ok rows=%d cols=%d iterations=%d\n", o.rows, o.cols, o.iterations)
		return nil
	}
	arm := 0
	if o.arm == "target" {
		arm = 1
	}
	path, err := os.Executable()
	if err != nil {
		return err
	}
	binarySHA, err := campaignHash(path)
	if err != nil {
		return err
	}
	names, err := probeMetricNames(o.metrics)
	if err != nil {
		return err
	}
	id := selection.Selection.Identity
	opts := counters.CollectOptions{Selection: *selection.Selection, Arm: o.arm, CaptureSHA256: selection.Source.Arms[arm].SHA256, ExpectedOutputSHA256: workload.expectedSHA, ToolVersion: "executable-sha256:" + binarySHA, Metrics: names, MaxPasses: o.passes, LockDeclared: true}
	var controls []counterGEMVControl
	control := func() error {
		start := time.Now()
		if err := workload.prepare(); err != nil {
			return err
		}
		witness, err := workload.run(id)
		controls = append(controls, counterGEMVControl{WallNanos: time.Since(start).Nanoseconds(), OutputSHA256: witness.OutputSHA256, Pass: err == nil})
		return err
	}
	if err := control(); err != nil {
		return err
	}
	start := time.Now()
	result, collectErr := counters.Collect(dev, opts, counters.Replay{Prepare: func(uint) error { return workload.prepare() }, Run: func(uint) (counters.Witness, error) { return workload.run(id) }, Synchronize: driver.CtxSynchronize})
	collectionWall := time.Since(start).Nanoseconds()
	afterErr := control()
	report := struct {
		Version                   int                    `json:"version"`
		HistoricalSource          counterSelectionReport `json:"historical_selection_receipt"`
		SourceCapturesRevalidated bool                   `json:"source_captures_revalidated"`
		Implementation            string                 `json:"implementation"`
		SourceSHA256              string                 `json:"compiled_source_sha256"`
		InputSHA256               string                 `json:"restored_input_sha256"`
		ExpectedOutputSHA256      string                 `json:"expected_output_sha256"`
		Collection                *counters.Collection   `json:"collection"`
		Controls                  []counterGEMVControl   `json:"ordinary_controls"`
		CollectionWall            int64                  `json:"end_to_end_collection_wall_ns"`
		Limits                    string                 `json:"limits"`
	}{1, selection, true, o.implementation, workload.sourceSHA, workload.inputSHA, workload.expectedSHA, result, controls, collectionWall, "controlled matrix/vector workload, not integrated Gemma evidence; every ordinary/replay output checked against exact CPU result; collection wall includes query/configuration/replay/validation/cleanup and is not kernel profiling overhead; historical selection revalidated against original capture bytes and decoded shape comparison; input/source hashes distinguish actual replay workloads"}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return errors.Join(collectErr, afterErr, enc.Encode(report))
}

type counterGEMVControl struct {
	WallNanos    int64  `json:"wall_ns"`
	OutputSHA256 string `json:"completed_output_sha256"`
	Pass         bool   `json:"correctness_pass"`
}

func parseCounterGEMV(operation string, args []string, out io.Writer) (*counterGEMVOptions, error) {
	if operation != "workload" && operation != "collect" {
		return nil, fmt.Errorf("counter-gemv: unknown operation %q", operation)
	}
	o := &counterGEMVOptions{operation: operation}
	fs := flag.NewFlagSet("counter-gemv "+operation, flag.ContinueOnError)
	fs.SetOutput(out)
	fs.Usage = func() { fmt.Fprintf(out, "Usage: cudaprof counter-gemv %s [options]\n", operation); fs.PrintDefaults() }
	fs.IntVar(&o.device, "device", 0, "CUDA device ordinal")
	fs.IntVar(&o.rows, "rows", 128, "matrix rows (1 to 1024)")
	fs.IntVar(&o.cols, "cols", 256, "matrix columns (1 to 8192)")
	fs.StringVar(&o.implementation, "implementation", "simple", "simple or unroll, with the same operation and geometry")
	fs.IntVar(&o.iterations, "iterations", 4, "ordinary workload marker count (2 to 1024)")
	fs.StringVar(&o.selection, "selection", "", "qualified counter-receipt select JSON (required for collect)")
	fs.StringVar(&o.arm, "arm", "base", "historical base or target launch")
	fs.IntVar(&o.token, "token", 0, "exact selected token index")
	fs.StringVar(&o.metrics, "metrics", "", "comma-separated native metrics (required for collect)")
	fs.UintVar(&o.passes, "max-passes", 16, "maximum user replay passes (1 to 64)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil, nil
		}
		return nil, err
	}
	if fs.NArg() != 0 || o.device < 0 || o.rows < 1 || o.rows > 1024 || o.cols < 1 || o.cols > 8192 || o.rows*o.cols > 4<<20 || o.iterations < 2 || o.iterations > 1024 || o.token < 0 || o.token >= 1024 || o.passes == 0 || o.passes > 64 || o.implementation != "simple" && o.implementation != "unroll" || o.arm != "base" && o.arm != "target" {
		return nil, errors.New("counter-gemv: invalid workload or replay bounds")
	}
	if operation == "collect" {
		if o.selection == "" || o.metrics == "" {
			return nil, errors.New("counter-gemv: collect requires selection and metrics")
		}
		if _, err := probeMetricNames(o.metrics); err != nil {
			return nil, err
		}
	}
	return o, nil
}

func counterGEMVIdentity(o counterGEMVOptions, token int) shape.Identity {
	layer := 0
	shared := uint32(0)
	no := false
	return shape.Identity{Phase: "decode", Token: token, Layer: &layer, Operation: fmt.Sprintf("controlled_gemv_%dx%d", o.rows, o.cols), DType: "f32", CacheLength: 0, Grid: [3]int{(o.rows + 127) / 128, 1, 1}, Block: [3]int{128, 1, 1}, SharedMemory: &shared, Lookahead: &no, Materialization: &no}
}

const counterGEMVSource = `extern "C" __global__ void cudaprof_gemv(const float* a,const float* x,float* y,int rows,int cols) {
 int row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows)return;
 float sum=0;
 int col=0;
#if UNROLL
 for(;col+3<cols;col+=4) {
  sum+=a[row*cols+col]*x[col];
  sum+=a[row*cols+col+1]*x[col+1];
  sum+=a[row*cols+col+2]*x[col+2];
  sum+=a[row*cols+col+3]*x[col+3];
 }
#endif
 for(;col<cols;col++)sum+=a[row*cols+col]*x[col];
 y[row]=sum;
}`

type counterGEMVWorkload struct {
	dev                              *cuda.Device
	o                                counterGEMVOptions
	fn                               driver.Function
	a, x, y                          cudamem.Slice[float32]
	matrix, vector, expected         []float32
	sourceSHA, inputSHA, expectedSHA string
	closeModule                      func() error
}

func counterGEMVInputs(rows, cols int) (matrix, vector, expected []float32) {
	matrix = make([]float32, rows*cols)
	vector = make([]float32, cols)
	expected = make([]float32, rows)
	for j := range vector {
		vector[j] = 1
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v := float32((i+j)%7 - 3)
			matrix[i*cols+j] = v
			expected[i] += v
		}
	}
	return
}
func counterGEMVFloatHash(values ...[]float32) string {
	h := sha256.New()
	var b [4]byte
	for _, vs := range values {
		for _, v := range vs {
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
func newCounterGEMV(dev *cuda.Device, o counterGEMVOptions) (w *counterGEMVWorkload, err error) {
	w = &counterGEMVWorkload{dev: dev, o: o}
	defer func() {
		if err != nil {
			err = errors.Join(err, w.close())
		}
	}()
	source := "#define UNROLL 0\n" + counterGEMVSource
	if o.implementation == "unroll" {
		source = "#define UNROLL 1\n" + counterGEMVSource
	}
	digest := sha256.Sum256([]byte(source))
	w.sourceSHA = hex.EncodeToString(digest[:])
	w.matrix, w.vector, w.expected = counterGEMVInputs(o.rows, o.cols)
	w.inputSHA = counterGEMVFloatHash(w.matrix, w.vector)
	w.expectedSHA = counterGEMVFloatHash(w.expected)
	cm, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		return w, err
	}
	module, err := cm.Load(dev)
	if err != nil {
		return w, err
	}
	w.closeModule = func() error { return dev.WithContext(module.Close) }
	if err = dev.WithContext(func() error { var e error; w.fn, e = module.Function("cudaprof_gemv"); return e }); err != nil {
		return w, err
	}
	w.a, err = cudamem.Alloc[float32](dev, len(w.matrix))
	if err != nil {
		return w, err
	}
	w.x, err = cudamem.Alloc[float32](dev, len(w.vector))
	if err != nil {
		return w, err
	}
	w.y, err = cudamem.Alloc[float32](dev, len(w.expected))
	return w, err
}
func (w *counterGEMVWorkload) close() error {
	var errs []error
	if w.a.Ptr() != 0 {
		errs = append(errs, w.a.Free())
	}
	if w.x.Ptr() != 0 {
		errs = append(errs, w.x.Free())
	}
	if w.y.Ptr() != 0 {
		errs = append(errs, w.y.Free())
	}
	if w.closeModule != nil {
		errs = append(errs, w.closeModule())
	}
	return errors.Join(errs...)
}
func (w *counterGEMVWorkload) prepare() error {
	poison := make([]float32, len(w.expected))
	for i := range poison {
		poison[i] = -999
	}
	return errors.Join(w.a.CopyFromHost(w.matrix), w.x.CopyFromHost(w.vector), w.y.CopyFromHost(poison))
}
func (w *counterGEMVWorkload) run(id shape.Identity) (witness counters.Witness, err error) {
	dtype := id.DType
	cache := id.CacheLength
	token := id.Token
	metadata := marker.Metadata{Phase: id.Phase, Token: &token, Layer: id.Layer, Operation: id.Operation, DType: &dtype, CacheLength: &cache, Lookahead: id.Lookahead, Materialization: id.Materialization}
	text, err := marker.Encode(metadata)
	if err != nil {
		return witness, err
	}
	err = w.dev.WithContext(func() error {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		nvtx.RangePush(text)
		launchErr := cudalaunch.Kernel(w.fn).Grid(uint32(id.Grid[0]), uint32(id.Grid[1]), uint32(id.Grid[2])).Block(uint32(id.Block[0]), uint32(id.Block[1]), uint32(id.Block[2])).Launch(w.a, w.x, w.y, int32(w.o.rows), int32(w.o.cols))
		nvtx.RangePop()
		if launchErr != nil {
			return launchErr
		}
		return driver.CtxSynchronize()
	})
	if err != nil {
		return witness, err
	}
	values := make([]float32, len(w.expected))
	if err = w.y.CopyToHost(values); err != nil {
		return witness, err
	}
	witness = counters.Witness{Identity: id, OutputSHA256: counterGEMVFloatHash(values)}
	if witness.OutputSHA256 != w.expectedSHA {
		return witness, errors.New("counter-gemv: completed GPU output differs from exact CPU result")
	}
	return witness, nil
}

func revalidateCounterGEMVSelection(report counterSelectionReport) error {
	if report.Selection == nil || len(report.Source.Arms) != 2 {
		return errors.New("counter-gemv: missing historical source captures")
	}
	var records [2]timeline.Records
	for i, arm := range report.Source.Arms {
		if arm.Path == "" {
			return errors.New("counter-gemv: original source capture path is required")
		}
		cap, hash, err := readAlignmentCapture(arm.Path, cudacapture.Limits{})
		if err != nil {
			return fmt.Errorf("counter-gemv source %d: %w", i, err)
		}
		if hash != arm.SHA256 {
			return errors.New("counter-gemv: original capture bytes differ from historical receipt")
		}
		_, records[i], err = decodeCompleteCapture(cap, 0, 0)
		if err != nil {
			return err
		}
	}
	fresh := shape.CompareScoped(records[0], records[1], report.Source.Domain, report.Source.Shape.RequestPairs)
	selection, err := counters.Select(fresh, report.Selection.Identity)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(selection, *report.Selection) {
		return errors.New("counter-gemv: historical selection differs from redecoded captures")
	}
	return nil
}
