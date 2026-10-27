package cudadiagdemo

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

var errNumericDownload = errors.New("download failed")

// numericFault selects faults by plan index; -1 selects none. The wrong
// attempt downloads an incorrect last element. The fail attempt downloads
// half its output, records the partly written destination and fails.
type numericFault struct {
	wrong, fail int
	partial     []float32
}

// numericFaultStream wraps one fakeNative stream with fault.
type numericFaultStream struct {
	nativeStream
	fault *numericFault
	index int
}

func (s *numericFaultStream) upload(input []float32) error {
	s.index = int(input[0])
	return s.nativeStream.upload(input)
}

func (s *numericFaultStream) download(output []float32) error {
	if err := s.nativeStream.download(output); err != nil {
		return err
	}
	switch s.index {
	case s.fault.wrong:
		output[len(output)-1]++
	case s.fault.fail:
		clear(output[len(output)/2:])
		s.fault.partial = slices.Clone(output)
		return errNumericDownload
	}
	return nil
}

// numericExecutor returns f's executor retaining vectors, as RunNative
// configures it for an OutputDir, with its streams wrapped by fault.
func numericExecutor(f *fakeNative, fault *numericFault) *nativeExecutor {
	e := f.executor()
	e.retain = true
	e.open = func(own *nativeCleanup) (nativeStream, error) {
		s, err := f.open(own)
		if err != nil {
			return nil, err
		}
		return &numericFaultStream{nativeStream: s, fault: fault}, nil
	}
	return e
}

func noNumericFault() *numericFault { return &numericFault{wrong: -1, fail: -1} }

// numericAttempts runs plan through e and records the attempts as RunNative
// does with OutputDir dir.
func numericAttempts(t *testing.T, ctx context.Context, e *nativeExecutor, workers int, plan []nativeRequest, dir string) (*NativeResult, error) {
	t.Helper()
	numerics, err := nativeNumericsFor(NativeConfig{Elements: e.elements, Rounds: len(plan) / len(nativeArms), OutputDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	result := &NativeResult{PlannedRequests: len(plan), NumericalGate: "unavailable", Numerics: numerics, cleanup: &nativeCleanup{}}
	attempts, err := e.run(ctx, workers, plan, result.cleanup)
	result.recordAttempts(attempts)
	return result, err
}

// numericRun also exports the result to dir, returning both errors.
func numericRun(t *testing.T, ctx context.Context, e *nativeExecutor, workers int, plan []nativeRequest, dir string) (result *NativeResult, runErr, exportErr error) {
	t.Helper()
	result, runErr = numericAttempts(t, ctx, e, workers, plan, dir)
	return result, runErr, exportNative(dir, result, nil)
}

type numericReceipt struct {
	Numerics       NativeNumerics `json:"numerics"`
	Trials         []NativeTrial  `json:"trials"`
	NumericalGate  string         `json:"numerical_gate"`
	RemainingGates []string       `json:"remaining_gates"`
}

func readNumericReceipt(t *testing.T, dir string) numericReceipt {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, "native-receipt.json"))
	if err != nil {
		t.Fatal(err)
	}
	var r numericReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// readVector reads the file of ref, checks that ref binds its plan name, role,
// size and digest, and decodes it independently of encodeFloats. A nil ref
// returns nil.
func readVector(t *testing.T, ref *cudaregress.ArtifactRef, plan int, role string) []float32 {
	t.Helper()
	if ref == nil {
		return nil
	}
	data, err := os.ReadFile(ref.Path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	name := fmt.Sprintf("numeric-%03d-%s.f32", plan, role)
	if ref.Name != name || filepath.Base(ref.Path) != name || ref.Role != role || ref.SizeBytes != int64(len(data)) || len(data) != 4*8 || ref.Digest != hex.EncodeToString(sum[:]) {
		t.Fatalf("%s ref %+v does not bind its %d bytes", name, ref, len(data))
	}
	values := make([]float32, len(data)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	return values
}

// numericWant computes 2*x+1 for each decoded input value.
func numericWant(input []float32) []float32 {
	out := make([]float32, len(input))
	for i, x := range input {
		out[i] = float32(2*x) + 1
	}
	return out
}

func sameBits(a, b []float32) bool {
	return slices.EqualFunc(a, b, func(x, y float32) bool { return math.Float32bits(x) == math.Float32bits(y) })
}

// checkCorrect verifies a correct attempt's retained vectors against its
// trial digests and an independent reference.
func checkCorrect(t *testing.T, v NativeVectors, trial NativeTrial) {
	t.Helper()
	in := readVector(t, v.Input, v.Plan, "input")
	ref := readVector(t, v.Reference, v.Plan, "reference")
	out := readVector(t, v.Output, v.Plan, "output")
	if !sameBits(in, nativeInput(8, v.Plan)) || !sameBits(ref, numericWant(in)) || !sameBits(out, ref) {
		t.Fatalf("attempt %d vectors: input %v reference %v output %v", v.Plan, in, ref, out)
	}
	if v.Input.Digest != trial.InputSHA256 || v.Output.Digest != trial.OutputSHA256 || !trial.Correct || v.Error != "" || v.RequestID != trial.RequestID || v.Attempt != trial.Attempt {
		t.Fatalf("attempt %d vectors %+v do not match trial %+v", v.Plan, v, trial)
	}
}

func TestNativeNumericsRetained(t *testing.T) {
	ctx := testContext(t)
	dir := t.TempDir()
	plan := planNativeRequests(1, 2)
	_, runErr, exportErr := numericRun(t, ctx, numericExecutor(&fakeNative{}, noNumericFault()), 1, plan, dir)
	if runErr != nil || exportErr != nil {
		t.Fatal(runErr, exportErr)
	}
	r := readNumericReceipt(t, dir)
	n := r.Numerics
	if n.Retention != "retained" || n.Error != "" || n.Operation != nativeOperation || n.Encoding != nativeEncoding || n.PlannedBytes != 36*8 || len(n.Vectors) != len(plan) || len(r.Trials) != len(plan) {
		t.Fatalf("numerics = %+v", n)
	}
	for i, v := range n.Vectors {
		if v.Plan != i {
			t.Fatalf("vectors %d have plan index %d", i, v.Plan)
		}
		checkCorrect(t, v, r.Trials[i])
	}
	files, err := filepath.Glob(filepath.Join(dir, "numeric-*"))
	if err != nil || len(files) != 3*len(plan) {
		t.Fatalf("retained files %v, %v", files, err)
	}
}

func TestNativeNumericsDistinctConcurrentRequests(t *testing.T) {
	ctx := testContext(t)
	const workers = 3
	b := newBarrier(workers)
	f := &fakeNative{synchronize: func(context.Context, int) error { return b.wait(ctx) }}
	dir := t.TempDir()
	plan := planNativeRequests(2, 4)
	_, runErr, exportErr := numericRun(t, ctx, numericExecutor(f, noNumericFault()), workers, plan, dir)
	if runErr != nil || exportErr != nil || f.maxInflight != workers {
		t.Fatalf("run %v export %v max in-flight %d", runErr, exportErr, f.maxInflight)
	}
	r := readNumericReceipt(t, dir)
	if len(r.Numerics.Vectors) != len(plan) {
		t.Fatalf("retained %d of %d attempts", len(r.Numerics.Vectors), len(plan))
	}
	inputs, outputs := map[string]bool{}, map[string]bool{}
	for i, v := range r.Numerics.Vectors {
		if v.Plan != i {
			t.Fatalf("vectors %d have plan index %d", i, v.Plan)
		}
		checkCorrect(t, v, r.Trials[i])
		if inputs[v.Input.Digest] || outputs[v.Output.Digest] {
			t.Fatalf("attempt %d shares vectors with another request", i)
		}
		inputs[v.Input.Digest], outputs[v.Output.Digest] = true, true
	}
}

func TestNativeNumericsWrongOutput(t *testing.T) {
	ctx := testContext(t)
	dir := t.TempDir()
	fault := noNumericFault()
	fault.wrong = 1
	result, runErr, exportErr := numericRun(t, ctx, numericExecutor(&fakeNative{}, fault), 1, planNativeRequests(1, 2), dir)
	if runErr == nil || !strings.Contains(runErr.Error(), "native output 7") || exportErr != nil {
		t.Fatalf("run %v export %v", runErr, exportErr)
	}
	r := readNumericReceipt(t, dir)
	if r.Numerics.Retention != "retained" || r.NumericalGate != "failed" || len(r.Numerics.Vectors) != 2 || result.NumericalGate != "failed" {
		t.Fatalf("receipt = %+v", r)
	}
	checkCorrect(t, r.Numerics.Vectors[0], r.Trials[0])
	v, trial := r.Numerics.Vectors[1], r.Trials[1]
	in := readVector(t, v.Input, 1, "input")
	ref := readVector(t, v.Reference, 1, "reference")
	out := readVector(t, v.Output, 1, "output")
	if !sameBits(in, nativeInput(8, 1)) || !sameBits(ref, numericWant(in)) {
		t.Fatalf("reference %v is not 2*input+1 for input %v", ref, in)
	}
	if out == nil || sameBits(out, ref) || !sameBits(out[:7], ref[:7]) || out[7] != ref[7]+1 {
		t.Fatalf("incorrect output %v not retained beside reference %v", out, ref)
	}
	if trial.Correct || trial.Error == "" || v.Error != trial.Error || v.Output.Digest != trial.OutputSHA256 || v.Input.Digest != trial.InputSHA256 {
		t.Fatalf("failure not retained: vectors %+v trial %+v", v, trial)
	}
}

func TestNativeNumericsDownloadFailure(t *testing.T) {
	ctx := testContext(t)
	dir := t.TempDir()
	fault := noNumericFault()
	fault.fail = 1
	result, runErr, exportErr := numericRun(t, ctx, numericExecutor(&fakeNative{}, fault), 1, planNativeRequests(2, 2), dir)
	if !errors.Is(runErr, errNumericDownload) || !errors.Is(runErr, errNativeRequestFailed) || !strings.Contains(runErr.Error(), "4 of 6 not dispatched") || exportErr != nil {
		t.Fatalf("run %v export %v", runErr, exportErr)
	}
	if len(fault.partial) != 8 || fault.partial[0] == 0 || fault.partial[7] != 0 {
		t.Fatalf("destination %v was not partly written", fault.partial)
	}
	r := readNumericReceipt(t, dir)
	if r.Numerics.Retention != "retained" || len(r.Trials) != 2 || len(r.Numerics.Vectors) != 2 || len(result.vectors) != 2 || result.vectors[1].output != nil {
		t.Fatalf("receipt = %+v", r)
	}
	checkCorrect(t, r.Numerics.Vectors[0], r.Trials[0])
	v, trial := r.Numerics.Vectors[1], r.Trials[1]
	in := readVector(t, v.Input, 1, "input")
	if ref := readVector(t, v.Reference, 1, "reference"); !sameBits(ref, numericWant(in)) || v.Output != nil || trial.OutputSHA256 != "" || !strings.Contains(v.Error, errNumericDownload.Error()) {
		t.Fatalf("failed download retained as %+v, trial %+v", v, trial)
	}
	if _, err := os.Lstat(filepath.Join(dir, "numeric-001-output.f32")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("failed download exported: %v", err)
	}
	// Undispatched attempts 2 through 5 ran nothing and retain nothing.
	files, err := filepath.Glob(filepath.Join(dir, "numeric-*"))
	if err != nil || len(files) != 5 {
		t.Fatalf("retained files %v, %v", files, err)
	}
}

func TestNativeNumericsBound(t *testing.T) {
	tests := []struct {
		name            string
		elements, round int
		limit           int64
		want            int64
	}{
		{"exact limit", 8, 2, 576, 576},
		{"one below limit", 8, 2, 575, -1},
		{"largest admitted at cap", 29127, 64, nativeMaxNumericBytes, 67108608},
		{"smallest refused at cap", 29128, 64, nativeMaxNumericBytes, -1},
		{"max elements one round", 1 << 20, 1, nativeMaxNumericBytes, 37748736},
		{"max elements two rounds", 1 << 20, 2, nativeMaxNumericBytes, -1},
		{"overflow", math.MaxInt, math.MaxInt, math.MaxInt64, -1},
		{"negative", -1, 1, nativeMaxNumericBytes, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := nativeNumericBytes(tt.elements, tt.round, tt.limit)
			if tt.want < 0 {
				if !errors.Is(err, errNativeNumericBound) || got != 0 {
					t.Fatalf("nativeNumericBytes = %d, %v; want bound refusal", got, err)
				}
			} else if err != nil || got != tt.want {
				t.Fatalf("nativeNumericBytes = %d, %v; want %d", got, err, tt.want)
			}
		})
	}
}

func TestNativeNumericsRefusedBeforeLease(t *testing.T) {
	dir := t.TempDir()
	// A lease under a regular file cannot be created, so without the bound
	// RunNative would fail at the lease, before any device work.
	blocker := filepath.Join(dir, "file")
	if err := os.WriteFile(blocker, nil, 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out")
	cfg := NativeConfig{Elements: 29128, Rounds: 64, Concurrency: 1, OutputDir: out, LockDir: filepath.Join(blocker, "lock")}
	result, err := RunNative(testContext(t), cfg)
	if result != nil || !errors.Is(err, errNativeNumericBound) {
		t.Fatalf("RunNative = %+v, %v; want refusal before lease", result, err)
	}
	if _, err := os.Lstat(out); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("refused run created OutputDir: %v", err)
	}
	for _, tt := range []struct {
		elements int
		want     int64
	}{{29127, 67108608}, {29128, 0}} {
		n, err := nativeNumericsFor(NativeConfig{Elements: tt.elements, Rounds: 64, OutputDir: out})
		if n.PlannedBytes != tt.want || (err == nil) != (tt.want > 0) {
			t.Fatalf("%d elements: planned %d, %v; want %d", tt.elements, n.PlannedBytes, err, tt.want)
		}
	}
}

func TestNativeNumericsOutputDirEmpty(t *testing.T) {
	n, err := nativeNumericsFor(NativeConfig{Elements: 1 << 20, Rounds: 64})
	if err != nil || n.Retention != nativeRetentionNone || n.PlannedBytes != 0 {
		t.Fatalf("numerics = %+v, %v", n, err)
	}
	// RunNative retains vectors only with an OutputDir.
	e := (&fakeNative{}).executor()
	attempts, err := e.run(testContext(t), 1, planNativeRequests(1, 2), &nativeCleanup{})
	if err != nil {
		t.Fatal(err)
	}
	result := &NativeResult{Numerics: n}
	result.recordAttempts(attempts)
	for i, a := range attempts {
		if a.vectors != nil || !a.trial.Correct || a.trial.InputSHA256 != digestFloats(nativeInput(8, i)) || a.trial.OutputSHA256 != digestFloats(numericWant(nativeInput(8, i))) {
			t.Fatalf("attempt %d = %+v", i, a.trial)
		}
	}
	data, err := json.Marshal(result)
	if err != nil || len(result.vectors) != 0 || !bytes.Contains(data, []byte(`"retention":"none: OutputDir empty; no vector files retained"`)) || bytes.Contains(data, []byte("numeric-")) {
		t.Fatalf("receipt %s, %v", data, err)
	}
}

func TestNativeNumericsFileCollision(t *testing.T) {
	ctx := testContext(t)
	dir := t.TempDir()
	target := filepath.Join(t.TempDir(), "target")
	if err := os.WriteFile(target, []byte("keep"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(dir, "numeric-001-output.f32")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "numeric-002-input.f32"), []byte("old"), 0644); err != nil {
		t.Fatal(err)
	}
	_, runErr, exportErr := numericRun(t, ctx, numericExecutor(&fakeNative{}, noNumericFault()), 1, planNativeRequests(1, 2), dir)
	if runErr != nil || !errors.Is(exportErr, fs.ErrExist) {
		t.Fatalf("run %v export %v", runErr, exportErr)
	}
	r := readNumericReceipt(t, dir)
	n := r.Numerics
	if n.Retention != "incomplete" || !strings.Contains(n.Error, "numeric-001-output.f32") || !strings.Contains(n.Error, "numeric-002-input.f32") || !slices.Contains(r.RemainingGates, "complete numerical vector retention") || len(n.Vectors) != 3 {
		t.Fatalf("collision not visible: %+v gates %v", n, r.RemainingGates)
	}
	checkCorrect(t, n.Vectors[0], r.Trials[0])
	if v := n.Vectors[1]; v.Output != nil || v.Input == nil || v.Reference == nil {
		t.Fatalf("attempt 1 vectors = %+v", v)
	}
	if v := n.Vectors[2]; v.Input != nil || v.Reference == nil || v.Output == nil {
		t.Fatalf("attempt 2 vectors = %+v", v)
	}
	if data, err := os.ReadFile(target); err != nil || string(data) != "keep" {
		t.Fatalf("symlink followed: %q, %v", data, err)
	}
	if fi, err := os.Lstat(filepath.Join(dir, "numeric-001-output.f32")); err != nil || fi.Mode()&fs.ModeSymlink == 0 {
		t.Fatalf("symlink replaced: %v, %v", fi, err)
	}
	if data, err := os.ReadFile(filepath.Join(dir, "numeric-002-input.f32")); err != nil || string(data) != "old" {
		t.Fatalf("existing file overwritten: %q, %v", data, err)
	}
}

// numericFaultFile writes half of its data and fails, writes half of its
// data and reports no error if short, or fails its close after closing the
// file.
type numericFaultFile struct {
	io.WriteCloser
	write, close error
	short        bool
}

func (f *numericFaultFile) Write(p []byte) (int, error) {
	if f.write != nil || f.short {
		n, _ := f.WriteCloser.Write(p[:len(p)/2])
		return n, f.write
	}
	return f.WriteCloser.Write(p)
}

func (f *numericFaultFile) Close() error { return errors.Join(f.WriteCloser.Close(), f.close) }

func TestNativeNumericsWriteFailure(t *testing.T) {
	ctx := testContext(t)
	dir := t.TempDir()
	result, err := numericAttempts(t, ctx, numericExecutor(&fakeNative{}, noNumericFault()), 1, planNativeRequests(1, 2), dir)
	if err != nil {
		t.Fatal(err)
	}
	errWrite, errClose := errors.New("write failed"), errors.New("close failed")
	create := func(path string) (io.WriteCloser, error) {
		w, err := createNativeFile(path)
		if err != nil {
			return nil, err
		}
		switch filepath.Base(path) {
		case "numeric-000-reference.f32":
			return &numericFaultFile{WriteCloser: w, write: errWrite}, nil
		case "numeric-001-input.f32":
			return &numericFaultFile{WriteCloser: w, short: true}, nil
		case "numeric-002-output.f32":
			return &numericFaultFile{WriteCloser: w, close: errClose}, nil
		}
		return w, nil
	}
	err = retainNativeVectors(dir, result, create)
	n := result.Numerics
	if !errors.Is(err, errWrite) || !errors.Is(err, errClose) || !errors.Is(err, io.ErrShortWrite) || n.Retention != "incomplete" || n.Error != err.Error() || !slices.Contains(result.RemainingGates, "complete numerical vector retention") {
		t.Fatalf("retain = %v, numerics %+v", err, n)
	}
	if n.Vectors[0].Reference != nil || n.Vectors[0].Input == nil || n.Vectors[2].Output != nil || n.Vectors[2].Input == nil || n.Vectors[1].Input != nil || n.Vectors[1].Output == nil {
		t.Fatalf("failed writes referenced: %+v", n.Vectors)
	}
	// Failed files stay in place, unreferenced.
	for name, size := range map[string]int64{"numeric-000-reference.f32": 16, "numeric-001-input.f32": 16, "numeric-002-output.f32": 32} {
		if fi, err := os.Stat(filepath.Join(dir, name)); err != nil || fi.Size() != size {
			t.Fatalf("%s: %v, %v; want %d bytes left in place", name, fi, err, size)
		}
	}
}

func TestNativeNumericsFailedDrainNoResubmit(t *testing.T) {
	ctx := testContext(t)
	errClose := errors.New("stream close failed")
	failures := 1
	f := &fakeNative{close: func(name string, index int) error {
		if name == "stream" && index == 1 && failures > 0 {
			failures--
			return errClose
		}
		return nil
	}}
	dir := t.TempDir()
	result, runErr, exportErr := numericRun(t, ctx, numericExecutor(f, noNumericFault()), 1, planNativeRequests(1, 2), dir)
	if !errors.Is(runErr, errClose) || exportErr != nil || len(result.cleanup.actions) != 3 {
		t.Fatalf("run %v export %v retained %d owners", runErr, exportErr, len(result.cleanup.actions))
	}
	r := readNumericReceipt(t, dir)
	if len(r.Numerics.Vectors) != 2 {
		t.Fatalf("numerics = %+v", r.Numerics)
	}
	v := r.Numerics.Vectors[1]
	out := readVector(t, v.Output, 1, "output")
	if !sameBits(out, numericWant(readVector(t, v.Input, 1, "input"))) || !strings.Contains(v.Error, "cleanup") {
		t.Fatalf("attempt with failed drain retained as %+v", v)
	}
	if err := result.Close(); err != nil || result.CleanupPending {
		t.Fatalf("Close = %v pending=%t", err, result.CleanupPending)
	}
	if f.opened != 2 || f.launches != 2 {
		t.Fatalf("retry resubmitted work: opened=%d launches=%d", f.opened, f.launches)
	}
	if files, err := filepath.Glob(filepath.Join(dir, "numeric-*")); err != nil || len(files) != 6 {
		t.Fatalf("retained files %v, %v", files, err)
	}
}
