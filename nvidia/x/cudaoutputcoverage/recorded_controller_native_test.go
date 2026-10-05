//go:build linux && native

package cudaoutputcoverage_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// This controller does not initialize CUDA and must remain outside any lease.
// Each independent RunRecorded attempt acquires its own real lease. These
// explicit three arms are not a campaign or pristine/candidate comparison.
func TestRecordedOutputNativeConsumer(t *testing.T) {
	if os.Getenv("RECORDED_OUTPUT_NATIVE") != "1" {
		t.Skip("owned native recorded output dispatch not selected")
	}
	var opts cudaschedule.RecordedRunOptions
	if err := recordedReadJSON(os.Getenv("RECORDED_OUTPUT_OPTIONS"), &opts); err != nil {
		t.Fatal(err)
	}
	if opts.CompareOutput || !opts.RequireOutput || opts.OracleAttached {
		t.Fatal("three-arm output fixture requires output, no comparison or instrumentation claim")
	}
	if len(opts.Args) != 0 {
		t.Fatal("worker arguments belong to this closed fixture")
	}
	packet := os.Getenv("RECORDED_OUTPUT_PACKET")
	if !filepath.IsAbs(packet) || !filepath.IsAbs(opts.ArtifactDir) {
		t.Fatal("absolute packet/artifact paths required")
	}
	source, err := recordedImage(filepath.Join(packet, "output.cu"))
	if err != nil {
		t.Fatal(err)
	}
	if string(source) != recordedOutputSource {
		t.Fatal("prepared source differs from owned fixed kernel")
	}
	var prepared struct {
		SourceDigest, PTXDigest, NVRTCPath, Arch string
		NVRTCMajor, NVRTCMinor                   int32
	}
	if err := recordedReadJSON(filepath.Join(packet, "prepared.json"), &prepared); err != nil {
		t.Fatal(err)
	}
	ptx, err := recordedImage(filepath.Join(packet, "output.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	if prepared.SourceDigest != recordedHash(source) || prepared.PTXDigest != recordedHash(ptx) || prepared.Arch != "sm_121" || prepared.NVRTCPath == "" || prepared.NVRTCMajor <= 0 {
		t.Fatal("prepared compiler/source/PTX binding differs")
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	executable, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	if opts.Executable != exe || opts.ExecutableSHA256 != recordedHash(executable) {
		t.Fatal("worker must be this exact prebuilt testbinary")
	}
	if err := os.MkdirAll(opts.ArtifactDir, 0700); err != nil {
		t.Fatal(err)
	}
	if opts.PristinePTX != string(ptx) {
		t.Fatal("supplied pristine PTX differs from actual prepared image")
	}
	for i, arm := range []string{"floor", "corrupt", "healthy"} {
		if !t.Run(arm, func(t *testing.T) {
			selected := opts
			selected.Args = []string{"-test.run=^TestRecordedOutputWorker$", "-recorded-output-arm=" + arm}
			selected.Label = opts.Label + "-" + arm
			rec, err := cudaschedule.RunRecorded(context.Background(), uint64(i), string(ptx), selected)
			verdict := cudaschedule.Classify(rec)
			body, marshalErr := json.MarshalIndent(struct {
				Run     cudaschedule.RunRecord
				Verdict cudaschedule.Verdict
			}{rec, verdict}, "", "  ")
			if marshalErr != nil {
				t.Fatal(marshalErr)
			}
			if writeErr := os.WriteFile(filepath.Join(opts.ArtifactDir, "output-"+arm+".json"), body, 0600); writeErr != nil {
				t.Fatal(writeErr)
			}
			if err != nil {
				t.Fatal(err)
			}
			if rec.Custody == nil || !rec.Custody.Closed || !rec.Custody.WorkerReaped || !rec.Custody.StdoutDrained || !rec.Custody.StderrDrained || rec.Output == nil {
				t.Fatalf("incomplete actual worker/cleanup evidence: %+v", rec)
			}
			if rec.PTXDigest != recordedHash(ptx) || rec.PTXDigest != rec.ExpectedPTXDigest || rec.Output.PTXDigest != rec.PTXDigest {
				t.Fatal("loaded PTX binding differs")
			}
			report, err := cudaoutputcoverage.CheckOutput(rec.Output.TwoFill, rec.Output.Ordinary, rec.Output.Policy)
			if err != nil {
				t.Fatal(err)
			}
			if arm == "floor" {
				if len(report.Written.Unwritten) != 1 || report.Written.Unwritten[0] != (cudaoutputcoverage.Range{Start: 768, End: 1003}) || report.NumericalMismatch != nil || verdict != cudaschedule.VerdictMismatch {
					t.Fatalf("floor actual evidence: %+v %s", report, verdict)
				}
			}
			if arm == "corrupt" {
				if len(report.Written.Unwritten) != 0 || report.NumericalMismatch == nil || verdict != cudaschedule.VerdictMismatch {
					t.Fatalf("corrupt actual evidence: %+v %s", report, verdict)
				}
			}
			if arm == "healthy" {
				if len(report.Written.Unwritten) != 0 || report.NumericalMismatch != nil || verdict != cudaschedule.VerdictPass {
					t.Fatalf("healthy actual evidence: %+v %s", report, verdict)
				}
			}
		}) {
			return
		}
	}
}

var retainedRecordedVariantCompileOwners []io.Closer

// Compile-only: no driver, device probe or real lock is used.
func TestRecordedOutputVariantsPrepare(t *testing.T) {
	dir := os.Getenv("RECORDED_OUTPUT_VARIANTS_PREPARE")
	if dir == "" {
		t.Skip("compile-only variant packet not selected")
	}
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute fresh variant packet required")
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	registry := recordedCompilerRegistry{Arch: "sm_121", NVRTCPath: nvrtc.Path(), NVRTCMajor: major, NVRTCMinor: minor}
	sources := recordedOutputSources()
	for _, name := range []string{"healthy", "floor", "corrupt"} {
		source := sources[name]
		compiled, err := nvrtcx.Compile(source, nvrtcx.WithArch(12, 1))
		if err != nil {
			var owner io.Closer
			if errors.As(err, &owner) {
				if closeErr := owner.Close(); closeErr != nil {
					retainedRecordedVariantCompileOwners = append(retainedRecordedVariantCompileOwners, owner)
					t.Logf("retain compiler cleanup owner: %v", closeErr)
				}
			}
			t.Fatal(err)
		}
		ptx := []byte(compiled.PTX())
		if len(ptx) == 0 {
			t.Fatal("empty actual compiler PTX")
		}
		for filename, body := range map[string][]byte{name + ".cu": []byte(source), name + ".ptx": ptx, name + ".log": []byte(compiled.Log())} {
			if err := os.WriteFile(filepath.Join(dir, filename), body, 0600); err != nil {
				t.Fatal(err)
			}
		}
		registry.Outputs = append(registry.Outputs, recordedCompiledOutput{Name: name, SourceDigest: recordedHash([]byte(source)), PTXDigest: recordedHash(ptx)})
	}
	b, err := json.MarshalIndent(registry, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "registry.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("actual compile-only source/PTX registry %+v", registry)
}

// Actual CompareOutput baseline/candidate consumer. It is not RunCampaign.
func TestRecordedOutputBaselineCandidateNative(t *testing.T) {
	if os.Getenv("RECORDED_OUTPUT_COMPARE_NATIVE") != "1" {
		t.Skip("owned baseline/candidate native dispatch not selected")
	}
	var opts cudaschedule.RecordedRunOptions
	if err := recordedReadJSON(os.Getenv("RECORDED_OUTPUT_OPTIONS"), &opts); err != nil {
		t.Fatal(err)
	}
	if !opts.CompareOutput || !opts.RequireOutput || opts.OracleAttached || len(opts.Args) != 0 {
		t.Fatal("comparison requires output, fixed fixture args and no instrumentation claim")
	}
	dir := os.Getenv("RECORDED_OUTPUT_REGISTRY")
	healthy, err := recordedImage(filepath.Join(dir, "healthy.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	registered, err := recordedRegisteredOutput(dir, healthy)
	if err != nil || registered.Name != "healthy" {
		t.Fatal("actual healthy baseline not registered", err)
	}
	if opts.PristinePTX != string(healthy) {
		t.Fatal("pristine input differs from actual registered healthy PTX")
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	executable, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	if opts.Executable != exe || opts.ExecutableSHA256 != recordedHash(executable) {
		t.Fatal("worker must be this exact prebuilt testbinary")
	}
	if !filepath.IsAbs(opts.ArtifactDir) {
		t.Fatal("absolute artifacts required")
	}
	if err := os.MkdirAll(opts.ArtifactDir, 0700); err != nil {
		t.Fatal(err)
	}
	opts.Args = []string{"-test.run=^TestRecordedOutputWorker$", "-recorded-output-arm=healthy", "-recorded-output-registry=" + dir}
	for i, name := range []string{"floor", "corrupt", "healthy"} {
		if !t.Run(name, func(t *testing.T) {
			supplied, err := recordedImage(filepath.Join(dir, name+".ptx"))
			if err != nil {
				t.Fatal(err)
			}
			selected, err := recordedRegisteredOutput(dir, supplied)
			if err != nil {
				t.Fatal(err)
			}
			rec, err := cudaschedule.RunRecorded(context.Background(), uint64(i), string(supplied), opts)
			verdict := cudaschedule.Classify(rec)
			b, marshalErr := json.MarshalIndent(struct {
				Run     cudaschedule.RunRecord
				Verdict cudaschedule.Verdict
			}{rec, verdict}, "", "  ")
			if marshalErr != nil {
				t.Fatal(marshalErr)
			}
			if writeErr := os.WriteFile(filepath.Join(opts.ArtifactDir, "compare-"+name+".json"), b, 0600); writeErr != nil {
				t.Fatal(writeErr)
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, run := range []*cudaschedule.RunRecord{rec.Baseline, &rec} {
				if run == nil || run.Output == nil || run.Custody == nil || !run.Custody.Closed || !run.Custody.WorkerReaped || !run.Custody.StdoutDrained || !run.Custody.StderrDrained {
					t.Fatal("actual baseline/candidate worker retention incomplete")
				}
				if run.Output.TwoFill.Geometry.Grid != ([3]uint32{4, 1, 1}) || run.Output.TwoFill.Geometry.Block != ([3]uint32{256, 1, 1}) {
					t.Fatal("fixed worker geometry changed")
				}
			}
			if cudaschedule.Classify(*rec.Baseline) != cudaschedule.VerdictPass || rec.Baseline.PTXDigest != recordedHash(healthy) || rec.PTXDigest != recordedHash(supplied) || rec.Output.TwoFill.SourceDigest != selected.SourceDigest || rec.Baseline.Output.TwoFill.SourceDigest != registered.SourceDigest {
				t.Fatal("actual pristine/candidate source and loaded PTX binding differs")
			}
			report, err := cudaoutputcoverage.CheckOutput(rec.Output.TwoFill, rec.Output.Ordinary, rec.Output.Policy)
			if err != nil {
				t.Fatal(err)
			}
			want := cudaschedule.VerdictMismatch
			if name == "healthy" {
				want = cudaschedule.VerdictPass
			}
			if verdict != want {
				t.Fatalf("actual output verdict %s want %s", verdict, want)
			}
			if name == "floor" && (len(report.Written.Unwritten) != 1 || report.Written.Unwritten[0] != (cudaoutputcoverage.Range{Start: 768, End: 1003}) || report.NumericalMismatch != nil) {
				t.Fatal("source-encoded bounded-store control differs", report)
			}
			if name == "corrupt" && (len(report.Written.Unwritten) != 0 || report.NumericalMismatch == nil) {
				t.Fatal("source-encoded wrong-value control differs", report)
			}
			if name == "healthy" && (len(report.Written.Unwritten) != 0 || report.NumericalMismatch != nil) {
				t.Fatal("source-encoded positive control differs", report)
			}
		}) {
			return
		}
	}
}
