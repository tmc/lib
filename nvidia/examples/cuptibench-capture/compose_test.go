package cuptibenchcapture_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	cuptibenchcapture "github.com/tmc/lib/nvidia/examples/cuptibench-capture"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

const (
	fixtureKernel    = "compute"
	fixtureCapture   = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
	fixtureRun       = "00112233445566778899aabbccddeeff"
	fixtureProfile   = "pipeline.pprof"
	fixtureBenchmark = "BenchmarkPipeline/full"
)

// readArtifacts loads the compiled module fixture. The cubin is stored hex
// encoded so the capture's only ELF input stays reviewable text in git.
func readArtifacts(t *testing.T) cuptibenchcapture.Artifacts {
	t.Helper()
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin: %v", err)
	}
	return cuptibenchcapture.Artifacts{
		Prefix:      "kernels",
		Source:      read("kernels.cu"),
		PTX:         read("kernels.ptx"),
		SASS:        read("kernels.sass"),
		CUBIN:       cubin,
		ImageSHA256: sha256Hex(cubin),
		Name:        "pipeline",
		Generation:  1,
		Kernels:     []string{fixtureKernel},
	}
}

func sha256Hex(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }

// result builds one complete cuptibench result in kernel attribution mode.
func result() *cuptibench.Result {
	return &cuptibench.Result{
		Operations: 1,
		Timing:     &cuptibench.Timing{Operations: 1, Total: 250 * time.Microsecond, Method: "cuda-events"},
		Quality:    cuptirange.CollectionQuality{PassesStarted: 1, PassesStopped: 1, AllPassesSubmitted: true, Decoded: true},
		Ranges: []cuptibench.Range{{
			ID: "k0", Name: fixtureKernel, NativeName: "0", Kind: "kernel",
			Metrics: []cuptibench.Metric{
				{Name: "sm__ctas_launched.sum", Unit: "cta", Value: 258, Additive: true},
				{Name: "sm__throughput.avg.pct_of_peak_sustained_elapsed", Unit: "%", Value: 12.5},
			},
		}},
		Metadata: map[string]string{
			"capture_id": fixtureCapture, "benchmark": fixtureBenchmark, "run_id": fixtureRun,
			"mode": "kernels", "view": "full", "replay_mode": "kernel-replay",
			"metrics": "sm__ctas_launched.sum", "chip": "gb100", "device_uuid": "deadbeef",
			"cuda_driver_version": "13000", "cupti_version": "29",
			"preparation":       "immutable input; clear scratch; caches uncontrolled",
			"timing_capture_id": "timing-" + fixtureRun,
		},
	}
}

// export renders the two files a cuptibench -cuptiprofile destination produces.
// The pprof shape follows x/cuptibench/export.go: declared counter sample types
// and one sample per exported range, labelled with the capture's provenance.
func export(t *testing.T, results []*cuptibench.Result) cuptibenchcapture.Export {
	t.Helper()
	b := cudaprof.NewBuilder()
	schema := []cudaprof.SampleType{{Name: "sm__ctas_launched.sum", Unit: "cta"}}
	if err := b.SetDeclaredSampleTypes(schema...); err != nil {
		t.Fatal(err)
	}
	if err := b.SetDefaultSampleType(schema[0].Name); err != nil {
		t.Fatal(err)
	}
	var captures []string
	for _, r := range results {
		captures = append(captures, r.Metadata["capture_id"])
		for _, rng := range r.Ranges {
			labels := map[string]string{
				"capture_id": r.Metadata["capture_id"], "benchmark": r.Metadata["benchmark"],
				"run_id": r.Metadata["run_id"], "mode": r.Metadata["mode"], "view": r.Metadata["view"],
				"range-id": rng.ID, "range-kind": rng.Kind, "range-name": rng.Name, "kernel": rng.Name,
			}
			frames := []cudaprof.Frame{{FunctionName: rng.Name}, {FunctionName: r.Metadata["benchmark"]}}
			if err := b.AddDeclaredSample([]int64{258}, frames, labels); err != nil {
				t.Fatal(err)
			}
		}
	}
	var buf bytes.Buffer
	if err := b.WriteGzip(&buf); err != nil {
		t.Fatal(err)
	}
	manifest := struct {
		Version  int
		Profiles []struct {
			Path     string
			Captures []string
		}
		Results []*cuptibench.Result
	}{Version: 1, Results: results}
	manifest.Profiles = append(manifest.Profiles, struct {
		Path     string
		Captures []string
	}{Path: fixtureProfile, Captures: captures})
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return cuptibenchcapture.Export{Name: fixtureProfile, Profile: buf.Bytes(), Results: data}
}

// TestComposeParsesBothForms covers the acceptance requirement that the raw
// counter profile and the packaged capture both parse, and that a reader
// holding only the capture resolves source and disassembly offline.
func TestComposeParsesBothForms(t *testing.T) {
	a := readArtifacts(t)
	e := export(t, []*cuptibench.Result{result()})

	if _, err := profile.ParseData(e.Profile); err != nil {
		t.Fatalf("raw counter profile does not parse: %v", err)
	}
	c, err := cuptibenchcapture.Compose(e, a)
	if err != nil {
		t.Fatalf("Compose: %v", err)
	}
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, c); err != nil {
		t.Fatalf("Write: %v", err)
	}
	// Everything below sees only the capture file, as cmd/cudaprof does.
	read, err := cudacapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(read.Modules) != 1 {
		t.Fatalf("got %d modules, want 1", len(read.Modules))
	}
	id := read.Modules[0].ID
	if want := "sha256:" + a.ImageSHA256; id != want {
		t.Errorf("module ID = %q, want %q", id, want)
	}
	code, err := cudacapture.VerifyModule(read, id, a.ImageSHA256, []string{fixtureKernel})
	if err != nil {
		t.Fatalf("VerifyModule: %v", err)
	}
	k := code.Kernel(fixtureKernel)
	if k == nil {
		t.Fatalf("kernel %q not in capture", fixtureKernel)
	}
	if k.SourceLine == 0 || len(k.Instructions) == 0 {
		t.Fatalf("kernel %q resolved without source line or instructions", fixtureKernel)
	}
	if got := string(read.Files[code.SourceFile]); !strings.Contains(got, "void "+fixtureKernel) {
		t.Errorf("retained source does not define %q", fixtureKernel)
	}
	if _, ok := read.Files["kernels.cuptibench.json"]; !ok {
		t.Error("capture does not retain the cuptibench results file")
	}
}

// TestComposeSeparatesTimingFromCounters covers the requirement that the timed
// execution and the counter replay stay separately identified: timing has its
// own descriptors and cuptibench's own timing capture identity, and a counter
// already carried by the profile is not repeated as an observation.
func TestComposeSeparatesTimingFromCounters(t *testing.T) {
	c, err := cuptibenchcapture.Compose(export(t, []*cuptibench.Result{result()}), readArtifacts(t))
	if err != nil {
		t.Fatalf("Compose: %v", err)
	}
	byID := make(map[string]cudacapture.MetricDescriptor)
	for _, d := range c.Descriptors {
		byID[d.ID] = d
	}
	scopes := map[string]string{}
	for _, o := range c.Observations {
		d := byID[o.DescriptorID]
		scopes[d.Scope] = o.EntityID
		if d.Name == "sm__ctas_launched.sum" {
			t.Errorf("counter %q is both a profile sample and an observation", d.Name)
		}
	}
	if got := scopes["benchmark"]; got != "timing-"+fixtureRun {
		t.Errorf("timing entity = %q, want %q", got, "timing-"+fixtureRun)
	}
	if got := scopes["range"]; !strings.HasPrefix(got, fixtureCapture+"/") {
		t.Errorf("range entity = %q, want a range of capture %q", got, fixtureCapture)
	}
	for _, name := range []string{"timing_total", "timing_operations"} {
		found := false
		for _, d := range c.Descriptors {
			if d.Name == name && d.Window == "uninstrumented timed execution" {
				found = true
			}
		}
		if !found {
			t.Errorf("no descriptor %q for the separately timed execution", name)
		}
	}
}

// TestComposeRefusesDisagreement covers the acceptance requirement that a
// failed or partial capture is not published, and that a capture whose code and
// counters describe different work is refused rather than written.
func TestComposeRefusesDisagreement(t *testing.T) {
	tests := []struct {
		name string
		// run mutates the inputs before export; export mutates the exported
		// files afterwards. Faults are injected one at a time, so everything
		// else in the fixture still agrees.
		run    func(*cuptibenchcapture.Artifacts, *cuptibench.Result)
		export func(*cuptibenchcapture.Export)
		want   string
	}{
		{name: "dropped ranges", run: func(_ *cuptibenchcapture.Artifacts, r *cuptibench.Result) {
			r.Quality.DroppedRanges = 1
		}, want: "incomplete or lossy"},
		{name: "unfinished passes", run: func(_ *cuptibenchcapture.Artifacts, r *cuptibench.Result) {
			r.Quality.PassesStopped = 0
		}, want: "incomplete or lossy"},
		{name: "undecoded counters", run: func(_ *cuptibenchcapture.Artifacts, r *cuptibench.Result) {
			r.Quality.Decoded = false
		}, want: "incomplete or lossy"},
		{name: "missing provenance", run: func(_ *cuptibenchcapture.Artifacts, r *cuptibench.Result) {
			delete(r.Metadata, "run_id")
		}, want: "missing provenance"},
		{name: "other module", run: func(a *cuptibenchcapture.Artifacts, _ *cuptibench.Result) {
			a.CUBIN = append([]byte(nil), a.CUBIN...)
			a.CUBIN[len(a.CUBIN)-1] ^= 0xff
		}, want: "sha256 is"},
		{name: "unrecorded image", run: func(a *cuptibenchcapture.Artifacts, _ *cuptibench.Result) {
			a.ImageSHA256 = ""
		}, want: "missing loaded image digest"},
		{name: "missing disassembly", run: func(a *cuptibenchcapture.Artifacts, _ *cuptibench.Result) {
			a.SASS = nil
		}, want: "missing sass artifact"},
		{name: "kernel not in module", run: func(a *cuptibenchcapture.Artifacts, _ *cuptibench.Result) {
			a.Kernels = []string{"project"}
		}, want: "does not define kernel"},
		{name: "profile kernel not in module", run: func(_ *cuptibenchcapture.Artifacts, r *cuptibench.Result) {
			r.Ranges[0].Name = "silu"
		}, want: "does not define kernel"},
		{name: "another run's profile", export: func(e *cuptibenchcapture.Export) {
			e.Name = "other.pprof"
		}, want: "do not name profile"},
		{name: "unknown export schema", export: func(e *cuptibenchcapture.Export) {
			e.Results = bytes.Replace(e.Results, []byte(`"Version": 1`), []byte(`"Version": 2`), 1)
		}, want: "unsupported cuptibench export version"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, r := readArtifacts(t), result()
			if tt.run != nil {
				tt.run(&a, r)
			}
			e := export(t, []*cuptibench.Result{r})
			if tt.export != nil {
				tt.export(&e)
			}
			_, err := cuptibenchcapture.Compose(e, a)
			if err == nil {
				t.Fatalf("Compose succeeded, want error containing %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Compose error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

// TestComposeAcceptsTheUnmutatedFixture keeps the refusal table honest: the
// same fixture, with no fault injected, must compose.
func TestComposeAcceptsTheUnmutatedFixture(t *testing.T) {
	if _, err := cuptibenchcapture.Compose(export(t, []*cuptibench.Result{result()}), readArtifacts(t)); err != nil {
		t.Fatalf("Compose: %v", err)
	}
}

func ExampleCompose() {
	// Compose refuses inputs whose provenance it cannot check, rather than
	// writing a capture that looks complete.
	_, err := cuptibenchcapture.Compose(cuptibenchcapture.Export{}, cuptibenchcapture.Artifacts{Prefix: "kernels"})
	fmt.Println(err)
	// Output:
	// cuptibenchcapture: missing source artifact
}
