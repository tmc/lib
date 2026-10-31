package cudadiag_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

func probeReportFixture(t *testing.T) (cudadiag.ProbeReportOptions, []cudatracehost.Result) {
	t.Helper()
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry _Z1fv() {ret;}"), "_Z1fv", cudakernel.Schema{}, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	process := cudaprobe.Process{PID: 12, Generation: 30}
	o := cudadiag.ProbeReportOptions{
		Manifest: cudatracehost.Manifest{Process: process, Backend: "bpf", Clock: "linux_monotonic", Epoch: "recorded-boot", LossStatus: "unknown", Complete: false},
		ThreadID: 14, ThreadGeneration: 44, ExecID: 9, AddressSpace: 1, ModuleGeneration: 7,
		FirstSequence: 1, LastSequence: 2, LossKnown: true,
		Received: time.Unix(1700000000, 0), Threshold: 100 * time.Nanosecond,
		MaxBytes: 64 << 10, MaxLineBytes: 16 << 10, MaxRecords: 8, MaxOutliers: 4,
		Bindings: []cudaprobe.Binding{{Process: process, AddressSpace: 1, Function: 99, ModuleGeneration: 7, First: 1, Last: 2, Evidence: "recorded caller registration; not authenticated", Artifact: a}},
	}
	var results []cudatracehost.Result
	for i, span := range [][2]uint64{{100, 250}, {300, 320}} {
		id := cudaprobe.Identity{Process: process, ThreadID: 14, ThreadGeneration: 44, ThreadGenerationKnown: true, AddressSpace: 1, Provenance: "recorded-policy-fixture", Library: "/recorded/libcuda.so", BuildID: "recorded-build", Version: "fixture-v1", Symbol: "cuLaunchKernel", ABI: "linux-amd64-sysv-v1"}
		capture := &cudaprobe.Capture{Identity: id}
		binary.LittleEndian.PutUint16(capture.Data[:], 1)
		binary.LittleEndian.PutUint16(capture.Data[2:], 3)
		binary.LittleEndian.PutUint64(capture.Data[8:], uint64(i+1))
		binary.LittleEndian.PutUint64(capture.Data[16:], span[0])
		binary.LittleEndian.PutUint64(capture.Data[32:], 99)
		for j := 1; j <= 6; j++ {
			binary.LittleEndian.PutUint64(capture.Data[32+j*8:], 1)
		}
		results = append(results, cudatracehost.Result{Raw: "retained synthetic call pair", Capture: capture,
			Call: &cudatracehost.CallInterval{Process: process, TID: 14, ThreadGeneration: 44, ThreadGenerationKnown: true, ExecID: 9, PairingStatus: "verified_thread_lifetime", EntryTimestamp: span[0], ReturnTimestamp: span[1], Clock: "linux_monotonic", Epoch: "recorded-boot"}})
	}
	return o, results
}

func probeReportJSON(t *testing.T, results []cudatracehost.Result) []byte {
	t.Helper()
	var out bytes.Buffer
	for _, r := range results {
		if err := json.NewEncoder(&out).Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	return out.Bytes()
}

func TestProbeReportOutliers(t *testing.T) {
	o, results := probeReportFixture(t)
	input := probeReportJSON(t, results)
	r, err := cudadiag.ReadProbeReport(context.Background(), bytes.NewReader(input), o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Records != 2 || len(r.RawSHA256) != 64 || !r.Partial || r.Gap || len(r.Outliers) != 1 {
		t.Fatalf("report=%+v", r)
	}
	x := r.Outliers[0]
	if x.Sequence != 1 || x.HostDurationNanos != 150 || x.EntryTimestamp != 100 || x.ReturnTimestamp != 250 || x.Record.KernelName != "_Z1fv" || x.Record.Labels["kernel.digest"] != o.Bindings[0].Artifact.Digest() || x.Record.Labels["duration_status"] != "unavailable" || x.Record.Labels["kernel.binding_scope"] != "caller_consistency_only" {
		t.Fatalf("outlier=%+v", x)
	}
	// Threshold is inclusive; a short host interval is not selected as an outlier.
	o.Threshold = 151 * time.Nanosecond
	r, err = cudadiag.ReadProbeReport(context.Background(), bytes.NewReader(input), o)
	if err != nil || len(r.Outliers) != 0 {
		t.Fatalf("short control=%+v, %v", r, err)
	}
}

func TestProbeReportPairing(t *testing.T) {
	for _, status := range []string{"verified_thread_lifetime", "unverified_thread_lifetime", ""} {
		t.Run(status, func(t *testing.T) {
			o, results := probeReportFixture(t)
			// Change only the structured call's pairing status. Capture identity,
			// the outer result, bindings and caller context remain positive.
			results[0].Call.PairingStatus = status
			r, err := cudadiag.ReadProbeReport(context.Background(), bytes.NewReader(probeReportJSON(t, results)), o)
			if status == "" {
				if err == nil || r != nil {
					t.Fatalf("missing pairing admitted: %+v, %v", r, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if status == "unverified_thread_lifetime" {
				if !r.Partial || !r.Gap || r.Records != 2 || len(r.Outliers) != 0 {
					t.Fatalf("unverified pairing admitted interval or identity: %+v", r)
				}
			} else if r.Gap || len(r.Outliers) != 1 || r.Outliers[0].Record.KernelName != "_Z1fv" || r.Outliers[0].HostDurationNanos != 150 {
				t.Fatalf("verified pairing lost: %+v", r)
			}
		})
	}
}

func TestProbeReportIdentity(t *testing.T) {
	for _, name := range []string{"missing", "stale", "wrong-process", "overlap", "function", "binding-space", "module", "unknown-module", "capture-space-zero", "context-space", "thread", "exec"} {
		t.Run(name, func(t *testing.T) {
			o, results := probeReportFixture(t)
			switch name {
			case "missing":
				o.Bindings = nil
			case "stale":
				o.Bindings[0].First, o.Bindings[0].Last = 3, 4
			case "wrong-process":
				o.Bindings[0].Process.PID++
			case "overlap":
				o.Bindings = append(o.Bindings, o.Bindings[0])
			case "function":
				o.Bindings[0].Function++
			case "binding-space":
				o.Bindings[0].AddressSpace++
			case "module":
				o.ModuleGeneration++
			case "unknown-module":
				o.ModuleGeneration = 0
			case "capture-space-zero":
				results[0].Capture.Identity.AddressSpace = 0
			case "context-space":
				o.AddressSpace++
			case "thread":
				o.ThreadGeneration++
			case "exec":
				o.ExecID++
			}
			r, err := cudadiag.ReadProbeReport(context.Background(), bytes.NewReader(probeReportJSON(t, results)), o)
			if err != nil || len(r.Outliers) != 1 {
				t.Fatalf("host outlier discarded: %+v, %v", r, err)
			}
			x := r.Outliers[0]
			if x.HostDurationNanos != 150 || x.Record.KernelName != "" || x.Record.Labels["kernel.digest"] != "" || x.Record.Labels["kernel_identity"] != "unavailable" || x.Record.Labels["kernel_identity_reason"] == "" {
				t.Fatalf("unadmitted identity: %+v", x)
			}
			if name == "capture-space-zero" && x.Record.Labels["probe.address_space"] != "0" {
				t.Fatal("filled unknown capture address space")
			}
		})
	}
}

func TestProbeReportLoss(t *testing.T) {
	for _, name := range []string{"drop", "unknown-loss", "gap", "sequence", "end-gap", "pairing", "rescan"} {
		t.Run(name, func(t *testing.T) {
			o, results := probeReportFixture(t)
			switch name {
			case "drop":
				o.Dropped = 1
			case "unknown-loss":
				o.LossKnown = false
			case "gap":
				o.Gap = true
			case "sequence":
				binary.LittleEndian.PutUint64(results[1].Capture.Data[8:], 3)
			case "end-gap":
				results = results[:1]
			case "pairing":
				results[1].PairingStatus = "withheld_dynamic_thread_coverage"
			case "rescan":
				results[1].ThreadRescan = &cudatracehost.ThreadRescan{}
			}
			r, err := cudadiag.ReadProbeReport(context.Background(), bytes.NewReader(probeReportJSON(t, results)), o)
			if err != nil || len(r.Outliers) != 1 || r.Outliers[0].Record.KernelName != "" || r.Outliers[0].Record.Labels["kernel_identity_reason"] == "" {
				t.Fatalf("loss admitted identity: %+v, %v", r, err)
			}
		})
	}
}

func TestProbeReportInputRefusals(t *testing.T) {
	for _, name := range []string{"process", "clock", "epoch", "reversed", "timestamp", "call-capture", "malformed-capture", "malformed-json", "unknown-json", "truncated", "short-capture", "oversized", "line-bound", "record-bound", "outlier-bound", "options"} {
		t.Run(name, func(t *testing.T) {
			o, results := probeReportFixture(t)
			switch name {
			case "process":
				results[0].Capture.Identity.Process.Generation++
			case "clock":
				results[0].Call.Clock = "wall"
			case "epoch":
				results[0].Call.Epoch = "another-boot"
			case "reversed":
				results[0].Call.ReturnTimestamp = 99
			case "timestamp":
				results[0].Call.EntryTimestamp++
			case "call-capture":
				results[0].Capture = nil
			case "malformed-capture":
				results[0].Capture.Data[4] = 1
			case "line-bound":
				o.MaxLineBytes = 20
			case "record-bound":
				o.MaxRecords = 1
				o.LastSequence = 1
			case "outlier-bound":
				o.MaxOutliers = 1
				o.Threshold = time.Nanosecond
			case "options":
				o.FirstSequence = 0
			}
			data := probeReportJSON(t, results)
			switch name {
			case "malformed-json":
				data = []byte("{\n")
			case "unknown-json":
				data = []byte("{\"invented\":1}\n")
			case "truncated":
				data = data[:len(data)-1]
			case "short-capture":
				var wire map[string]any
				if err := json.Unmarshal(bytes.Split(data, []byte("\n"))[0], &wire); err != nil {
					t.Fatal(err)
				}
				wire["Capture"].(map[string]any)["Data"] = []int{1, 0, 3, 0}
				data, _ = json.Marshal(wire)
				data = append(data, '\n')
			case "oversized":
				o.MaxBytes = int64(len(data) - 1)
				o.MaxLineBytes = int(o.MaxBytes)
			}
			r, err := cudadiag.ReadProbeReport(context.Background(), bytes.NewReader(data), o)
			if err == nil || r != nil {
				t.Fatalf("accepted invalid input: %+v, %v", r, err)
			}
		})
	}
	o, results := probeReportFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.ReadProbeReport(ctx, bytes.NewReader(probeReportJSON(t, results)), o); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestProbeReportDemangler(t *testing.T) {
	o, results := probeReportFixture(t)
	path := filepath.Join(t.TempDir(), "fixture-cu++filt")
	body := "#!/bin/sh\nif [ \"$1\" = -v ]; then printf 'CPU fixture only 1\\n'; exit 0; fi\nwhile IFS= read -r symbol; do\n case \"$symbol\" in _Z1fv) printf 'f()\\n';; *) printf '%s\\n' \"$symbol\";; esac\ndone\n"
	if err := os.WriteFile(path, []byte(body), 0700); err != nil {
		t.Fatal(err)
	}
	d, err := cudaprobe.NewDemangler(path, o.MaxOutliers)
	if err != nil {
		t.Fatal(err)
	}
	o.Demangler = d
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	r, err := cudadiag.ReadProbeReport(ctx, bytes.NewReader(probeReportJSON(t, results)), o)
	if err != nil {
		t.Fatal(err)
	}
	x := r.Outliers[0].Record
	if x.KernelName != "f()" || x.Labels["kernel.symbol"] != "_Z1fv" || x.Labels["demangler.path"] != path || x.Labels["demangler.version"] != "CPU fixture only 1" || x.Labels["demangler.status"] != "demangled" || x.Labels["duration_status"] != "unavailable" {
		t.Fatalf("demangler not composed: %+v", x)
	}
	// The tool receives only admitted names; a missing binding stays unknown.
	o.Bindings = nil
	r, err = cudadiag.ReadProbeReport(ctx, bytes.NewReader(probeReportJSON(t, results)), o)
	if err != nil || r.Outliers[0].Record.KernelName != "" {
		t.Fatalf("demangled unbound call: %+v, %v", r, err)
	}
}
