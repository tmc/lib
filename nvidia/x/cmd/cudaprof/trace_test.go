package main

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTracePacketFraming(t *testing.T) {
	for _, tt := range []struct {
		name string
		data []byte
		want bool
	}{
		{"packet", []byte{10, 2, 8, 1}, true},
		{"empty", nil, false},
		{"truncated", []byte{10, 3, 8, 1}, false},
		{"wrong envelope", []byte{18, 2, 8, 1}, false},
		{"trailing", []byte{10, 2, 8, 1, 0}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := validTracePackets(tt.data); got != tt.want {
				t.Fatalf("got %v want %v", got, tt.want)
			}
		})
	}
}

func TestTraceCheckExternalEvidence(t *testing.T) {
	if os.PathSeparator != '/' {
		t.Skip("POSIX executable fixture")
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "arbitrary.extension")
	if err := os.WriteFile(input, []byte{10, 2, 8, 1}, 0600); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, body string
		extra      []string
		want       string
		code       int
	}{
		{"import healthy", "printf '" + "name,severity,value\\n" + "'", nil, "passed", 0},
		{"no collector", "printf 'name,severity,value\\n'", []string{"-require", "collector"}, "passed", 2},
		{"positive error", "printf 'name,severity,value\\nbad,error,1\\n'", nil, "failed", 2},
		{"unsupported version", "", nil, "unavailable", 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			processor := filepath.Join(dir, "processor-"+tt.name)
			version := "Perfetto v58.2-test"
			if tt.name == "unsupported version" {
				version = "Perfetto v1"
			}
			script := "#!/bin/sh\nif [ \"$1\" = --version ]; then echo '" + version + "'; exit; fi\n" + "if grep -q \"AS section\" \"$3\"; then printf 'section,identity,value\\n'; exit; fi\n" + tt.body + "\n"
			if err := os.WriteFile(processor, []byte(script), 0700); err != nil {
				t.Fatal(err)
			}
			args := append([]string{"-json", "-perfetto", processor}, tt.extra...)
			args = append(args, input)
			var out bytes.Buffer
			err := runTraceCheck(args, &out)
			var exit *exitCodeError
			if tt.code == 0 && err != nil {
				t.Fatal(err)
			}
			if tt.code != 0 && (!errors.As(err, &exit) || exit.Code != tt.code) {
				t.Fatalf("error %v want exit %d", err, tt.code)
			}
			var r traceCheckReport
			if err := json.Unmarshal(out.Bytes(), &r); err != nil {
				t.Fatal(err)
			}
			if r.Kind != "pftrace" || r.Checks["import"].Disposition != tt.want || r.Checks["collector"].Disposition != "unavailable" {
				t.Fatalf("receipt %+v", r)
			}
			if r.Qualified != (tt.code == 0) {
				t.Fatal("exit and receipt disagree")
			}
		})
	}
}

func TestTraceCheckInvalidRequests(t *testing.T) {
	for _, args := range [][]string{{"-require", "invented", "missing"}, {"-expect-markers", "1", "missing"}, {"-domain", "invalid", "missing"}} {
		if err := runTraceCheck(args, new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestTraceBoundedOutput(t *testing.T) {
	out := traceBoundedOutput{limit: 3}
	if _, err := out.Write([]byte("abc")); err != nil {
		t.Fatal(err)
	}
	if _, err := out.Write([]byte("d")); err == nil {
		t.Fatal("output limit ignored")
	}
}

func TestTraceCheckNativeCollector(t *testing.T) {
	for _, tt := range []struct {
		name  string
		alter func(map[string]any)
		code  int
	}{
		{"complete", func(map[string]any) {}, 0},
		{"partial", func(s map[string]any) { s["run_completeness"] = "partial" }, 2},
		{"crashed", func(s map[string]any) { s["target_exit_code"] = 7 }, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			status := completeStatus()
			tt.alter(status)
			cap := compareCapture(iterationRecs(5), status)
			file := filepath.Join(t.TempDir(), "capture.unrelated")
			f, err := os.Create(file)
			if err != nil {
				t.Fatal(err)
			}
			if err := cudacapture.Write(f, cap); err != nil {
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			var out bytes.Buffer
			err = runTraceCheck([]string{"-json", file}, &out)
			var exit *exitCodeError
			if tt.code == 0 && err != nil {
				t.Fatal(err)
			}
			if tt.code != 0 && (!errors.As(err, &exit) || exit.Code != tt.code) {
				t.Fatalf("error %v", err)
			}
			var report traceCheckReport
			if err := json.Unmarshal(out.Bytes(), &report); err != nil {
				t.Fatal(err)
			}
			if report.Kind != "capture" || report.Qualified != (tt.code == 0) {
				t.Fatalf("report %+v", report)
			}
		})
	}
}

func TestTraceCheckBackendExecutionExit(t *testing.T) {
	if os.PathSeparator != '/' {
		t.Skip("POSIX fixture")
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "trace")
	if err := os.WriteFile(input, []byte{10, 2, 8, 1}, 0600); err != nil {
		t.Fatal(err)
	}
	processor := filepath.Join(dir, "processor")
	if err := os.WriteFile(processor, []byte("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'Perfetto v58.2-test'; exit; fi\nexit 9\n"), 0700); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	err := runTraceCheck([]string{"-json", "-perfetto", processor, input}, &out)
	var code *exitCodeError
	if err == nil || errors.As(err, &code) {
		t.Fatalf("backend execution should exit 1, got %v", err)
	}
	var receipt traceCheckReport
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Qualified {
		t.Fatal("backend failure reported qualification success")
	}
}

func TestTraceCheckResourceOverrides(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "trace")
	if err := os.WriteFile(input, []byte{10, 2, 8, 1}, 0600); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"-input-bytes", "3", input}, {"-max-records", "0", input}, {"-backend-timeout", "0s", input}} {
		if err := runTraceCheck(args, new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestTraceProcessorGenericQuerySchema(t *testing.T) {
	if os.PathSeparator != '/' {
		t.Skip("POSIX fixture")
	}
	processor := filepath.Join(t.TempDir(), "processor")
	if err := os.WriteFile(processor, []byte("#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'Perfetto v58.2-test'; exit; fi\nprintf 'count\\n2\\n'\n"), 0700); err != nil {
		t.Fatal(err)
	}
	result, err := queryTraceProcessor([]byte{10, 2, 8, 1}, processor, "SELECT COUNT(*) AS count FROM slice")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 2 || result.Rows[1][0] != "2" {
		t.Fatalf("rows %v", result.Rows)
	}
}

func traceArchiveFixture(t *testing.T, mutate func(map[string][]byte)) []byte {
	t.Helper()
	hash := strings.Repeat("a", 64)
	a := alignmentReceipt{Version: 1, TimeScale: 1, Arms: [2]alignmentArm{{Trace: "base.pftrace", SHA256: hash}, {Trace: "target.pftrace", SHA256: hash}}}
	alignment, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	members := map[string][]byte{"base.pftrace": {10, 2, 8, 1}, "target.pftrace": {10, 2, 8, 1}, ".alignment.json": alignment, "perfetto_manifest.json": []byte(`{"perfetto_manifest":{"version":1,"trace_time":{"clock":"BOOTTIME","file":"base.pftrace"},"files":[{"path":"base.pftrace"},{"path":"target.pftrace","clocks":{"clock":"BOOTTIME","offset_ns":0,"sync_to":{"clock":"BOOTTIME","file":"base.pftrace"}}}]}}`)}
	index := tracecontract.Bindings{Version: 1, Schema: "cudaprof.bindings.v1"}
	for name, data := range members {
		kind := "metadata"
		source := ""
		if strings.HasSuffix(name, ".pftrace") {
			kind = "trace"
			source = hash
		}
		member, err := tracecontract.Digest(name, kind, source, data)
		if err != nil {
			t.Fatal(err)
		}
		index.Members = append(index.Members, member)
	}
	indexData, err := index.JSON()
	if err != nil {
		t.Fatal(err)
	}
	members[".cudaprof-bindings.json"] = indexData
	if mutate != nil {
		mutate(members)
	}
	var buf bytes.Buffer
	writer := tar.NewWriter(&buf)
	for name, data := range members {
		if err := writer.WriteHeader(&tar.Header{Name: name, Mode: 0600, Size: int64(len(data))}); err != nil {
			t.Fatal(err)
		}
		if _, err := writer.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestTraceArchiveBindings(t *testing.T) {
	for _, tt := range []struct {
		name    string
		mutate  func(map[string][]byte)
		wantErr bool
	}{
		{"complete", nil, false},
		{"hash mismatch", func(m map[string][]byte) { m["base.pftrace"] = []byte{10, 2, 8, 2} }, true},
		{"unbound member", func(m map[string][]byte) { m["extra"] = []byte("added") }, true},
		{"missing member", func(m map[string][]byte) { delete(m, "target.pftrace") }, true},
		{"unsafe path", func(m map[string][]byte) { m["../extra"] = []byte("added") }, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := inspectTraceArchive(traceArchiveFixture(t, tt.mutate))
			if (err != nil) != tt.wantErr {
				t.Fatalf("error %v", err)
			}
		})
	}
}
