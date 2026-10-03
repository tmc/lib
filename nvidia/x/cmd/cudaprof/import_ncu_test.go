package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudancuimport"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"rsc.io/script"
	"rsc.io/script/scripttest"
)

func ncuWriteFixture(t *testing.T, name string, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, b, 0600); err != nil {
		t.Fatal(err)
	}
	return p
}
func ncuEnvelopeFixture(t *testing.T) cudancuimport.Envelope {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "cudancuimport", "testdata", "transpose-details.csv"))
	if err != nil {
		t.Fatal(err)
	}
	return cudancuimport.Envelope{SchemaVersion: 1, ToolVersion: cudancuimport.ToolVersion, Format: cudancuimport.Format, SourcePath: "NVIDIA bundled sharedBankConflicts/transposeNoBankConflicts.ncu-rep", ReportDigest: "544b42110730bd2adf2325eeea44a69be6455767443fe1d7a1572bbd7183bb29", CSVDigest: ncuDigest(b), CSV: string(b)}
}
func ncuDigest(b []byte) string { return fmtHex(sha256.Sum256(b)) }
func fmtHex(h [32]byte) string  { return hex.EncodeToString(h[:]) }
func ncuEnvelopeFile(t *testing.T, e cudancuimport.Envelope) string {
	t.Helper()
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	return ncuWriteFixture(t, "envelope.json", b)
}
func TestImportNCUCommandKnownOffline(t *testing.T) {
	path := ncuEnvelopeFile(t, ncuEnvelopeFixture(t))
	var b bytes.Buffer
	if err := runImportNCUCommand([]string{"-json", "-limit", "2", path}, &b); err != nil {
		t.Fatal(err)
	}
	var summary ncuSummary
	if err := json.Unmarshal(b.Bytes(), &summary); err != nil {
		t.Fatal(err)
	}
	if summary.TotalRows != 65 || summary.Measured != 0 || summary.Unread != 58 || summary.Rules != 7 || !summary.Truncated || len(summary.Rows) != 2 {
		t.Fatalf("bad summary %+v", summary)
	}
	if summary.Rows[0].Raw[13] != "hz" || summary.Rows[0].Metric.RawValue != "7549191685.91" || summary.Rows[0].Metric.Value.Record().Number != nil {
		t.Fatal("unsupported raw metric changed")
	}
	b.Reset()
	if err := runImportNCUCommand([]string{"-limit", "1", path}, &b); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), "external measured 0; unread 58") || !strings.Contains(b.String(), "summary truncated") {
		t.Fatal(b.String())
	}
	if command, ok := lookupCommand("import-ncu"); !ok || command.usage != importNCUUsage {
		t.Fatal("command not registered")
	}
}

func ncuJoinCommandFixture(t *testing.T) (cudancuimport.Envelope, string, string) {
	t.Helper()
	read := func(name string) []byte {
		t.Helper()
		b, err := os.ReadFile(filepath.Join("..", "..", "cudacapture", "testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	raw, err := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if err != nil {
		t.Fatal(err)
	}
	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, "fill", schema, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	w := cudacache.WorkloadIdentity{ProgramID: "synthetic offline CLI launch", ShapeDims: []int{64}, NumericalMode: "fp32"}
	key := cudacache.CompilerCacheKey(a.Digest, cudacache.CompilerIdentity{TargetArch: a.Target, OutputKind: nvrtcx.OutputCUBIN})
	id, err := cudaprof.BindEvidence(a, w, key)
	if err != nil {
		t.Fatal(err)
	}
	code := cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"}
	c := &cudacapture.Capture{Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}}}, Code: code, Modules: []cudacapture.Module{{ID: "module", Generation: 1, Code: code}}, Files: map[string][]byte{"kernels.cu": read("kernels.cu"), "kernels.ptx": read("kernels.ptx"), "kernels.cubin": raw, "kernels.sass": read("kernels.sass")}}
	var packet bytes.Buffer
	if err := cudacapture.WriteLimits(&packet, c, cudacapture.Limits{}); err != nil {
		t.Fatal(err)
	}
	artifact := ncuArtifact(*a)
	receipt := ncuJoinReceipt{SchemaVersion: 1, Artifact: &artifact, Workload: w, CompilerKey: key}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	e := ncuEnvelopeFixture(t)
	e.SourcePath = "synthetic offline CLI launch"
	e.ReportDigest = ncuDigest([]byte(e.SourcePath))
	rows, err := csv.NewReader(strings.NewReader(e.CSV)).Read()
	if err != nil {
		t.Fatal(err)
	}
	var csvOut bytes.Buffer
	writer := csv.NewWriter(&csvOut)
	if err := writer.Write(rows); err != nil {
		t.Fatal(err)
	}
	if err := writer.Write([]string{"0", "1", "synthetic", "host", "fill", "1", "7", "(64, 1, 1)", "(1, 1, 1)", "0", "9.0", "Timing", "gpu__time_duration.sum", "ns", "42", ""}); err != nil {
		t.Fatal(err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	e.CSV = csvOut.String()
	e.CSVDigest = ncuDigest(csvOut.Bytes())
	e.Bindings = []cudancuimport.Binding{{Launch: cudancuimport.LaunchKey{ID: "0", Process: "1", Context: "1", Stream: "7"}, Identity: id}}
	return e, ncuWriteFixture(t, "capture.pprof", packet.Bytes()), ncuWriteFixture(t, "receipt.json", b)
}
func TestImportNCUCommandExactJoin(t *testing.T) {
	e, capture, receipt := ncuJoinCommandFixture(t)
	var b bytes.Buffer
	args := []string{"-json", "-capture", capture, "-join-receipt", receipt, ncuEnvelopeFile(t, e)}
	if err := runImportNCUCommand(args, &b); err != nil {
		t.Fatal(err)
	}
	var s ncuSummary
	if err := json.Unmarshal(b.Bytes(), &s); err != nil {
		t.Fatal(err)
	}
	if s.Measured != 1 || s.Unread != 0 || s.Rows[0].ModuleID != "module" || s.Rows[0].Metric.Value.Record().Basis != "external:ncu" {
		t.Fatalf("bad join %+v", s)
	}
	e.Bindings = nil
	b.Reset()
	args[len(args)-1] = ncuEnvelopeFile(t, e)
	if err := runImportNCUCommand(args, &b); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b.Bytes(), &s); err != nil {
		t.Fatal(err)
	}
	if s.Measured != 0 || s.Unread != 1 {
		t.Fatal("capture supplied missing external identity")
	}
}
func TestImportNCUCommandRefusals(t *testing.T) {
	e := ncuEnvelopeFixture(t)
	path := ncuEnvelopeFile(t, e)
	for _, args := range [][]string{{"-max-bytes", "1", path}, {"-limit", "0", path}, {"-limit", "1025", path}, {"-capture", path, path}, {"-join-receipt", path, path}, {"-unknown", path}, {path, path}, {}} {
		var b bytes.Buffer
		if err := runImportNCUCommand(args, &b); err == nil || b.Len() != 0 {
			t.Fatalf("args=%q err=%v", args, err)
		}
	}
	var b bytes.Buffer
	if err := runImportNCUCommand([]string{"-h"}, &b); err != nil || !strings.Contains(b.String(), "import-ncu") {
		t.Fatal(err)
	}
	_, capture, receipt := ncuJoinCommandFixture(t)
	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{
		bytes.Replace(data, []byte(`"schema_version":1`), []byte(`"schema_version":1,"schema_version":1`), 1),
		bytes.Replace(data, []byte(`"kind":2`), []byte(`"kind":2,"unknown":1`), 1),
		bytes.Replace(data, []byte(`"compiler_key"`), []byte(`"Compiler_Key"`), 1),
		bytes.Replace(data, []byte(`"schema_version"`), []byte(`"ſchema_version"`), 1),
		bytes.Replace(data, []byte(`"schema_version"`), []byte(`"\u017fchema_version"`), 1),
		bytes.Replace(data, []byte(`"schema_version"`), []byte(`"\u0053chema_version"`), 1),
		bytes.Replace(data, []byte(`"schema"`), []byte(`"ſchema"`), 1),
		bytes.Replace(data, []byte(`"schema"`), []byte(`"\u017fchema"`), 1),
		append(append([]byte(nil), data...), []byte(` {}`)...),
	} {
		b.Reset()
		p := ncuWriteFixture(t, "bad.json", bad)
		if err := runImportNCUCommand([]string{"-capture", capture, "-join-receipt", p, path}, &b); err == nil {
			t.Fatal("invalid receipt accepted")
		}
	}
}

type ncuShortWriter struct{}

func (ncuShortWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }
func TestImportNCUCommandShortWrite(t *testing.T) {
	path := ncuEnvelopeFile(t, ncuEnvelopeFixture(t))
	if err := runImportNCUCommand([]string{path}, ncuShortWriter{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
}

func TestImportNCUCommandRegularFile(t *testing.T) {
	path := ncuEnvelopeFile(t, ncuEnvelopeFixture(t))
	link := filepath.Join(t.TempDir(), "link.json")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{link, t.TempDir()} {
		var out bytes.Buffer
		if err := runImportNCUCommand([]string{p}, &out); err == nil || out.Len() != 0 {
			t.Fatalf("nonregular file admitted: %q", p)
		}
	}
}

// This script runs the public CLI offline. It uses the real report export and
// never invokes ncu, CUDA, or a collector.
func TestImportNCUScript(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	// scripttest schedules parallel subtests; keep their context live until
	// their completion rather than canceling when this parent returns.
	t.Cleanup(cancel)
	dir := t.TempDir()
	bin := filepath.Join(dir, "cudaprof")
	build := exec.CommandContext(ctx, "go", "build", "-o", bin, ".")
	build.Env = append(os.Environ(), "GOWORK=off")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build CLI: %v\n%s", err, out)
	}
	fixture := ncuEnvelopeFile(t, ncuEnvelopeFixture(t))
	commands := `exec cudaprof import-ncu -limit 1 $ENVELOPE
stdout 'external measured 0; unread 58; unsupported rules 7'
stdout 'summary truncated to 1 rows'
exec cudaprof import-ncu -json -limit 2 $ENVELOPE
stdout '"total_rows":65'
stdout '"external_measured":0'
! exec cudaprof import-ncu -max-bytes 1 $ENVELOPE
stderr 'not a bounded regular file'
exec cudaprof help import-ncu
stdout 'usage: cudaprof import-ncu'
`
	path := filepath.Join(dir, "offline.txt")
	if err := os.WriteFile(path, []byte(commands), 0600); err != nil {
		t.Fatal(err)
	}
	engine := &script.Engine{Cmds: scripttest.DefaultCmds(), Conds: scripttest.DefaultConds()}
	env := []string{"PATH=" + dir + string(os.PathListSeparator) + os.Getenv("PATH"), "HOME=" + os.Getenv("HOME"), "GOWORK=off", "ENVELOPE=" + fixture}
	scripttest.Test(t, ctx, engine, env, path)
}
