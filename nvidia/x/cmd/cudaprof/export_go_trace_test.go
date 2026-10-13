package main

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

var goTraceLeasedFiles = []string{"export_stack_samples.go", "export.go", "export_compare.go", "trace_stacks.go", "trace_query.go", "trace.go"}

// Q6 (cmd part): the only root contexts in the leased command files are the
// three command-wrapper signal contexts.
func TestExportGoTraceSourceGate(t *testing.T) {
	var lines []string
	for _, name := range goTraceLeasedFiles {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(string(data), "\n") {
			if strings.Contains(line, "context.Background()") || strings.Contains(line, "context.TODO()") {
				lines = append(lines, name+": "+strings.TrimSpace(line))
			}
		}
	}
	notify := 0
	for _, line := range lines {
		if strings.Contains(line, "signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)") {
			notify++
		}
	}
	if len(lines) != 3 || notify != 3 {
		t.Errorf("background contexts:\n%s", strings.Join(lines, "\n"))
	}
}

const goTraceExportReceiptFields = `"version":1,"pid":12,"process":"p","session":"s","generation":7,"start_wall_utc":"2026-10-12T00:00:00Z","end_wall_utc":"2026-10-12T00:00:01Z","gpu_clock_alignment":"unavailable","cpu_sample_timestamps":"not_retained","goos":"darwin"`

var goTraceExportProducer = languagestack.GoProducer{GoVersion: "go1.27.2", GOOS: "darwin", GOARCH: "arm64"}

// Q3 and Q4: the host receipt is read strictly before any decode.
func TestExportGoTraceReceipt(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	full := goTraceExportReceiptFields
	for _, tt := range []struct {
		name, receipt, want string
	}{
		{"base shape without goarch", `{` + full + `,"go_version":"go1.27.2"}`, "producer receipt incomplete: goarch"},
		{"empty go_version", `{` + full + `,"goarch":"arm64","go_version":""}`, "producer receipt incomplete: go_version"},
		{"version 2", `{` + strings.Replace(full, `"version":1`, `"version":2`, 1) + `,"goarch":"arm64","go_version":"go1.27.2"}`, "producer receipt incomplete: version"},
		{"duplicate goos", `{` + full + `,"goarch":"arm64","go_version":"go1.27.2","goos":"linux"}`, "ambiguous host receipt: goos"},
		{"case variant", `{` + full + `,"goarch":"arm64","GoVersion":"go1.27.2"}`, "ambiguous host receipt: GoVersion"},
		{"unknown key", `{` + full + `,"goarch":"arm64","go_version":"go1.27.2","extra":1}`, "ambiguous host receipt: extra"},
		{"complete", `{` + full + `,"goarch":"arm64","go_version":"go1.27.2"}`, "runtime trace attachment missing"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{Files: map[string][]byte{"stacks-go-1-host.json": []byte(tt.receipt)}}
			groups, err := readLanguageSamples(context.Background(), cap, goTraceOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if len(groups) != 1 || groups[0].Unavailable != tt.want {
				t.Fatalf("groups = %+v, want %q", groups, tt.want)
			}
			if (groups[0].Producer != nil) != (tt.name == "complete") {
				t.Fatalf("producer = %+v", groups[0].Producer)
			}
		})
	}
	for _, bad := range []string{`{"version":1`, `{"version":1} {}`, `[]`, `null`, `{"pid":"x"}`} {
		cap := &cudacapture.Capture{Files: map[string][]byte{"stacks-go-1-host.json": []byte(bad)}}
		if _, err := readLanguageSamples(context.Background(), cap, goTraceOptions{}); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
}

// goTraceExportCapture writes a capture holding a complete host receipt and a
// header-26 trace attachment. The trace is not decodable; no decoder is
// admitted, so none may run.
func goTraceExportCapture(t *testing.T) string {
	t.Helper()
	return goTraceExportCaptureReceipt(t, `{`+goTraceExportReceiptFields+`,"goarch":"arm64","go_version":"go1.27.2"}`)
}

func goTraceExportCaptureReceipt(t *testing.T, receipt string) string {
	t.Helper()
	cap := compareCapture(iterationRecs(5), completeStatus())
	cap.Files["stacks-go-1-host.json"] = []byte(receipt)
	cap.Files["stacks-go-1.trace"] = []byte("go 1.26 trace\x00\x00\x00\x01\x02\x03\x04")
	path := filepath.Join(t.TempDir(), "go-trace.pprof")
	writeTestCaptureFile(t, path, cap)
	return path
}

type goTraceQueryResult struct {
	Options  map[string]any `json:"options"`
	Evidence struct {
		Groups []languageSamples `json:"groups"`
	} `json:"evidence"`
}

func goTraceQuery(t *testing.T, ctx context.Context, path string, options ...string) languageSamples {
	t.Helper()
	var out bytes.Buffer
	args := append(append([]string(nil), options...), "-json", "host-stacks", path)
	if err := runTraceQueryContext(ctx, args, &out); err != nil {
		t.Fatalf("query: %v\n%s", err, out.Bytes())
	}
	var r goTraceQueryResult
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("decode query: %v\n%s", err, out.Bytes())
	}
	var found []languageSamples
	for _, g := range r.Evidence.Groups {
		if g.Receipt == "stacks-go-1-host.json" {
			found = append(found, g)
		}
	}
	if len(found) != 1 {
		t.Fatalf("groups = %+v", r.Evidence.Groups)
	}
	return found[0]
}

// goTraceArgs places the capture path last, as every entry point expects.
func goTraceArgs(path string, options ...string) []string {
	return append(options, path)
}

// Q5 (structural part): the caller's context and options reach the decoder
// along every export, compare and query path. Export and compare draw
// nothing for an unavailable Go group and only report its detail, so this
// gate also checks the exact arguments of each call on the chain;
// TestExportGoTraceUnavailable checks the reported detail.
func TestExportGoTracePropagation(t *testing.T) {
	fset := token.NewFileSet()
	calls := make(map[string][]string)
	for _, name := range goTraceLeasedFiles {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				var callee string
				switch f := call.Fun.(type) {
				case *ast.Ident:
					callee = f.Name
				case *ast.SelectorExpr:
					callee = f.Sel.Name
				default:
					return true
				}
				var args []string
				for _, a := range call.Args {
					var b bytes.Buffer
					printer.Fprint(&b, fset, a)
					args = append(args, b.String())
				}
				key := fn.Name.Name + "->" + callee
				calls[key] = append(calls[key], strings.Join(args, ", "))
				return true
			})
		}
	}
	for key, want := range map[string]string{
		"runExportCommand->runExport":                   "ctx, args",
		"runExport->readLanguageSamples":                "ctx, cap, opts.goTrace",
		"runExport->appendLanguageGroups":               "records, groups",
		"runExportCompareCommand->runExportCompare":     "ctx, args",
		"runExportCompare->loadAlignmentArm":            "ctx, opts, i",
		"loadAlignmentArm->readLanguageSamples":         "ctx, cap, opts.goTrace",
		"loadAlignmentArm->appendLanguageGroups":        "arm.records, groups",
		"appendLanguageSamples->appendLanguageGroups":   "records, groups",
		"runTraceQuery->runTraceQueryContext":           "ctx, args, out",
		"runTraceQueryContext->nativeLanguageStackView": "ctx, input.Capture, definition.Name, goTrace",
		"nativeLanguageStackView->readLanguageSamples":  "ctx, cap, goTrace",
		"appendLanguageSamples->readLanguageSamples":    "ctx, cap, goTrace",
		"analyzeRetainedGoTrace->Analyze":               "ctx, data, producer, identity, languagestack.Limits{}",
		"readLanguageSamples->decoderFor":               "",
	} {
		if got := calls[key]; len(got) != 1 || got[0] != want {
			t.Errorf("%s: calls %q, want [%q]", key, got, want)
		}
	}
	for _, key := range []string{"runExport->appendLanguageSamples", "loadAlignmentArm->appendLanguageSamples"} {
		if got := calls[key]; len(got) != 0 {
			t.Errorf("%s: calls %q, want none", key, got)
		}
	}
	got := calls["readLanguageSamples->analyzeRetainedGoTrace"]
	if len(got) != 1 || !strings.HasPrefix(got[0], "ctx, dec, trace, producer, ") {
		t.Errorf("readLanguageSamples->analyzeRetainedGoTrace: calls %q", got)
	}
	for _, wrapper := range []string{"runExportCommand", "runExportCompareCommand", "runTraceQuery"} {
		if got := calls[wrapper+"->NotifyContext"]; len(got) != 1 || got[0] != "context.Background(), os.Interrupt, syscall.SIGTERM" {
			t.Errorf("%s: NotifyContext calls %q", wrapper, got)
		}
	}
}

// Q1 and Q1b: an unselected decoder leaves the Go group unavailable through
// every entry point, with the stack sentinel's exact text and no child.
// The query output shows the group; every other entry point is shown to
// reach the strict host-receipt reader at run time, and
// TestExportGoTracePropagation fixes what it passes there.
func TestExportGoTraceUnselected(t *testing.T) {
	t.Setenv("PATH", t.TempDir()) // a bare "go" fallback cannot be found
	path := goTraceExportCapture(t)
	dir := t.TempDir()
	for name, run := range map[string]func() error{
		"alias": func() error {
			return runTraceAlias("export", []string{"export", "-o", filepath.Join(dir, "a.pftrace"), path})
		},
		"trace export": func() error { return runTraceCommand([]string{"export", "-o", filepath.Join(dir, "b.pftrace"), path}) },
		"export ctx": func() error {
			return runExport(context.Background(), []string{"-o", filepath.Join(dir, "c.pftrace"), path})
		},
		"compare arm":   func() error { return goTraceCompareArm(context.Background(), path) },
		"query wrapper": func() error { return runTraceQuery([]string{"-json", "host-stacks", path}, io.Discard) },
	} {
		if err := goTraceQuiet(run); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	broken := goTraceExportCaptureReceipt(t, `{"version":1,`)
	for name, run := range map[string]func() error{
		"alias": func() error {
			return runTraceAlias("export", []string{"export", "-o", filepath.Join(dir, "d.pftrace"), broken})
		},
		"trace export": func() error {
			return runTraceCommand([]string{"export", "-o", filepath.Join(dir, "e.pftrace"), broken})
		},
		"export ctx": func() error {
			return runExport(context.Background(), []string{"-o", filepath.Join(dir, "f.pftrace"), broken})
		},
		"compare arm": func() error { return goTraceCompareArm(context.Background(), broken) },
		"query":       func() error { return runTraceQuery([]string{"-json", "host-stacks", broken}, io.Discard) },
	} {
		if err := goTraceQuiet(run); err == nil || !strings.Contains(err.Error(), "read host receipt stacks-go-1-host.json") {
			t.Errorf("%s did not reach the host receipt reader: %v", name, err)
		}
	}
	g := goTraceQuery(t, context.Background(), path)
	if g.Unavailable != languagestack.ErrGoTraceDecoderUnselected.Error() || g.Unavailable != "go trace decoder not selected" {
		t.Errorf("Unavailable = %q", g.Unavailable)
	}
	if g.Producer == nil || *g.Producer != goTraceExportProducer || g.GoTrace != nil {
		t.Errorf("group = %+v", g)
	}
}

func goTraceCompareArm(ctx context.Context, path string, options ...string) error {
	opts, err := parseExportCompareArgs(append(append([]string(nil), options...), "-iteration-marker", "Embed", "-iteration-count", "2", path, path), io.Discard)
	if err != nil {
		return err
	}
	_, err = loadAlignmentArm(ctx, opts, 0)
	return err
}

// goTraceQuiet runs fn with stdout discarded and returns its error.
func goTraceQuiet(fn func() error) error {
	orig := os.Stdout
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	os.Stdout = devnull
	defer func() { os.Stdout = orig; devnull.Close() }()
	return fn()
}

// Q2: pairing, repeats and path form are refused while parsing, before the
// capture is read.
func TestExportGoTraceOptionParse(t *testing.T) {
	missing := "/nonexistent-c6d1/capture.pprof"
	for _, tt := range []struct {
		name    string
		options []string
		want    string
	}{
		{"decoder alone", []string{"-go-trace-decoder", "/nonexistent-c6d1/trace"}, "selected together"},
		{"scratch alone", []string{"-go-trace-scratch=/nonexistent-c6d1/scratch"}, "selected together"},
		{"repeated", []string{"-go-trace-decoder=/a/trace", "-go-trace-decoder=/b/trace", "-go-trace-scratch=/s"}, "repeated -go-trace-decoder"},
		{"relative", []string{"-go-trace-decoder=trace", "-go-trace-scratch=/s"}, "absolute, clean"},
		{"unclean", []string{"-go-trace-decoder=/a/../trace", "-go-trace-scratch=/s"}, "absolute, clean"},
		{"unclean scratch", []string{"-go-trace-decoder=/a/trace", "-go-trace-scratch=/s/"}, "absolute, clean"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := parseExportArgs(goTraceArgs(missing, tt.options...)); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("export: %v", err)
			}
			if _, err := parseExportCompareArgs(append(append([]string(nil), tt.options...), "-iteration-marker", "x", missing, missing), io.Discard); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("export-compare: %v", err)
			}
			if err := runTraceQueryContext(context.Background(), append(append([]string(nil), tt.options...), "host-stacks", missing), io.Discard); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("trace query: %v", err)
			}
		})
	}
}

// Q2b: valid options are accepted without touching the file system; the
// missing capture is then reported and the scratch is never created.
func TestExportGoTraceOptionNoSideEffects(t *testing.T) {
	scratch := filepath.Join(t.TempDir(), "absent-scratch")
	options := []string{"-go-trace-decoder=/nonexistent-c6d1/trace", "-go-trace-scratch=" + scratch}
	missing := filepath.Join(t.TempDir(), "missing.pprof")
	opts, err := parseExportArgs(goTraceArgs(missing, options...))
	if err != nil || opts.goTrace != (goTraceOptions{"/nonexistent-c6d1/trace", scratch}) {
		t.Fatalf("export parse: %+v, %v", opts.goTrace, err)
	}
	copts, err := parseExportCompareArgs(append(append([]string(nil), options...), "-iteration-marker", "x", missing, missing), io.Discard)
	if err != nil || copts.goTrace != opts.goTrace {
		t.Fatalf("export-compare parse: %+v, %v", copts.goTrace, err)
	}
	if err := runExport(context.Background(), goTraceArgs(missing, options...)); err == nil || !strings.Contains(err.Error(), "read capture") {
		t.Errorf("export: %v", err)
	}
	if err := runExportCompare(context.Background(), append(append([]string(nil), options...), "-iteration-marker", "x", missing, missing)); err == nil {
		t.Error("export-compare accepted a missing capture")
	}
	if err := runTraceQueryContext(context.Background(), append(append([]string(nil), options...), "host-stacks", missing), io.Discard); err == nil {
		t.Error("trace query accepted a missing capture")
	}
	if _, err := os.Lstat(scratch); !os.IsNotExist(err) {
		t.Errorf("scratch created: %v", err)
	}
}

// Q5: a selected decoder receives the caller's context; a cancelled context
// and an unadmitted tuple both stay unavailable without touching scratch.
func TestExportGoTraceContext(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	path := goTraceExportCapture(t)
	scratch := t.TempDir()
	if err := os.Chmod(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	options := []string{"-go-trace-decoder=/nonexistent-c6d1/trace", "-go-trace-scratch=" + scratch}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tt := range []struct {
		name string
		ctx  context.Context
		want string
	}{
		{"cancelled", cancelled, "context canceled"},
		{"unadmitted", context.Background(), languagestack.ErrGoTraceUnadmitted.Error()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			g := goTraceQuery(t, tt.ctx, path, options...)
			if !strings.Contains(g.Unavailable, tt.want) || g.GoTrace != nil {
				t.Errorf("query group = %+v", g)
			}
			cap, _, err := readAlignmentCapture(path, cudacapture.Limits{})
			if err != nil {
				t.Fatal(err)
			}
			groups, err := readLanguageSamples(tt.ctx, cap, goTraceOptions{"/nonexistent-c6d1/trace", scratch})
			if err != nil {
				t.Fatal(err)
			}
			for _, g := range groups {
				if g.Receipt == "stacks-go-1-host.json" && !strings.Contains(g.Unavailable, tt.want) {
					t.Errorf("group = %+v", g)
				}
			}
			dir := t.TempDir()
			if err := goTraceQuiet(func() error {
				return runExport(tt.ctx, goTraceArgs(path, append([]string{"-o", filepath.Join(dir, "x.pftrace")}, options...)...))
			}); err != nil {
				t.Errorf("export: %v", err)
			}
			if err := goTraceCompareArm(tt.ctx, path, options...); err != nil {
				t.Errorf("compare arm: %v", err)
			}
			entries, err := os.ReadDir(scratch)
			if err != nil || len(entries) != 0 {
				t.Errorf("scratch touched: %v %v", entries, err)
			}
		})
	}
	var out bytes.Buffer
	if err := runTraceQueryContext(context.Background(), append(append([]string(nil), options...), "-json", "host-stacks", path), &out); err != nil {
		t.Fatal(err)
	}
	var r goTraceQueryResult
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Options["go_trace_decoder"] != "/nonexistent-c6d1/trace" || r.Options["go_trace_scratch"] != scratch {
		t.Errorf("options = %v", r.Options)
	}
}

// Any Analyze error leaves the group unavailable, including a cleanup-only
// error returned with a decoded trace. No admitted decoder exists, so no
// zero-child run reaches the decoded-with-error case; this gate fixes the
// branch in readLanguageSamples instead.
func TestExportGoTraceAnalyzeErrorUnavailable(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "export_stack_samples.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	text := func(n ast.Node) string {
		var b bytes.Buffer
		printer.Fprint(&b, fset, n)
		return b.String()
	}
	var found []string
	ast.Inspect(f, func(n ast.Node) bool {
		fn, ok := n.(*ast.FuncDecl)
		if ok && fn.Name.Name != "readLanguageSamples" {
			return false
		}
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			assign, ok := stmt.(*ast.AssignStmt)
			if !ok || len(assign.Rhs) != 1 || !strings.HasPrefix(text(assign.Rhs[0]), "analyzeRetainedGoTrace(") {
				continue
			}
			next := "none"
			if i+1 < len(block.List) {
				if s, ok := block.List[i+1].(*ast.IfStmt); ok && s.Init == nil && len(s.Body.List) > 0 {
					next = text(s.Cond) + " -> " + text(s.Body.List[0])
				}
			}
			found = append(found, text(assign.Lhs[0])+", "+text(assign.Lhs[1])+": "+next)
		}
		return true
	})
	want := "decoded, err: err != nil -> group.Unavailable = err.Error()"
	if len(found) != 1 || found[0] != want {
		t.Errorf("analyze error handling = %q, want [%q]", found, want)
	}
}

// goTraceUnavailableLines returns the lines of out reporting an unavailable
// Go trace group.
func goTraceUnavailableLines(out string) []string {
	var lines []string
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "go trace ") && strings.Contains(line, ": unavailable: ") {
			lines = append(lines, line)
		}
	}
	return lines
}

// goTraceReceiptArms returns the arms of an alignment receipt with the
// capture identity and go_trace_unavailable removed, for comparison with a
// capture holding no Go groups.
func goTraceReceiptArms(t *testing.T, data []byte) string {
	t.Helper()
	var raw struct {
		Arms []map[string]any `json:"arms"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	for _, arm := range raw.Arms {
		delete(arm, "capture_path")
		delete(arm, "capture_sha256")
		delete(arm, "go_trace_unavailable")
		delete(arm, "go_trace_unavailable_omitted")
	}
	v, err := json.Marshal(raw.Arms)
	if err != nil {
		t.Fatal(err)
	}
	return string(v)
}

// G4: export and compare report each unavailable Go group on one status
// line, and compare records it in the arm receipt, with the detail the
// caller's context produced. Records, counts and verdicts equal those of the
// same capture without Go groups, and the scratch directory is never
// touched. An absent receipt entry would mean no unavailable observation was
// recorded, not decoder success.
func TestExportGoTraceUnavailable(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	path := goTraceExportCapture(t)
	plain := filepath.Join(t.TempDir(), "plain.pprof")
	writeTestCaptureFile(t, plain, compareCapture(iterationRecs(5), completeStatus()))
	plainTar := filepath.Join(t.TempDir(), "plain.tar")
	stdout := captureStdout(t, func() error {
		return runExportCompare(context.Background(), []string{"-o", plainTar, "-iteration-marker", "Embed", "-iteration-count", "2", plain, plain})
	})
	if got := goTraceUnavailableLines(stdout); len(got) != 0 {
		t.Fatalf("baseline lines = %q", got)
	}
	baseline := goTraceReceiptArms(t, goTraceTarMembers(t, plainTar)[".alignment.json"])
	scratch := t.TempDir()
	if err := os.Chmod(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	options := []string{"-go-trace-decoder=/nonexistent-c6d1/trace", "-go-trace-scratch=" + scratch}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	cap, _, err := readAlignmentCapture(path, cudacapture.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	trace, receipt := sha256.Sum256(cap.Files["stacks-go-1.trace"]), sha256.Sum256(cap.Files["stacks-go-1-host.json"])
	artifacts := make(map[string][]byte)
	verdicts := make(map[string]string)
	for _, tt := range []struct {
		name   string
		ctx    context.Context
		detail string
	}{
		{"cancelled", cancelled, "go trace decode: context canceled"},
		{"unadmitted", context.Background(), "go trace decoder or producer not admitted"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			out := filepath.Join(dir, "x.pftrace")
			stdout := captureStdout(t, func() error {
				return runExport(tt.ctx, goTraceArgs(path, append([]string{"-o", out}, options...)...))
			})
			want := `go trace "stacks-go-1.trace": unavailable: ` + tt.detail
			if got := goTraceUnavailableLines(stdout); len(got) != 1 || got[0] != want {
				t.Errorf("export lines = %q, want [%q]\n%s", got, want, stdout)
			}
			if !strings.Contains(stdout, ": 15 record(s) on 1 track(s)\n") {
				t.Errorf("export summary changed:\n%s", stdout)
			}
			data, err := os.ReadFile(out)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(data, []byte(tt.detail)) {
				t.Errorf("artifact holds the unavailable detail")
			}
			artifacts[tt.name] = data

			tarPath := filepath.Join(dir, "c.tar")
			stdout = captureStdout(t, func() error {
				return runExportCompare(tt.ctx, append(append([]string{"-o", tarPath, "-iteration-marker", "Embed", "-iteration-count", "2"}, options...), path, path))
			})
			got := goTraceUnavailableLines(stdout)
			if len(got) != 2 || got[0] != "base: "+want || got[1] != "target: "+want {
				t.Errorf("compare lines = %q, want base and target %q\n%s", got, want, stdout)
			}
			members := goTraceTarMembers(t, tarPath)
			var r alignmentReceipt
			if err := json.Unmarshal(members[".alignment.json"], &r); err != nil {
				t.Fatal(err)
			}
			entry := goTraceUnavailable{Journal: "stacks-go-1.trace", SHA256: hex.EncodeToString(trace[:]), Receipt: "stacks-go-1-host.json", ReceiptSHA256: hex.EncodeToString(receipt[:]), Detail: tt.detail}
			for i, arm := range r.Arms {
				if len(arm.GoTraceUnavailable) != 1 || arm.GoTraceUnavailable[0] != entry || arm.GoTraceUnavailableOmitted != 0 {
					t.Errorf("arm %d go_trace_unavailable = %+v, want [%+v]", i, arm.GoTraceUnavailable, entry)
				}
				if arm.Records != 15 || arm.Iterations != 2 || arm.Completeness != "passed" || arm.MarkerVerdict != "unchecked" || !arm.Stationarity.Pass || !arm.FullGate.Pass {
					t.Errorf("arm %d records %d, iterations %d, completeness %q, marker %q, stationarity %t, full gate %t", i, arm.Records, arm.Iterations, arm.Completeness, arm.MarkerVerdict, arm.Stationarity.Pass, arm.FullGate.Pass)
				}
			}
			if bytes.Contains(members[".alignment.json"], []byte("go_trace_unavailable_omitted")) || strings.Contains(stdout, "omitted from this report") {
				t.Errorf("omitted count reported with nothing omitted")
			}
			if got := goTraceReceiptArms(t, members[".alignment.json"]); got != baseline {
				t.Errorf("arms differ from the capture without Go groups:\n%s\nwant\n%s", got, baseline)
			}
			verdicts[tt.name] = string(members[".alignment.json"]) + string(members["base.pftrace"]) + string(members["target.pftrace"])

			entries, err := os.ReadDir(scratch)
			if err != nil || len(entries) != 0 {
				t.Errorf("scratch touched: %v %v", entries, err)
			}
		})
	}
	if !bytes.Equal(artifacts["cancelled"], artifacts["unadmitted"]) {
		t.Error("export artifact depends on the unavailable detail")
	}
	if strings.ReplaceAll(verdicts["cancelled"], "go trace decode: context canceled", "go trace decoder or producer not admitted") != verdicts["unadmitted"] {
		t.Error("compare receipt or traces depend on the unavailable detail")
	}
}

func goTraceTarMembers(t *testing.T, path string) map[string][]byte {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	members := make(map[string][]byte)
	tr := tar.NewReader(f)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return members
		}
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(tr)
		if err != nil {
			t.Fatal(err)
		}
		members[h.Name] = data
	}
}

// G4: a journal label or detail holding quotes, line breaks or control
// characters is reported on exactly one line per group.
func TestExportGoTraceUnavailableInjection(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	cap := compareCapture(iterationRecs(5), completeStatus())
	cap.Files["stacks-go-a\"\nb-host.json"] = []byte(`{"x\ny\u001bz\r\u2028":1}`)
	cap.Files["stacks-go-c-host.json"] = []byte(`{` + goTraceExportReceiptFields + `,"goarch":"arm64","go_version":"go1.27.2","q\"\n":1}`)
	path := filepath.Join(t.TempDir(), "inject.pprof")
	writeTestCaptureFile(t, path, cap)
	stdout := captureStdout(t, func() error {
		return runExport(context.Background(), []string{"-o", filepath.Join(t.TempDir(), "x.pftrace"), path})
	})
	want := []string{
		`go trace "stacks-go-a\"\nb-host.json": unavailable: ambiguous host receipt: x; y\x1bz\r\u2028`,
		`go trace "stacks-go-c-host.json": unavailable: ambiguous host receipt: q"; `,
	}
	got := goTraceUnavailableLines(stdout)
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("lines = %q, want %q\nstdout:\n%s", got, want, stdout)
	}
	for _, line := range strings.Split(stdout, "\n") {
		if strings.HasPrefix(line, "y") || strings.HasPrefix(line, "b-host") || strings.HasPrefix(line, `":`) {
			t.Errorf("injected line %q", line)
		}
	}
}

// G4: the detail keeps the recorded error text, so cancellation, an
// unadmitted decoder and a cleanup failure stay distinguishable, and a
// joined error renders on one bounded line.
func TestExportGoTraceUnavailableDetail(t *testing.T) {
	cleanup := fmt.Errorf("%w: %s: %w", languagestack.ErrGoTraceCleanup, "/s/gotrace-1", errors.New("unlinkat: permission denied"))
	cancelled := fmt.Errorf("go trace decode: %w", context.Canceled)
	long := strings.Repeat("é", 600)
	for _, tt := range []struct {
		name, journal, unavailable, want string
	}{
		{"cancelled", "stacks-go-1.trace", cancelled.Error(), `go trace "stacks-go-1.trace": unavailable: go trace decode: context canceled`},
		{"unadmitted", "stacks-go-1.trace", languagestack.ErrGoTraceUnadmitted.Error(), `go trace "stacks-go-1.trace": unavailable: go trace decoder or producer not admitted`},
		{"cleanup only", "stacks-go-1.trace", errors.Join(nil, cleanup).Error(), `go trace "stacks-go-1.trace": unavailable: go trace scratch cleanup failed: /s/gotrace-1: unlinkat: permission denied`},
		{"cancelled then cleanup", "stacks-go-1.trace", errors.Join(cancelled, cleanup).Error(), `go trace "stacks-go-1.trace": unavailable: go trace decode: context canceled; go trace scratch cleanup failed: /s/gotrace-1: unlinkat: permission denied`},
		{"invalid utf-8", "stacks-go-\xff.trace", "bad\xffbyte\x00", `go trace "stacks-go-\xff.trace": unavailable: bad\xffbyte\x00`},
		{"long", "stacks-go-x" + long + ".trace", long, `go trace "stacks-go-x` + strings.Repeat("é", 122) + `...[962 bytes truncated]": unavailable: ` + strings.Repeat("é", 512) + `...[176 bytes truncated]`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, omitted := unavailableLanguageGroups([]languageSamples{{Journal: "stacks-python-wall-1.jsonl"}, {Journal: tt.journal, Unavailable: tt.unavailable}})
			if len(got) != 1 || omitted != 0 {
				t.Fatalf("entries = %+v, omitted %d", got, omitted)
			}
			line := got[0].line()
			if line != tt.want {
				t.Errorf("line =\n%q\nwant\n%q", line, tt.want)
			}
			if strings.ContainsAny(line, "\n\r") || len(got[0].Detail) > maxGoTraceDetailText+32 || len(got[0].Journal) > maxGoTraceJournalText+32 {
				t.Errorf("unbounded or multi-line entry %+v", got[0])
			}
		})
	}
	// The receipt name is bounded to 256 bytes plus the marker; this cut
	// backs off to the rune boundary at 255.
	receipt := "stacks-go-x" + long + "-host.json"
	got, _ := unavailableLanguageGroups([]languageSamples{{Journal: "stacks-go-1.trace", Receipt: receipt, Unavailable: "u"}})
	want := "stacks-go-x" + strings.Repeat("é", 122) + "...[966 bytes truncated]"
	if len(got) != 1 || got[0].Receipt != want {
		t.Errorf("receipt = %+v, want %q", got, want)
	}
}

// goTraceBudgetCapture writes, as root/name/capture.pprof, the iteration
// capture plus one unavailable Go group per key: a host receipt whose only
// key is unknown. Files records the capture's own file bound.
func goTraceBudgetCapture(t *testing.T, root, name string, keys []string, files int) string {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	cap := compareCapture(iterationRecs(5), completeStatus())
	for i, key := range keys {
		cap.Files[fmt.Sprintf("stacks-go-%05d-host.json", i)] = []byte("{" + strconv.Quote(key) + ":1}")
	}
	cap.Limits.Files = files
	path := filepath.Join(dir, "capture.pprof")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.WriteLimits(f, cap, cudacapture.Limits{Recorded: true}); err != nil {
		t.Fatal(err)
	}
	return path
}

// goTraceBudgetRun runs export and compare on path and returns the export
// artifact with the capture identity replaced by plain's, the stdout of
// both, and the alignment receipt. path and plain must have equal lengths.
func goTraceBudgetRun(t *testing.T, path, plain string) (artifact []byte, export, compare string, receipt []byte) {
	t.Helper()
	dir := t.TempDir()
	out := filepath.Join(dir, "x.pftrace")
	export = captureStdout(t, func() error { return runExport(context.Background(), []string{"-o", out, path}) })
	artifact, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if path != plain {
		_, sha, err := readAlignmentCapture(path, cudacapture.Limits{})
		if err != nil {
			t.Fatal(err)
		}
		_, plainSHA, err := readAlignmentCapture(plain, cudacapture.Limits{})
		if err != nil {
			t.Fatal(err)
		}
		artifact = bytes.ReplaceAll(bytes.ReplaceAll(artifact, []byte(sha), []byte(plainSHA)), []byte(path), []byte(plain))
	}
	tarPath := filepath.Join(dir, "c.tar")
	compare = captureStdout(t, func() error {
		return runExportCompare(context.Background(), []string{"-o", tarPath, "-iteration-marker", "Embed", "-iteration-count", "2", path, path})
	})
	return artifact, export, compare, goTraceTarMembers(t, tarPath)[".alignment.json"]
}

// goTraceBudgetArms decodes the report fields of each arm of an alignment
// receipt, keeping the raw encoding of go_trace_unavailable.
func goTraceBudgetArms(t *testing.T, receipt []byte) []alignmentArm {
	t.Helper()
	var r alignmentReceipt
	if err := json.Unmarshal(receipt, &r); err != nil {
		t.Fatal(err)
	}
	for i, arm := range r.Arms {
		if arm.Records != 15 || arm.Iterations != 2 || arm.Completeness != "passed" || arm.MarkerVerdict != "unchecked" || !arm.Stationarity.Pass || !arm.FullGate.Pass {
			t.Errorf("arm %d records %d, iterations %d, completeness %q, marker %q, stationarity %t, full gate %t", i, arm.Records, arm.Iterations, arm.Completeness, arm.MarkerVerdict, arm.Stationarity.Pass, arm.FullGate.Pass)
		}
	}
	return r.Arms[:]
}

// goTraceRawUnavailable returns the raw go_trace_unavailable bytes of each
// arm, as written in the receipt.
func goTraceRawUnavailable(t *testing.T, receipt []byte) [2]json.RawMessage {
	t.Helper()
	var raw struct {
		Arms [2]map[string]json.RawMessage `json:"arms"`
	}
	if err := json.Unmarshal(receipt, &raw); err != nil {
		t.Fatal(err)
	}
	return [2]json.RawMessage{raw.Arms[0]["go_trace_unavailable"], raw.Arms[1]["go_trace_unavailable"]}
}

// G4 report budget: export and each compare arm list at most 64 unavailable
// groups and count the rest. The count is of diagnostic entries omitted
// from this report, not lost profile records, completed decoding, missing
// evidence or a health verdict; zero is not written. Records, verdicts and
// the export artifact match the capture without Go groups.
func TestExportGoTraceUnavailableBudget(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	root := t.TempDir()
	plain := goTraceBudgetCapture(t, root, "p00", nil, 0)
	want, _, _, _ := goTraceBudgetRun(t, plain, plain)
	for _, n := range []int{63, 64, 65} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			keys := make([]string, n)
			for i := range keys {
				keys[i] = "k"
			}
			path := goTraceBudgetCapture(t, root, fmt.Sprintf("c%02d", n), keys, 0)
			artifact, export, compare, receipt := goTraceBudgetRun(t, path, plain)
			listed, omitted := min(n, 64), max(n-64, 0)
			omittedLine := fmt.Sprintf("go trace: %d unavailable group(s) omitted from this report", omitted)
			if got := goTraceUnavailableLines(export); len(got) != listed || got[0] != `go trace "stacks-go-00000-host.json": unavailable: ambiguous host receipt: k` {
				t.Errorf("export listed %d lines, want %d", len(got), listed)
			}
			if got := strings.Count(export, "omitted from this report"); omitted > 0 && (got != 1 || !strings.Contains(export, "\n"+omittedLine+"\n")) || omitted == 0 && got != 0 {
				t.Errorf("export omitted lines = %d, want %q:\n%s", got, omittedLine, export)
			}
			if got := goTraceUnavailableLines(compare); len(got) != 2*listed {
				t.Errorf("compare listed %d lines, want %d", len(got), 2*listed)
			}
			if got := strings.Count(compare, "omitted from this report"); omitted > 0 && (got != 2 || !strings.Contains(compare, "\nbase: "+omittedLine+"\n") || !strings.Contains(compare, "\ntarget: "+omittedLine+"\n")) || omitted == 0 && got != 0 {
				t.Errorf("compare omitted lines = %d, want %q per arm", got, omittedLine)
			}
			for i, arm := range goTraceBudgetArms(t, receipt) {
				if len(arm.GoTraceUnavailable) != listed || arm.GoTraceUnavailableOmitted != omitted {
					t.Errorf("arm %d lists %d, omits %d; want %d, %d", i, len(arm.GoTraceUnavailable), arm.GoTraceUnavailableOmitted, listed, omitted)
				}
			}
			if bytes.Contains(receipt, []byte("go_trace_unavailable_omitted")) != (omitted > 0) {
				t.Errorf("go_trace_unavailable_omitted present = %t with %d omitted", !(omitted > 0), omitted)
			}
			if !bytes.Equal(artifact, want) {
				t.Errorf("export artifact differs from the capture without Go groups")
			}
		})
	}
}

// G4 report budget: with 12000 unavailable groups under a raised recorded
// file bound, each detail holding 1000 '<' that the receipt's JSON encoder
// writes as six bytes each, the encoded go_trace_unavailable field binds
// before the entry cap: it stays within 256 KiB, no further entry would
// fit, and every later group, even one that would fit, is counted as
// omitted from the report.
func TestExportGoTraceUnavailableBudgetEscaped(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	const n = 12000
	root := t.TempDir()
	plain := goTraceBudgetCapture(t, root, "p00000", nil, 2*n)
	want, _, _, _ := goTraceBudgetRun(t, plain, plain)
	key := strings.Repeat("<", 1000)
	keys := make([]string, n)
	for i := range keys {
		keys[i] = key
	}
	keys[n-1] = "k" // would fit, but follows an omitted group
	path := goTraceBudgetCapture(t, root, fmt.Sprint("c", n), keys, 2*n)
	artifact, export, compare, receipt := goTraceBudgetRun(t, path, plain)
	arms := goTraceBudgetArms(t, receipt)
	listed := len(arms[0].GoTraceUnavailable)
	t.Logf("listed %d of %d unavailable groups", listed, n)
	if listed == 0 {
		t.Fatal("no entries listed")
	}
	if listed >= 64 {
		t.Errorf("listed %d entries; want the byte budget to bind below 64", listed)
	}
	omittedLine := fmt.Sprintf("go trace: %d unavailable group(s) omitted from this report", n-listed)
	if got := goTraceUnavailableLines(export); len(got) != listed || !strings.Contains(export, "\n"+omittedLine+"\n") {
		t.Errorf("export listed %d lines, want %d and %q", len(got), listed, omittedLine)
	}
	if got := goTraceUnavailableLines(compare); len(got) != 2*listed || !strings.Contains(compare, "\nbase: "+omittedLine+"\n") || !strings.Contains(compare, "\ntarget: "+omittedLine+"\n") {
		t.Errorf("compare listed %d lines, want %d and %q per arm", len(got), 2*listed, omittedLine)
	}
	detail := "ambiguous host receipt: " + key
	for i, raw := range goTraceRawUnavailable(t, receipt) {
		arm := arms[i]
		if len(arm.GoTraceUnavailable) != listed || arm.GoTraceUnavailableOmitted != n-listed {
			t.Errorf("arm %d lists %d, omits %d; want %d, %d", i, len(arm.GoTraceUnavailable), arm.GoTraceUnavailableOmitted, listed, n-listed)
		}
		if len(raw) > 256<<10 || !bytes.Contains(raw, []byte(`\u003c`)) {
			t.Errorf("arm %d encoded go_trace_unavailable is %d bytes, want escaped and at most %d", i, len(raw), 256<<10)
		}
		for j, e := range arm.GoTraceUnavailable {
			if e.Journal != fmt.Sprintf("stacks-go-%05d-host.json", j) || e.Detail != detail {
				t.Fatalf("arm %d entry %d = %+v", i, j, e)
			}
		}
		next := arm.GoTraceUnavailable[listed-1]
		next.Journal = fmt.Sprintf("stacks-go-%05d-host.json", listed)
		more, err := json.MarshalIndent(append(arm.GoTraceUnavailable, next), "      ", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if len(more) <= 256<<10 {
			t.Errorf("arm %d: %d entries encode in %d bytes; the budget did not bind", i, listed+1, len(more))
		}
	}
	if !bytes.Equal(artifact, want) {
		t.Errorf("export artifact differs from the capture without Go groups")
	}
}

// goTraceScanUnavailable returns the bytes of each go_trace_unavailable
// value in an alignment receipt, found by scanning brackets and strings
// rather than by decoding, and each go_trace_unavailable_omitted count.
func goTraceScanUnavailable(t *testing.T, data []byte) (values [][]byte, omitted []int) {
	t.Helper()
	s := string(data)
	const key = `"go_trace_unavailable": `
	for i := 0; ; {
		j := strings.Index(s[i:], key)
		if j < 0 {
			break
		}
		start := i + j + len(key)
		if s[start] != '[' {
			t.Fatalf("value at %d is not an array", start)
		}
		depth, inString, escaped, end := 0, false, false, -1
		for k := start; k < len(s) && end < 0; k++ {
			switch c := s[k]; {
			case inString && escaped:
				escaped = false
			case inString && c == '\\':
				escaped = true
			case inString && c == '"':
				inString = false
			case inString:
			case c == '"':
				inString = true
			case c == '[' || c == '{':
				depth++
			case c == ']' || c == '}':
				depth--
				if depth == 0 {
					end = k + 1
				}
			}
		}
		if end < 0 {
			t.Fatalf("unterminated value at %d", start)
		}
		values = append(values, data[start:end])
		i = end
	}
	const okey = `"go_trace_unavailable_omitted": `
	for i := 0; ; {
		j := strings.Index(s[i:], okey)
		if j < 0 {
			break
		}
		start := i + j + len(okey)
		k := start
		for k < len(s) && s[k] >= '0' && s[k] <= '9' {
			k++
		}
		n, err := strconv.Atoi(s[start:k])
		if err != nil {
			t.Fatal(err)
		}
		omitted = append(omitted, n)
		i = k
	}
	return values, omitted
}

// G4 report budget, exact boundary: the 256 KiB budget is measured on the
// bytes the receipt writes and is inclusive. Unavailable groups are tuned so
// each arm's written go_trace_unavailable value is exactly 262144 bytes;
// that last entry is listed and the next small group is omitted. Adapted
// from the budget review's TestReviewBudgetExactBoundary.
func TestExportGoTraceUnavailableBudgetExact(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	const budget = 256 << 10
	root := t.TempDir()
	runs := 0
	run := func(keys []string) (values [][]byte, omitted []int, stdout string, receipt []byte) {
		runs++
		path := goTraceBudgetCapture(t, root, fmt.Sprint("r", runs), keys, 0)
		tarPath := filepath.Join(root, fmt.Sprint("r", runs, ".tar"))
		stdout = captureStdout(t, func() error {
			return runExportCompare(context.Background(), []string{"-o", tarPath, "-iteration-marker", "Embed", "-iteration-count", "2", path, path})
		})
		receipt = goTraceTarMembers(t, tarPath)[".alignment.json"]
		values, omitted = goTraceScanUnavailable(t, receipt)
		return values, omitted, stdout, receipt
	}
	entries := func(v []byte) int { return strings.Count(string(v), `"journal": `) }
	big := strings.Repeat("<", 1000)
	keys := make([]string, 41)
	for i := range keys {
		keys[i] = big
	}
	// Grow with large groups until a final group with an empty key leaves
	// less than one large detail of room; then pad that key to fill it.
	var room int
	for {
		values, _, _, _ := run(append(append([]string(nil), keys...), ""))
		if len(values) != 2 || entries(values[0]) != len(keys)+1 {
			t.Fatalf("probe listed %d of %d", entries(values[0]), len(keys)+1)
		}
		room = budget - len(values[0])
		if room <= 6*len(big)-6 {
			break
		}
		keys = append(keys, big)
	}
	if room < 0 || room/6+room%6 > len(big) {
		t.Fatalf("room %d cannot be filled by one detail", room)
	}
	n := len(keys) + 1
	keys = append(keys, strings.Repeat("<", room/6)+strings.Repeat("x", room%6), "k")
	values, omitted, stdout, receipt := run(keys)
	if len(values) != 2 {
		t.Fatalf("receipt holds %d go_trace_unavailable values", len(values))
	}
	for i, v := range values {
		if len(v) != budget || entries(v) != n {
			t.Errorf("arm %d writes %d bytes in %d entries; want %d in %d", i, len(v), entries(v), budget, n)
		}
	}
	if len(omitted) != 2 || omitted[0] != 1 || omitted[1] != 1 {
		t.Errorf("omitted = %v, want [1 1]", omitted)
	}
	if got := strings.Count(stdout, ": go trace: 1 unavailable group(s) omitted from this report\n"); got != 2 {
		t.Errorf("compare omitted lines = %d, want 2:\n%s", got, stdout)
	}
	goTraceBudgetArms(t, receipt)
}

// Q7: every help surface names the options.
func TestExportGoTraceHelp(t *testing.T) {
	var text []string
	for _, c := range traceCommands() {
		if c.name != "export" && c.name != "compare" && c.name != "query" {
			continue
		}
		out, err := goTraceStdout(func() { traceHelp(c, "trace "+c.name) })
		if err != nil {
			t.Fatal(err)
		}
		text = append(text, out)
	}
	export, err := goTraceStdout(printExportHelp)
	if err != nil {
		t.Fatal(err)
	}
	var query bytes.Buffer
	if err := runTraceQuery([]string{"-h"}, &query); err != nil {
		t.Fatal(err)
	}
	text = append(text, export, exportCompareUsage, query.String())
	if len(text) != 6 {
		t.Fatalf("help surfaces = %d", len(text))
	}
	for i, s := range text {
		if !strings.Contains(s, "-go-trace-decoder") || !strings.Contains(s, "-go-trace-scratch") {
			t.Errorf("help %d lacks the go trace options:\n%s", i, s)
		}
	}
}

func goTraceStdout(fn func()) (string, error) {
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		return "", err
	}
	os.Stdout = w
	done := make(chan []byte, 1)
	go func() {
		data, _ := io.ReadAll(r)
		done <- data
	}()
	fn()
	os.Stdout = orig
	w.Close()
	data := <-done
	r.Close()
	return string(data), nil
}
