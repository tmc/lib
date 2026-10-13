package main

import (
	"bytes"
	"context"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"os"
	"path/filepath"
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
// nothing for an unavailable Go group, so their outputs cannot show which
// context or options the decoder received; this gate checks the exact
// arguments of each call on the chain instead.
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
		"runExport->appendLanguageSamples":              "ctx, cap, records, opts.goTrace",
		"runExportCompareCommand->runExportCompare":     "ctx, args",
		"runExportCompare->loadAlignmentArm":            "ctx, opts, i",
		"loadAlignmentArm->appendLanguageSamples":       "ctx, cap, arm.records, opts.goTrace",
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
