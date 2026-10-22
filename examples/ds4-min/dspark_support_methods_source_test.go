package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// callerTests are the named tests in testdata/native_support_methods.txt.
// Every control run must finish each of them exactly once.
var callerTests = []string{"TestCallerWrites", "TestCallerLoads", "TestCallerFailures", "TestCallerOwner"}

// A callerControl is one child run of the caller fixture. The baseline names
// no test. A mutation names the test that must fail with assertion and any
// other tests the same mutation must also fail; all remaining tests must pass.
type callerControl struct {
	name, old, replacement string
	test, assertion        string
	collateral             []string
}

var callerControls = []callerControl{
	{name: "baseline"},
	{
		name:        "cuda-write",
		old:         "return writeDSparkSupportContainer(w, core.Bytes(), e, v.pos, v.generation, uint64(cudaStateMaxBytes))",
		replacement: "return nil",
		test:        "TestCallerWrites",
		assertion:   "cuda caller omitted or changed original container bytes",
	},
	{
		name:        "metal-write",
		old:         "return writeDSparkSupportContainer(w, v.data, e, v.dspark.journal.frontier, 0, uint64(metalStateMaxBytes))",
		replacement: "_ = e; return nil",
		test:        "TestCallerWrites",
		assertion:   "metal caller omitted or changed original container bytes",
	},
	{
		name:        "cuda-read",
		old:         "readDSparkSupportContainer(r, true, uint64(cudaStateMaxBytes))",
		replacement: "readDSparkSupportContainer(r, false, uint64(cudaStateMaxBytes))",
		test:        "TestCallerLoads",
		assertion:   "cuda caller failed original envelope load",
		collateral:  []string{"TestCallerFailures", "TestCallerOwner"},
	},
	{
		name:        "metal-read",
		old:         "readDSparkSupportContainer(r, false, uint64(metalStateMaxBytes))",
		replacement: "readDSparkSupportContainer(r, true, uint64(metalStateMaxBytes))",
		test:        "TestCallerLoads",
		assertion:   "metal caller failed original envelope load",
		collateral:  []string{"TestCallerFailures"},
	},
	{
		name:        "cuda-restore",
		old:         "s.dspark.restore(v, int(position), generation)",
		replacement: "fixtureNoRestore(v, int(position), generation)",
		test:        "TestCallerLoads",
		assertion:   "cuda caller did not publish support values",
	},
	{
		name:        "cuda-rollback",
		old:         "return s.restoreSnapshot(nativeBackup)",
		replacement: "return nil",
		test:        "TestCallerFailures",
		assertion:   "cuda caller rollback lost destination",
		collateral:  []string{"TestCallerOwner"},
	},
	{
		name:        "cuda-owner",
		old:         "finishBackendCleanup(err, stage)",
		replacement: "errors.Join(err, stage.Close())",
		test:        "TestCallerOwner",
		assertion:   "cuda caller lost retryable cleanup owner",
	},
	{
		name:        "metal-publish",
		old:         "f.publishSnapshot(v)",
		replacement: "fixtureNoPublish(v)",
		test:        "TestCallerLoads",
		assertion:   "metal caller did not publish support values",
	},
}

// forbiddenOutput marks child runs that are not evidence for any control.
var forbiddenOutput = []string{"panic:", "[build failed]", "[setup failed]", "no test files", "test timed out", "[no tests to run]"}

// checkCallerControl reports whether the go test -json output of one child
// run, which exited with rc, is exactly the outcome c requires.
func checkCallerControl(c callerControl, out []byte, rc int) error {
	wantRC, wantPackage := 1, "fail"
	if c.test == "" {
		wantRC, wantPackage = 0, "pass"
	}
	if rc != wantRC {
		return fmt.Errorf("exit code %d, want %d", rc, wantRC)
	}
	terminal := make(map[string][]string)
	output := make(map[string]*strings.Builder)
	var pkg []string
	scanner := bufio.NewScanner(bytes.NewReader(out))
	scanner.Buffer(nil, 16<<20)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var event struct {
			Action, Test, Output, FailedBuild string
		}
		if err := json.Unmarshal(line, &event); err != nil {
			return fmt.Errorf("invalid test event %q: %v", line, err)
		}
		if event.FailedBuild != "" {
			return fmt.Errorf("build failure in %s", event.FailedBuild)
		}
		for _, bad := range forbiddenOutput {
			if strings.Contains(event.Output, bad) {
				return fmt.Errorf("forbidden output %q in %q", bad, event.Output)
			}
		}
		switch event.Action {
		case "start", "run", "pause", "cont":
		case "output":
			if event.Test != "" {
				b := output[event.Test]
				if b == nil {
					b = new(strings.Builder)
					output[event.Test] = b
				}
				b.WriteString(event.Output)
			}
		case "pass", "fail":
			if event.Test == "" {
				pkg = append(pkg, event.Action)
			} else {
				terminal[event.Test] = append(terminal[event.Test], event.Action)
			}
		default:
			return fmt.Errorf("forbidden action %q for %q", event.Action, event.Test)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read test events: %v", err)
	}
	if len(pkg) != 1 || pkg[0] != wantPackage {
		return fmt.Errorf("package results %v, want [%s]", pkg, wantPackage)
	}
	for name := range terminal {
		if !slices.Contains(callerTests, name) {
			return fmt.Errorf("unexpected test %s", name)
		}
	}
	for _, name := range callerTests {
		actions := terminal[name]
		if len(actions) != 1 {
			return fmt.Errorf("%s has terminal actions %v, want exactly one", name, actions)
		}
		want := "pass"
		if name == c.test || slices.Contains(c.collateral, name) {
			want = "fail"
		}
		if actions[0] != want {
			return fmt.Errorf("%s %s, want %s", name, actions[0], want)
		}
		if c.assertion == "" {
			continue
		}
		logged := output[name] != nil && strings.Contains(output[name].String(), c.assertion)
		if name == c.test && !logged {
			return fmt.Errorf("%s failed without assertion %q", name, c.assertion)
		}
		if name != c.test && logged {
			return fmt.Errorf("assertion %q logged by %s, not %s", c.assertion, name, c.test)
		}
	}
	return nil
}

// These controls run complete persistence caller bodies with host buffers.
// They do not qualify native transfers, core serialization or synchronization.
func TestNativeSupportPersistenceCallers(t *testing.T) {
	read := func(name string) string {
		t.Helper()
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	cuda := read("dspark_cuda_persistence_linux.go")
	metal := read("dspark_metal_persistence_darwin.go")
	body := ""
	for _, item := range []struct{ source, signature string }{
		{cuda, "func cudaSupportShape("},
		{cuda, "func cudaSupportIdentity("},
		{cuda, "func writeCUDASupportSnapshot("},
		{cuda, "func (s *cudaRequestState) loadSupportWithHooks("},
		{metal, "func metalSupportEnvelope("},
		{metal, "func writeMetalStateSnapshot("},
		{metal, "func (s *metalRequestState) loadSupport("},
		{read("cuda_state_identity_linux.go"), "func cudaStateTransaction("},
	} {
		body += dsparkSourceFunction(t, item.source, item.signature) + "\n"
	}
	body = strings.ReplaceAll(body, "dsv4.ErrClosed", "fixtureErrClosed")
	body = strings.ReplaceAll(body, "dsv4.ErrDesynchronized", "fixtureErrDesynchronized")
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	module := filepath.Dir(cwd)
	gomod, gosum := callerFixtureModule(t, module)
	fixture := read("testdata/native_support_methods.txt")
	fixture += "\n" + dsparkSourceFunction(t, read("dspark_support_codec_test.go"), "func supportCodecFixture(")
	for _, tc := range callerControls {
		t.Run(tc.name, func(t *testing.T) {
			source := body
			if tc.old != "" {
				if strings.Count(source, tc.old) != 1 {
					t.Fatal("actual-body mutation anchor is not unique", tc.name)
				}
				source = strings.Replace(source, tc.old, tc.replacement, 1)
			}
			dir := t.TempDir()
			files := map[string]string{
				"go.mod":           gomod,
				"go.sum":           gosum,
				"methods.go":       "package main\nimport(\"bytes\";\"encoding/binary\";\"encoding/hex\";\"errors\";\"fmt\";\"io\")\n" + source,
				"methods_test.go":  fixture,
				"journal.go":       read("dspark_feature_journal.go"),
				"codec.go":         read("dspark_support_codec.go"),
				"container.go":     read("dspark_support_container.go"),
				"original_test.go": read("dspark_support_retained_writer_test.go"),
				"cleanup.go":       read("backend_cleanup.go"),
			}
			for name, contents := range files {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0600); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, "go", "test", "-mod=readonly", "-json", "-count=1", "-timeout=90s", ".")
			cmd.Dir = dir
			cmd.Env = append(os.Environ(), "GOWORK=off", "GOPROXY=off", "GOSUMDB=off", "GOFLAGS=-mod=readonly")
			out, runErr := cmd.CombinedOutput()
			rc := -1
			var exit *exec.ExitError
			if runErr == nil || errors.As(runErr, &exit) {
				rc = cmd.ProcessState.ExitCode()
			}
			t.Logf("raw caller control %s rc=%d:\n%s", tc.name, rc, out)
			if receipts := os.Getenv("DS4_SUPPORT_METHOD_RECEIPTS"); receipts != "" {
				if !filepath.IsAbs(receipts) {
					t.Fatal("caller receipt directory must be absolute")
				}
				if err := os.MkdirAll(receipts, 0700); err != nil {
					t.Fatal(err)
				}
				for suffix, data := range map[string][]byte{"json": out, "rc": fmt.Appendf(nil, "%d\n", rc)} {
					f, err := os.OpenFile(filepath.Join(receipts, tc.name+"."+suffix), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
					if err != nil {
						t.Fatal(err)
					}
					_, writeErr := f.Write(data)
					closeErr := f.Close()
					if writeErr != nil || closeErr != nil {
						t.Fatal(writeErr, closeErr)
					}
				}
			}
			if ctx.Err() != nil {
				t.Fatalf("caller control timeout: %v", ctx.Err())
			}
			if err := checkCallerControl(tc, out, rc); err != nil {
				t.Fatalf("caller control %s rejected: %v", tc.name, err)
			}
		})
	}
}

// callerFixtureModule returns the go.mod and go.sum of the caller fixture
// module. They are the real go.mod and go.sum of module, edited by go mod edit
// to rename the module, require and replace module itself, and make module's
// relative replacements absolute so that they resolve from a temporary
// directory. go.sum is copied unchanged.
func callerFixtureModule(t *testing.T, module string) (gomod, gosum string) {
	t.Helper()
	const examples = "github.com/tmc/lib/examples"
	dir := t.TempDir()
	for _, name := range []string{"go.mod", "go.sum"} {
		b, err := os.ReadFile(filepath.Join(module, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	gomodEdit := func(args ...string) []byte {
		t.Helper()
		cmd := exec.Command("go", append([]string{"mod", "edit"}, args...)...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=")
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("go mod edit %v: %v", args, err)
		}
		return out
	}
	var mod struct {
		Module  struct{ Path string }
		Replace []struct {
			Old, New struct{ Path, Version string }
		}
	}
	if err := json.Unmarshal(gomodEdit("-json"), &mod); err != nil {
		t.Fatal(err)
	}
	if mod.Module.Path != examples {
		t.Fatalf("module root %s holds module %q, want %q", module, mod.Module.Path, examples)
	}
	args := []string{
		"-module=" + examples + "/ds4-min/methodfixture",
		"-require=" + examples + "@v0.0.0",
		"-replace=" + examples + "=" + module,
	}
	for _, r := range mod.Replace {
		if r.New.Version != "" || filepath.IsAbs(r.New.Path) {
			continue
		}
		old := r.Old.Path
		if r.Old.Version != "" {
			old += "@" + r.Old.Version
		}
		args = append(args, "-replace="+old+"="+filepath.Join(module, r.New.Path))
	}
	gomodEdit(args...)
	b, err := os.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	s, err := os.ReadFile(filepath.Join(dir, "go.sum"))
	if err != nil {
		t.Fatal(err)
	}
	return string(b), string(s)
}

// callerStream is a synthetic go test -json stream for parser controls.
type callerStream struct {
	results map[string]string // test name to comma-separated terminal actions
	output  map[string]string // test name to its output
	pre     []string          // raw lines written before the first event
	extra   []string          // raw lines written before the package result
	pkg     string            // package terminal action; "" omits it
}

func (s callerStream) bytes() []byte {
	var b bytes.Buffer
	write := func(action, test, output string) {
		line, err := json.Marshal(struct {
			Action  string
			Package string
			Test    string `json:",omitempty"`
			Output  string `json:",omitempty"`
		}{action, "github.com/tmc/lib/examples/ds4-min/methodfixture", test, output})
		if err != nil {
			panic(err)
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	for _, line := range s.pre {
		b.WriteString(line + "\n")
	}
	write("start", "", "")
	names := make([]string, 0, len(s.results))
	for name := range s.results {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		write("run", name, "")
		if o := s.output[name]; o != "" {
			write("output", name, o)
		}
		for _, action := range strings.Split(s.results[name], ",") {
			write(action, name, "")
		}
	}
	for _, line := range s.extra {
		b.WriteString(line + "\n")
	}
	if s.pkg != "" {
		write(s.pkg, "", "")
	}
	return b.Bytes()
}

// The parser must accept the exact intended outcome and reject every
// near miss, so that a mutation is never credited for an unrelated failure.
func TestNativeSupportCallerParser(t *testing.T) {
	const pkgLine = `"Package":"github.com/tmc/lib/examples/ds4-min/methodfixture"`
	control := callerControl{
		name:       "cuda-read",
		test:       "TestCallerLoads",
		assertion:  "cuda caller failed original envelope load",
		collateral: []string{"TestCallerFailures"},
	}
	valid := func() callerStream {
		return callerStream{
			results: map[string]string{"TestCallerWrites": "pass", "TestCallerLoads": "fail", "TestCallerFailures": "fail", "TestCallerOwner": "pass"},
			output: map[string]string{
				"TestCallerLoads":    "    methods_test.go:117: cuda caller failed original envelope load\n",
				"TestCallerFailures": "    methods_test.go:134: cuda caller rollback lost destination\n",
			},
			pkg: "fail",
		}
	}
	allPass := func() callerStream {
		return callerStream{
			results: map[string]string{"TestCallerWrites": "pass", "TestCallerLoads": "pass", "TestCallerFailures": "pass", "TestCallerOwner": "pass"},
			output:  map[string]string{},
			pkg:     "pass",
		}
	}
	baseline := callerControls[0]
	if err := checkCallerControl(control, valid().bytes(), 1); err != nil {
		t.Fatalf("intended failure stream rejected: %v", err)
	}
	if err := checkCallerControl(baseline, allPass().bytes(), 0); err != nil {
		t.Fatalf("baseline stream rejected: %v", err)
	}
	// goModSetup is the complete output of every wave1 child run, whose
	// fixture go.mod lacked the requirements of the module it replaced.
	const goModSetup = "go: updates to go.mod needed, disabled by -mod=readonly; to update it:\n\tgo mod tidy\n"
	for _, tc := range []struct {
		name     string
		baseline bool
		rc       int
		raw      string // used as the whole stream when set
		edit     func(*callerStream)
	}{
		{name: "go-mod-setup", rc: 1, raw: goModSetup},
		{name: "go-mod-setup-baseline", baseline: true, rc: 1, raw: goModSetup},
		{name: "go-mod-setup-before-json", rc: 1, edit: func(s *callerStream) {
			s.pre = strings.Split(strings.TrimSuffix(goModSetup, "\n"), "\n")
		}},
		{name: "rc-2", rc: 2},
		{name: "rc-0", rc: 0},
		{name: "rc-unknown", rc: -1},
		{name: "missing-test", rc: 1, edit: func(s *callerStream) { delete(s.results, "TestCallerOwner") }},
		{name: "duplicate-test", rc: 1, edit: func(s *callerStream) { s.results["TestCallerWrites"] = "pass,pass" }},
		{name: "duplicate-intended", rc: 1, edit: func(s *callerStream) { s.results["TestCallerLoads"] = "fail,fail" }},
		{name: "unexpected-test", rc: 1, edit: func(s *callerStream) { s.results["TestCallerExtra"] = "pass" }},
		{name: "unrelated-failure-carries-assertion", rc: 1, edit: func(s *callerStream) {
			s.results["TestCallerLoads"] = "pass"
			s.results["TestCallerOwner"] = "fail"
			s.output["TestCallerOwner"] = s.output["TestCallerLoads"]
			delete(s.output, "TestCallerLoads")
		}},
		{name: "collateral-carries-assertion", rc: 1, edit: func(s *callerStream) {
			s.output["TestCallerFailures"] += s.output["TestCallerLoads"]
		}},
		{name: "assertion-by-passing-test", rc: 1, edit: func(s *callerStream) {
			s.output["TestCallerWrites"] = s.output["TestCallerLoads"]
			s.output["TestCallerLoads"] = "    methods_test.go:118: cuda caller did not publish support values\n"
		}},
		{name: "intended-without-assertion", rc: 1, edit: func(s *callerStream) { delete(s.output, "TestCallerLoads") }},
		{name: "collateral-passes", rc: 1, edit: func(s *callerStream) { s.results["TestCallerFailures"] = "pass" }},
		{name: "unaffected-fails", rc: 1, edit: func(s *callerStream) { s.results["TestCallerWrites"] = "fail" }},
		{name: "no-package-fail", rc: 1, edit: func(s *callerStream) { s.pkg = "" }},
		{name: "package-pass", rc: 1, edit: func(s *callerStream) { s.pkg = "pass" }},
		{name: "duplicate-package-fail", rc: 1, edit: func(s *callerStream) {
			s.extra = append(s.extra, `{"Action":"fail",`+pkgLine+`}`)
		}},
		{name: "skip", rc: 1, edit: func(s *callerStream) { s.results["TestCallerWrites"] = "skip" }},
		{name: "panic", rc: 1, edit: func(s *callerStream) {
			s.output["TestCallerLoads"] += "panic: runtime error: invalid memory address or nil pointer dereference\n"
		}},
		{name: "timeout", rc: 1, edit: func(s *callerStream) {
			s.extra = append(s.extra, `{"Action":"output",`+pkgLine+`,"Output":"panic: test timed out after 1m30s\n"}`)
		}},
		{name: "setup-failure", rc: 1, edit: func(s *callerStream) {
			s.extra = append(s.extra, `{"Action":"output",`+pkgLine+`,"Output":"FAIL\tgithub.com/tmc/lib/examples/ds4-min/methodfixture [setup failed]\n"}`)
		}},
		{name: "build-failure", rc: 1, edit: func(s *callerStream) {
			s.extra = append(s.extra, `{"Action":"build-fail","ImportPath":"github.com/tmc/lib/examples/ds4-min/methodfixture"}`)
		}},
		{name: "failed-build-field", rc: 1, edit: func(s *callerStream) {
			s.extra = append(s.extra, `{"Action":"output",`+pkgLine+`,"FailedBuild":"github.com/tmc/lib/examples/ds4-min/methodfixture","Output":"FAIL\n"}`)
		}},
		{name: "non-json", rc: 1, edit: func(s *callerStream) { s.extra = append(s.extra, "go: downloading example.com/m v1.0.0") }},
		{name: "baseline-with-failure", baseline: true, rc: 0, edit: func(s *callerStream) {
			*s = allPass()
			s.results["TestCallerLoads"] = "fail"
		}},
		{name: "baseline-rc-1", baseline: true, rc: 1, edit: func(s *callerStream) { *s = allPass() }},
		{name: "baseline-package-fail", baseline: true, rc: 0, edit: func(s *callerStream) {
			*s = allPass()
			s.pkg = "fail"
		}},
		{name: "baseline-missing-test", baseline: true, rc: 0, edit: func(s *callerStream) {
			*s = allPass()
			delete(s.results, "TestCallerOwner")
		}},
		{name: "baseline-skip", baseline: true, rc: 0, edit: func(s *callerStream) {
			*s = allPass()
			s.results["TestCallerOwner"] = "skip"
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := control
			if tc.baseline {
				c = baseline
			}
			s := valid()
			if tc.edit != nil {
				tc.edit(&s)
			}
			out := s.bytes()
			if tc.raw != "" {
				out = []byte(tc.raw)
			}
			err := checkCallerControl(c, out, tc.rc)
			if err == nil {
				t.Fatalf("parser accepted %s", tc.name)
			}
			t.Logf("rejected: %v", err)
		})
	}
}
