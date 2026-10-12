package cudacluster

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// clusterTarget returns the architecture the reduction fixture is compiled
// for. CUDA_CLUSTER_NATIVE_ARCH names another.
func clusterTarget() string {
	if target := os.Getenv("CUDA_CLUSTER_NATIVE_ARCH"); target != "" {
		return target
	}
	return "sm_121a"
}

// runReduction returns the total the reduction fixture produced on a device,
// gated by the process environment.
func runReduction(t *testing.T, clusterSize int, mut Mutant) (uint32, error) {
	t.Helper()
	return runReductionEnv(t, os.Getenv, clusterSize, mut)
}

// runReductionEnv is runReduction gated by getenv. A closed native gate skips
// t with an error wrapping [ErrHardwarePending]; it never yields a total.
// Every other failure, including an open gate without an inherited GPU lock,
// is returned for the caller to fail on.
func runReductionEnv(t *testing.T, getenv func(string) string, clusterSize int, mut Mutant) (uint32, error) {
	t.Helper()
	got, err := reduceOnDevice(t, getenv, clusterSize, mut)
	skipClosedGate(t, err)
	return got, err
}

// reduceOnDevice compiles and runs the fixture and returns the total it
// produced. The native gate is checked before any tool lookup, and a closed
// gate returns (0, errNativeGateClosed). A non-zero exit from the fixture is
// returned as an error naming the stage that failed, so that a compile failure
// and a wrong answer are never confused.
func reduceOnDevice(t *testing.T, getenv func(string) string, clusterSize int, mut Mutant) (uint32, error) {
	t.Helper()
	if err := requireNativeLock(getenv); err != nil {
		return 0, err
	}
	target := clusterTarget()
	if err := Check(target); err != nil {
		return 0, fmt.Errorf("check %s: %w", target, err)
	}
	var nvcc string
	if os.Getenv("CUDA_CLUSTER_PREBUILT_DIR") == "" {
		nvcc, _ = cudatest.NVCC(t)
	}

	src, err := ReductionFixture(target, clusterSize, mut)
	if err != nil {
		return 0, err
	}
	src += reductionMain(clusterSize)

	dir := t.TempDir()

	got, err := compileRunTotal(t, nvcc, target, dir, "reduce", src)
	if err != nil {
		return 0, err
	}
	if mut == MutantNone {
		oracle := cooperativeReductionFixture(clusterSize)
		want, err := compileRunTotal(t, nvcc, target, dir, "oracle", oracle)
		if err := compareOracleTotal(got, want, err); err != nil {
			return 0, err
		}
	}
	return got, nil
}

func compareOracleTotal(emitted, oracle uint32, oracleErr error) error {
	if oracleErr != nil {
		return fmt.Errorf("cooperative-groups oracle: %w", oracleErr)
	}
	if emitted != oracle {
		return fmt.Errorf("emitted PTX total %d differs from cooperative-groups oracle %d", emitted, oracle)
	}
	return nil
}

func reductionMain(clusterSize int) string {
	return fmt.Sprintf(`#include <cuda_runtime.h>
#include <stdio.h>
int main() {
  unsigned *d = 0, h = 0;
  if (cudaMalloc(&d, sizeof h) != cudaSuccess) return 2;
  if (cudaMemset(d, 0, sizeof h) != cudaSuccess) return 3;
  dsmem_reduce<<<%d, 32>>>(d);
  if (cudaGetLastError() != cudaSuccess) return %d;
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  if (cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
  cudaFree(d);
  printf("total %%u\n", h);
  return 0;
}
`, clusterSize, fixtureLaunchFailureCode)
}

// errNativeGateClosed reports that CUDA_CLUSTER_NATIVE=1 is unset, so no
// native fixture ran. It is pending, not a refusal: nothing examined a device.
var errNativeGateClosed error = nativeGateClosedError{}

// nativeGateClosedError matches [ErrHardwarePending] without repeating its
// text, which describes a target that was checked. A closed gate checked none.
type nativeGateClosedError struct{}

func (nativeGateClosedError) Error() string {
	return "cudacluster: native gate is closed: CUDA_CLUSTER_NATIVE=1 unset, no device examined"
}

func (nativeGateClosedError) Unwrap() error { return ErrHardwarePending }

// gateReporter is the part of [testing.T] that reports a native gate. Each
// method ends the test when called through a *testing.T.
type gateReporter interface {
	Helper()
	Skip(args ...any)
	Fatal(args ...any)
	Log(args ...any)
}

// skipClosedGate skips r when err is the closed native gate, and only then.
// Any other error, including another ErrHardwarePending such as a target with
// no recorded capability, is left for the caller to fail on.
func skipClosedGate(r gateReporter, err error) {
	r.Helper()
	if errors.Is(err, errNativeGateClosed) {
		r.Skip(err)
	}
}

// reportNativeGate reports the native gate for a run: a pass when the gate
// is open under an inherited lock, a pending skip when it is closed in an
// untagged build, and a failure otherwise.
func reportNativeGate(r gateReporter, native bool, gateErr error) {
	r.Helper()
	err := nativeRunError(native, gateErr)
	switch {
	case err == nil:
		r.Log("native gate open under an inherited GPU lock")
	case errors.Is(err, errVacuousNativeRun):
		r.Fatal(err)
	case errors.Is(err, errNativeGateClosed):
		r.Skip(err)
	default:
		r.Fatal(err)
	}
}

// recordedGate records how a gate was reported. Unlike *testing.T, its
// methods return, so the reporting functions must not fall through.
type recordedGate struct {
	calls []string
	text  string
}

func (r *recordedGate) Helper() {}

func (r *recordedGate) record(call string, args []any) {
	r.calls = append(r.calls, call)
	r.text = fmt.Sprint(args...)
}

func (r *recordedGate) Skip(args ...any)  { r.record("skip", args) }
func (r *recordedGate) Fatal(args ...any) { r.record("fatal", args) }
func (r *recordedGate) Log(args ...any)   { r.record("log", args) }

func TestSkipClosedGate(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want []string
	}{
		{"closed gate", errNativeGateClosed, []string{"skip"}},
		{"wrapped closed gate", fmt.Errorf("run: %w", errNativeGateClosed), []string{"skip"}},
		{"target pending", fmt.Errorf("check sm_86: %w", ErrHardwarePending), nil},
		{"no lock", fmt.Errorf("lock: %w", cudalock.ErrNotInherited), nil},
		{"fixture failed", errors.New("fixture exited 1"), nil},
		{"no error", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var r recordedGate
			skipClosedGate(&r, tc.err)
			if !slices.Equal(r.calls, tc.want) {
				t.Fatalf("skipClosedGate(%v) calls = %q, want %q", tc.err, r.calls, tc.want)
			}
		})
	}
}

func TestReportNativeGate(t *testing.T) {
	lockErr := fmt.Errorf("lock: %w", cudalock.ErrNotInherited)
	for _, tc := range []struct {
		name    string
		native  bool
		gateErr error
		want    string
	}{
		{"untagged closed", false, errNativeGateClosed, "skip"},
		{"native closed", true, errNativeGateClosed, "fatal"},
		{"untagged open", false, nil, "log"},
		{"native open", true, nil, "log"},
		{"untagged without lock", false, lockErr, "fatal"},
		{"native without lock", true, lockErr, "fatal"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var r recordedGate
			reportNativeGate(&r, tc.native, tc.gateErr)
			if !slices.Equal(r.calls, []string{tc.want}) {
				t.Fatalf("reportNativeGate(%t, %v) calls = %q, want [%s]", tc.native, tc.gateErr, r.calls, tc.want)
			}
		})
	}
}

// TestClosedGateSkipText requires the skip to say the native gate is closed
// and not to claim, through the sentinel's text, that a target was checked.
func TestClosedGateSkipText(t *testing.T) {
	var r recordedGate
	skipClosedGate(&r, errNativeGateClosed)
	if !strings.Contains(r.text, "native gate is closed") {
		t.Errorf("closed gate skip = %q, want it to say the native gate is closed", r.text)
	}
	if strings.Contains(r.text, ErrHardwarePending.Error()) {
		t.Errorf("closed gate skip = %q repeats %q", r.text, ErrHardwarePending)
	}
	if !errors.Is(errNativeGateClosed, ErrHardwarePending) {
		t.Error("closed gate does not match ErrHardwarePending")
	}
}

// requireNativeLock returns errNativeGateClosed when the native gate is
// closed, and a non-pending error when it is open without an inherited GPU
// lock.
func requireNativeLock(getenv func(string) string) error {
	if getenv("CUDA_CLUSTER_NATIVE") != "1" {
		return errNativeGateClosed
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("cudacluster: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

// errVacuousNativeRun reports a build tagged native whose native gate is
// closed. Every native test in it would skip, and the run would pass having
// measured nothing.
var errVacuousNativeRun = errors.New("cudacluster: built with -tags=native but every native test would skip")

// nativeRunError classifies the native gate for a run. In a build tagged
// native a closed gate is errVacuousNativeRun rather than pending, so that
// the native acceptance command cannot pass without opening the gate.
func nativeRunError(native bool, gateErr error) error {
	if native && errors.Is(gateErr, errNativeGateClosed) {
		return fmt.Errorf("%w: %w", errVacuousNativeRun, gateErr)
	}
	return gateErr
}

// TestNativeGateState reports the state of the native gate as its own result:
// skipped as pending when the gate is closed in an untagged build, failed when
// a build tagged native has a closed gate or when the gate is open without an
// inherited GPU lock, and passed only with the gate open under such a lock. It
// does not open the gate or run anything on a device.
func TestNativeGateState(t *testing.T) {
	reportNativeGate(t, nativeBuild, requireNativeLock(os.Getenv))
}

func TestNativeRunError(t *testing.T) {
	lockErr := errors.New("no lock")
	for _, tc := range []struct {
		name        string
		native      bool
		gateErr     error
		wantNil     bool
		wantVacuous bool
		wantPending bool
	}{
		{name: "untagged closed", gateErr: errNativeGateClosed, wantPending: true},
		{name: "native closed", native: true, gateErr: errNativeGateClosed, wantVacuous: true, wantPending: true},
		{name: "untagged open", wantNil: true},
		{name: "native open", native: true, wantNil: true},
		{name: "native open without lock", native: true, gateErr: lockErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := nativeRunError(tc.native, tc.gateErr)
			if (err == nil) != tc.wantNil || errors.Is(err, errVacuousNativeRun) != tc.wantVacuous || errors.Is(err, ErrHardwarePending) != tc.wantPending {
				t.Fatalf("nativeRunError(%t, %v) = %v, want nil=%t vacuous=%t pending=%t", tc.native, tc.gateErr, err, tc.wantNil, tc.wantVacuous, tc.wantPending)
			}
			if tc.gateErr != nil && !errors.Is(err, tc.gateErr) {
				t.Fatalf("nativeRunError dropped the gate error: %v", err)
			}
		})
	}
}

// nativeMarkers maps each file declaring nativeBuild to the build constraint
// that selects it and the value it must declare.
var nativeMarkers = []struct {
	file       string
	constraint string
	value      bool
}{
	{"native_tag_enabled_test.go", "native", true},
	{"native_tag_disabled_test.go", "!native", false},
}

// parseTestFile parses name, a file in this package, with its comments.
func parseTestFile(t *testing.T, name string) (*token.FileSet, *ast.File) {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, name, nil, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	return fset, file
}

// fileBuildConstraint returns the //go:build constraint of file, or nil if it
// has none.
func fileBuildConstraint(t *testing.T, file *ast.File) constraint.Expr {
	t.Helper()
	for _, g := range file.Comments {
		if g.Pos() >= file.Package {
			break
		}
		for _, c := range g.List {
			if constraint.IsGoBuild(c.Text) {
				expr, err := constraint.Parse(c.Text)
				if err != nil {
					t.Fatal(err)
				}
				return expr
			}
		}
	}
	return nil
}

// declaredNativeBuild returns the value file gives the constant nativeBuild.
func declaredNativeBuild(file *ast.File) (value string, ok bool) {
	for _, d := range file.Decls {
		g, isGen := d.(*ast.GenDecl)
		if !isGen || g.Tok != token.CONST {
			continue
		}
		for _, s := range g.Specs {
			v := s.(*ast.ValueSpec)
			for i, name := range v.Names {
				if name.Name == "nativeBuild" && i < len(v.Values) {
					if id, isIdent := v.Values[i].(*ast.Ident); isIdent {
						return id.Name, true
					}
					return "", true
				}
			}
		}
	}
	return "", false
}

// TestNativeBuildMarker checks that the compiler, not build metadata, decides
// whether the binary is a native build. The two marker files must have
// complementary constraints, so that for either state of the native tag
// exactly one of them is compiled, and each must declare the value matching
// the state that selects it. Omitting either file, or dropping or widening
// either constraint, also stops one of the two builds from compiling.
func TestNativeBuildMarker(t *testing.T) {
	values := make(map[string]string)
	exprs := make(map[string]constraint.Expr)
	for _, m := range nativeMarkers {
		_, file := parseTestFile(t, m.file)
		expr := fileBuildConstraint(t, file)
		if expr == nil || expr.String() != m.constraint {
			t.Errorf("%s build constraint = %v, want %q", m.file, expr, m.constraint)
			continue
		}
		exprs[m.file] = expr
		got, ok := declaredNativeBuild(file)
		if want := strconv.FormatBool(m.value); !ok || got != want {
			t.Errorf("%s declares nativeBuild = %q (found %t), want %s", m.file, got, ok, want)
		}
		values[m.file] = got
	}
	for _, native := range []bool{false, true} {
		var selected []string
		for _, m := range nativeMarkers {
			if expr := exprs[m.file]; expr != nil && expr.Eval(func(tag string) bool { return native && tag == "native" }) {
				selected = append(selected, m.file)
			}
		}
		if len(selected) != 1 {
			t.Errorf("native tag %t selects marker files %q, want exactly one", native, selected)
			continue
		}
		if got, want := values[selected[0]], strconv.FormatBool(native); got != want {
			t.Errorf("native tag %t selects %s, which declares nativeBuild = %s", native, selected[0], got)
		}
	}
}

// TestNativeGateStateUsesTheMarker requires TestNativeGateState to classify
// the run by nativeBuild alone, and no file in this package to read the
// binary's build metadata, which a binary may lack. With it absent, an
// earlier detector took a native build for an untagged one and skipped.
func TestNativeGateStateUsesTheMarker(t *testing.T) {
	fset, file := parseTestFile(t, "native_test.go")
	var fn *ast.FuncDecl
	for _, d := range file.Decls {
		if f, ok := d.(*ast.FuncDecl); ok && f.Name.Name == "TestNativeGateState" {
			fn = f
		}
	}
	if fn == nil {
		t.Fatal("TestNativeGateState not found")
	}
	var b strings.Builder
	if err := printer.Fprint(&b, fset, fn.Body); err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(strings.Fields(b.String()), " "), "{ reportNativeGate(t, nativeBuild, requireNativeLock(os.Getenv)) }"; got != want {
		t.Errorf("TestNativeGateState body = %s, want %s", got, want)
	}

	names, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(names) == 0 {
		t.Fatal("no Go files in the package directory")
	}
	for _, name := range names {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, name, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range file.Imports {
			if imp.Path.Value == `"runtime/debug"` {
				t.Errorf("%s imports runtime/debug; native builds are recognized by nativeBuild", fset.Position(imp.Pos()))
			}
		}
	}
}

func compileRunTotal(t *testing.T, nvcc, target, dir, name, src string) (uint32, error) {
	t.Helper()
	bin, err := clusterFixtureBinary(t, nvcc, target, dir, name, src)
	if err != nil {
		return 0, err
	}

	runCtx, runCancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer runCancel()
	run := exec.CommandContext(runCtx, bin)
	out, _ := run.CombinedOutput()
	retainClusterRun(t, dir, name, run.Args, out)
	if run.ProcessState == nil {
		return classifyFixtureResult(false, fixtureTimedOut(runCtx), 0, string(out))
	}
	return classifyFixtureResult(true, fixtureTimedOut(runCtx), run.ProcessState.ExitCode(), string(out))
}

func fixtureTimedOut(ctx context.Context) bool {
	return errors.Is(ctx.Err(), context.DeadlineExceeded)
}

type clusterBuildReceipt struct {
	SourceDigest string
	BinaryDigest string
	Target       string
	Command      []string
}

func clusterDigest(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }

func clusterFixtureKey(target, src string) string {
	return clusterDigest([]byte(target + "\x00" + src))
}

// clusterFixtureBinary separates offline compilation from locked execution.
// Receipts bind retained bytes; they are not driver or compiler attestations.
func clusterFixtureBinary(t *testing.T, nvcc, target, dir, name, src string) (string, error) {
	t.Helper()
	key := clusterFixtureKey(target, src)
	if root := os.Getenv("CUDA_CLUSTER_PREBUILT_DIR"); root != "" {
		return readClusterBinary(filepath.Join(root, key), target, src)
	}
	if root := os.Getenv("CUDA_CLUSTER_ARTIFACT_DIR"); root != "" {
		dir = filepath.Join(root, key)
		if err := os.Mkdir(dir, 0o755); err != nil {
			return "", fmt.Errorf("create fresh fixture packet: %w", err)
		}
	}
	file, bin := filepath.Join(dir, "fixture.cu"), filepath.Join(dir, "fixture")
	if err := os.WriteFile(file, []byte(src), 0o644); err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, nvcc, "-arch="+target, "-o", bin, file)
	out, buildErr := build.CombinedOutput()
	command, _ := json.Marshal(build.Args)
	for path, data := range map[string][]byte{"build-command.json": command, "build-output.txt": out} {
		if err := os.WriteFile(filepath.Join(dir, path), data, 0o644); err != nil {
			return "", err
		}
	}
	if buildErr != nil {
		return "", fmt.Errorf("nvcc %s: %w: %s", target, buildErr, out)
	}
	image, err := os.ReadFile(bin)
	if err != nil {
		return "", err
	}
	receipt, _ := json.Marshal(clusterBuildReceipt{clusterDigest([]byte(src)), clusterDigest(image), target, build.Args})
	if err := os.WriteFile(filepath.Join(dir, "build.json"), receipt, 0o644); err != nil {
		return "", err
	}
	t.Logf("fixture %s source=%s binary=%s build=%s", name, clusterDigest([]byte(src)), clusterDigest(image), command)
	return bin, nil
}

func readClusterBinary(dir, target, src string) (string, error) {
	data, err := os.ReadFile(filepath.Join(dir, "build.json"))
	if err != nil {
		return "", err
	}
	var receipt clusterBuildReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return "", err
	}
	stored, err := os.ReadFile(filepath.Join(dir, "fixture.cu"))
	if err != nil {
		return "", err
	}
	bin := filepath.Join(dir, "fixture")
	image, err := os.ReadFile(bin)
	if err != nil {
		return "", err
	}
	if string(stored) != src || receipt.Target != target || receipt.SourceDigest != clusterDigest(stored) || receipt.BinaryDigest != clusterDigest(image) || len(receipt.Command) == 0 {
		return "", fmt.Errorf("prebuilt fixture source, target, or binary differs from receipt")
	}
	return bin, nil
}

func retainClusterRun(t *testing.T, fallback, name string, command []string, output []byte) {
	t.Helper()
	root := os.Getenv("CUDA_CLUSTER_RUN_DIR")
	if root == "" {
		root = fallback
	}
	dir, err := os.MkdirTemp(root, name+"-")
	if err != nil {
		t.Fatal(err)
	}
	args, _ := json.Marshal(command)
	for path, data := range map[string][]byte{"run-command.json": args, "run-output.txt": output} {
		if err := os.WriteFile(filepath.Join(dir, path), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("actual fixture command=%s output=%q retained=%s", args, output, dir)
}

func TestBuildClusterNativeFixtures(t *testing.T) {
	if os.Getenv("CUDA_CLUSTER_BUILD_ONLY") != "1" {
		t.Skip("offline fixture compilation requires CUDA_CLUSTER_BUILD_ONLY=1")
	}
	if os.Getenv("CUDA_CLUSTER_ARTIFACT_DIR") == "" || os.Getenv("CUDA_CLUSTER_PREBUILT_DIR") != "" {
		t.Fatal("offline build requires fresh CUDA_CLUSTER_ARTIFACT_DIR and no prebuilt directory")
	}
	nvcc, _ := cudatest.NVCC(t)
	target := clusterTarget()
	for _, item := range []struct{ name, source string }{
		{"healthy2", mustClusterReduction(t, target, 2, MutantNone)},
		{"healthy4", mustClusterReduction(t, target, 4, MutantNone)},
		{"local4", mustClusterReduction(t, target, 4, MutantLocalWindow)},
		{"nosync4", mustClusterReduction(t, target, 4, MutantNoSync)},
		{"oracle2", cooperativeReductionFixture(2)},
		{"oracle4", cooperativeReductionFixture(4)},
		{"geometry4", rankGeometrySource(4)},
	} {
		if _, err := clusterFixtureBinary(t, nvcc, target, t.TempDir(), item.name, item.source); err != nil {
			t.Fatal(err)
		}
	}
}

func mustClusterReduction(t *testing.T, target string, size int, mutant Mutant) string {
	t.Helper()
	source, err := ReductionFixture(target, size, mutant)
	if err != nil {
		t.Fatal(err)
	}
	return source + reductionMain(size)
}

func TestClusterPrebuiltBytesBoundToSource(t *testing.T) {
	for _, changed := range []string{"none", "source", "binary", "target", "missing"} {
		t.Run(changed, func(t *testing.T) {
			dir := t.TempDir()
			source, image := []byte("fixture source"), []byte("host-only synthetic binary")
			receipt := clusterBuildReceipt{clusterDigest(source), clusterDigest(image), "sm_121a", []string{"nvcc", "-arch=sm_121a"}}
			data, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			for name, bytes := range map[string][]byte{"fixture.cu": source, "fixture": image, "build.json": data} {
				if err := os.WriteFile(filepath.Join(dir, name), bytes, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			target := "sm_121a"
			switch changed {
			case "source":
				source = []byte("different source")
			case "binary":
				if err := os.WriteFile(filepath.Join(dir, "fixture"), []byte("changed"), 0o600); err != nil {
					t.Fatal(err)
				}
			case "target":
				target = "sm_90"
			case "missing":
				if err := os.Remove(filepath.Join(dir, "build.json")); err != nil {
					t.Fatal(err)
				}
			}
			_, err = readClusterBinary(dir, target, string(source))
			if (err != nil) != (changed != "none") {
				t.Fatalf("readClusterBinary changed=%s error=%v", changed, err)
			}
		})
	}
}

var errFixtureTimeout = errors.New("fixture did not finish")

func classifyFixtureResult(hasProcessState, timedOut bool, exitCode int, output string) (uint32, error) {
	if !hasProcessState {
		return 0, fmt.Errorf("fixture process has no exit status: %s", output)
	}
	if timedOut {
		return 0, fmt.Errorf("%w: a cluster barrier in divergent flow may hang: %s", errFixtureTimeout, output)
	}
	switch exitCode {
	case 0:
		return parseTotal(output)
	case 2, 3, 4, 5, fixtureLaunchFailureCode:
		return 0, fmt.Errorf("fixture failed at CUDA stage %d: %s", exitCode, output)
	default:
		return 0, fmt.Errorf("fixture exited %d: %s", exitCode, output)
	}
}

func TestClassifyFixtureResult(t *testing.T) {
	tests := []struct {
		name        string
		hasState    bool
		timedOut    bool
		exitCode    int
		output      string
		want        uint32
		wantErr     bool
		wantTimeout bool
		wantErrText string
	}{
		{name: "success total", hasState: true, output: "total 628", want: 628},
		{name: "success without total", hasState: true, wantErr: true},
		{name: "cuda exit 2", hasState: true, exitCode: 2, output: "total 628", wantErr: true, wantErrText: "CUDA stage 2"},
		{name: "cuda exit 3", hasState: true, exitCode: 3, output: "total 628", wantErr: true, wantErrText: "CUDA stage 3"},
		{name: "cuda exit 4", hasState: true, exitCode: 4, output: "total 628", wantErr: true, wantErrText: "CUDA stage 4"},
		{name: "cuda exit 5", hasState: true, exitCode: 5, output: "total 628", wantErr: true, wantErrText: "CUDA stage 5"},
		{name: "launch error", hasState: true, exitCode: 6, output: "total 628", wantErr: true, wantErrText: "CUDA stage 6"},
		{name: "ordinary error", hasState: true, exitCode: 1, output: "total 628", wantErr: true, wantErrText: "fixture exited 1"},
		{name: "signal", hasState: true, exitCode: -1, output: "total 628", wantErr: true, wantErrText: "fixture exited -1"},
		{name: "no process state", output: "total 628", wantErr: true, wantErrText: "no exit status"},
		{name: "timeout", hasState: true, timedOut: true, exitCode: -1, output: "total 628", wantErr: true, wantTimeout: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := classifyFixtureResult(tc.hasState, tc.timedOut, tc.exitCode, tc.output)
			if got != tc.want || (err != nil) != tc.wantErr || errors.Is(err, errFixtureTimeout) != tc.wantTimeout {
				t.Fatalf("classifyFixtureResult = (%d, %v), want (%d, err=%v, timeout=%v)", got, err, tc.want, tc.wantErr, tc.wantTimeout)
			}
			if tc.wantErrText != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErrText)) {
				t.Fatalf("classifyFixtureResult error = %v, want substring %q", err, tc.wantErrText)
			}
		})
	}
}

func parseTotal(out string) (uint32, error) {
	fields := strings.Fields(out)
	for i := 0; i+1 < len(fields); i++ {
		if fields[i] != "total" {
			continue
		}
		n, err := strconv.ParseUint(fields[i+1], 10, 32)
		if err != nil {
			return 0, fmt.Errorf("parse total: %w", err)
		}
		return uint32(n), nil
	}
	return 0, fmt.Errorf("fixture output has no total: %q", out)
}

func TestParseTotal(t *testing.T) {
	for _, tc := range []struct {
		out  string
		want uint32
		bad  bool
	}{
		{"total 628\n", 628, false}, {"", 0, true}, {"total -1", 0, true}, {"total 4294967296", 0, true},
	} {
		got, err := parseTotal(tc.out)
		if (err != nil) != tc.bad || (!tc.bad && got != tc.want) {
			t.Errorf("parseTotal(%q) = %d, %v", tc.out, got, err)
		}
	}
}

func TestNativeGateRejectsPresenceOnlyLockBeforeToolLookup(t *testing.T) {
	env := func(k string) string {
		switch k {
		case "CUDA_CLUSTER_NATIVE":
			return "1"
		case "GPU_LOCK_HELD":
			return "1"
		}
		return ""
	}
	if err := requireNativeLock(env); err == nil {
		t.Fatal("native gate accepted GPU_LOCK_HELD=1")
	}
}

func TestNativeGateAcceptsGenuineInheritedLockOnHost(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(context.Background(), cudalock.WithDir(dir), cudalock.WithLabel("cudacluster-host-test"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := h.Release(); err != nil {
			t.Errorf("release test lock: %v", err)
		}
	})
	env := func(key string) string {
		switch key {
		case "CUDA_CLUSTER_NATIVE":
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return dir
		default:
			return ""
		}
	}
	if err := requireNativeLock(env); err != nil {
		t.Fatalf("requireNativeLock with inherited lock: %v", err)
	}
}

// TestNativeClusterAllToOneReduction is lane 56 slice 6.
func TestNativeClusterAllToOneReduction(t *testing.T) {
	const size = 2
	got, err := runReduction(t, size, MutantNone)
	if err != nil {
		t.Fatal(err)
	}
	if want := ExpectedReduction(size); got != want {
		t.Fatalf("reduction = %d, want %d", got, want)
	}
}

// TestNativeClusterRemoteWindowIsRemote is the test lane 56 says must not be
// weakened, and TestNativeControlLocalWindowMutantFails is what proves it can
// fail. Running them apart would leave the first one unfalsifiable.
func TestNativeClusterRemoteWindowIsRemote(t *testing.T) {
	const size = 4
	got, err := runReduction(t, size, MutantNone)
	if err != nil {
		t.Fatal(err)
	}
	want := ExpectedReduction(size)
	if got != want {
		t.Fatalf("reduction = %d, want %d: the per-rank seeds did not all arrive", got, want)
	}
	// The local-window total is the rank-0 seed repeated. If the two totals
	// were equal the test could not distinguish them, so assert the fixture
	// is capable of the distinction before trusting the result above.
	if local := uint32(size) * 7; local == want {
		t.Fatalf("fixture cannot distinguish remote from local: both totals are %d", want)
	}
}

func TestNativeControlLocalWindowMutantFails(t *testing.T) {
	const size = 4
	got, err := runReduction(t, size, MutantLocalWindow)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkLocalWindowTotal(size, got); err != nil {
		t.Fatal(err)
	}
}

// TestNativeControlNoSyncMutantIsNotDeterministicallyCorrect asserts the
// end-to-end slice exercises a real cross-CTA dependency. A mutant that drops
// the barrier and still gives the right answer every time means the CTAs were
// never actually depending on each other.
//
// It is deliberately tolerant of a hang: removing a barrier can deadlock as
// easily as it can race, and both outcomes falsify determinism. What it will
// not accept is repeated silent correctness.
func TestNativeControlNoSyncMutantIsNotDeterministicallyCorrect(t *testing.T) {
	const size = 4
	want := ExpectedReduction(size)
	const reps = 5
	runs := make([]noSyncRun, 0, reps)
	for i := 0; i < reps; i++ {
		got, err := runReduction(t, size, MutantNoSync)
		runs = append(runs, noSyncRun{total: got, err: err})
		if errors.Is(err, errFixtureTimeout) {
			t.Logf("rep %d: %v (a failure here also falsifies determinism)", i, err)
			break
		}
		if err != nil {
			break
		}
	}
	if err := checkNoSyncControl(want, runs); err != nil {
		t.Fatal(err)
	}
}

type noSyncRun struct {
	total uint32
	err   error
}

func checkNoSyncControl(want uint32, runs []noSyncRun) error {
	if len(runs) == 0 {
		return fmt.Errorf("no-sync control produced no runs")
	}
	correct := 0
	for i, run := range runs {
		if run.err != nil {
			if errors.Is(run.err, errFixtureTimeout) {
				return nil
			}
			return fmt.Errorf("rep %d failed outside runtime timeout: %w", i, run.err)
		}
		if run.total == want {
			correct++
		}
	}
	if correct == len(runs) {
		return fmt.Errorf("no-sync mutant produced the correct total %d in all %d runs: the reduction is not exercising a cross-CTA dependency", want, correct)
	}
	return nil
}

func checkLocalWindowTotal(size int, got uint32) error {
	want := uint32(size) * 7
	if got != want {
		return fmt.Errorf("local-window mutant total = %d, want %d", got, want)
	}
	return nil
}

func TestCompareOracleTotal(t *testing.T) {
	oracleErr := errors.New("oracle failed")
	for _, tc := range []struct {
		name      string
		emitted   uint32
		oracle    uint32
		oracleErr error
		wantErr   bool
	}{
		{name: "equal", emitted: 42, oracle: 42},
		{name: "different", emitted: 42, oracle: 43, wantErr: true},
		{name: "oracle failed", emitted: 42, oracleErr: oracleErr, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := compareOracleTotal(tc.emitted, tc.oracle, tc.oracleErr)
			if (err != nil) != tc.wantErr {
				t.Fatalf("compareOracleTotal error = %v, wantErr=%t", err, tc.wantErr)
			}
			if tc.name == "oracle failed" && !errors.Is(err, oracleErr) {
				t.Fatalf("oracle error = %v, want wrapped source error", err)
			}
		})
	}
}

func TestCheckNoSyncControl(t *testing.T) {
	timeoutErr := fmt.Errorf("%w: fixture", errFixtureTimeout)
	ordinaryErr := errors.New("compiler failed")
	for _, tc := range []struct {
		name    string
		runs    []noSyncRun
		wantErr bool
	}{
		{name: "one wrong total", runs: []noSyncRun{{total: 1}, {total: 2}}},
		{name: "all correct", runs: []noSyncRun{{total: 9}, {total: 9}}, wantErr: true},
		{name: "timeout falsifies determinism", runs: []noSyncRun{{err: timeoutErr}}},
		{name: "other error is fatal", runs: []noSyncRun{{err: ordinaryErr}}, wantErr: true},
		{name: "no runs", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkNoSyncControl(9, tc.runs)
			if (err != nil) != tc.wantErr {
				t.Fatalf("checkNoSyncControl error = %v, wantErr=%t", err, tc.wantErr)
			}
			if tc.name == "other error is fatal" && !errors.Is(err, ordinaryErr) {
				t.Fatalf("ordinary error = %v, want wrapped source error", err)
			}
		})
	}
}

func TestCheckLocalWindowTotal(t *testing.T) {
	if err := checkLocalWindowTotal(4, 28); err != nil {
		t.Fatalf("correct local total: %v", err)
	}
	if err := checkLocalWindowTotal(4, 628); err == nil {
		t.Fatal("remote total was accepted as the local-window mutant total")
	}
}

func TestFixtureTimedOut(t *testing.T) {
	if fixtureTimedOut(context.Background()) {
		t.Fatal("background context reported timeout")
	}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	if !fixtureTimedOut(ctx) {
		t.Fatal("expired run deadline was not recorded")
	}
}

// TestControl_NoSimulationWithoutHardware asserts that with the native gate
// closed no code path in the native harness produces a reduction total or a
// geometry verdict. Each returns ErrHardwarePending, and the runner skips
// rather than returning. The harness has no simulation path; this test exists
// so that adding one fails. It reads no process environment.
func TestControl_NoSimulationWithoutHardware(t *testing.T) {
	closed := func(string) string { return "" }
	for _, size := range []int{2, 4} {
		for _, mut := range []Mutant{MutantNone, MutantLocalWindow, MutantNoSync} {
			name := fmt.Sprintf("healthy/%d", size)
			if mut != MutantNone {
				name = fmt.Sprintf("%s/%d", mut, size)
			}
			total, err := reduceOnDevice(t, closed, size, mut)
			if total != 0 || !errors.Is(err, ErrHardwarePending) {
				t.Errorf("reduceOnDevice(%s) with closed gate = (%d, %v), want (0, ErrHardwarePending)", name, total, err)
			}
			var sub *testing.T
			t.Run("runner/"+name, func(t *testing.T) {
				sub = t
				total, err := runReductionEnv(t, closed, size, mut)
				t.Errorf("runReductionEnv with closed gate returned (%d, %v), want a pending skip", total, err)
			})
			if sub != nil && !sub.Skipped() {
				t.Errorf("runReductionEnv(%s) with closed gate did not skip", name)
			}
		}
	}
	if err := runRankGeometryEnv(t, closed, 4); !errors.Is(err, ErrHardwarePending) {
		t.Errorf("runRankGeometryEnv with closed gate = %v, want ErrHardwarePending", err)
	}

	src, err := ReductionFixture(clusterTarget(), 4, MutantNone)
	if err != nil {
		t.Fatal(err)
	}
	if src == "" {
		t.Fatal("ReductionFixture returned no source")
	}
	// The fixture is CUDA source text, not a result. Nothing in this package
	// converts it into a number without nvcc and a device.
	if err := CheckDSMEM(clusterTarget()); err == nil {
		t.Fatal("CheckDSMEM reported DSMEM established with no instrument having probed it")
	}
}

// TestCluster_RankGeometry verifies that BlockRank and ClusterSize agree with
// launched cluster dimensions.
func TestCluster_RankGeometry(t *testing.T) {
	err := runRankGeometry(t, 4)
	skipClosedGate(t, err)
	if err != nil {
		t.Fatalf("cluster size 4: %v", err)
	}
}

const rankGeometryTemplate = `#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned ptx_block_rank() {
    unsigned r;
    asm("{{BLOCK_RANK}}" : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned ptx_cluster_size() {
    unsigned s;
    asm("{{CLUSTER_SIZE}}" : "=r"(s));
    return s;
}

__device__ __forceinline__ unsigned ptx_cluster_ctaid_x() {
    unsigned id; asm("{{CTAID_X}}" : "=r"(id)); return id;
}
__device__ __forceinline__ unsigned ptx_cluster_ctaid_y() {
    unsigned id; asm("{{CTAID_Y}}" : "=r"(id)); return id;
}
__device__ __forceinline__ unsigned ptx_cluster_nctaid_x() {
    unsigned n; asm("{{NCTAID_X}}" : "=r"(n)); return n;
}
__device__ __forceinline__ unsigned ptx_cluster_nctaid_y() {
    unsigned n; asm("{{NCTAID_Y}}" : "=r"(n)); return n;
}

__global__ void __cluster_dims__({{CLUSTER_X}}, {{CLUSTER_Y}}, 1) rank_geometry_kernel(unsigned *ranks, unsigned *err) {
    if (threadIdx.x == 0) {
        cg::cluster_group cluster = cg::this_cluster();
        unsigned cg_rank = cluster.block_rank();
        unsigned cg_size = cluster.num_blocks();
        unsigned ptx_rank = ptx_block_rank();
        unsigned ptx_size = ptx_cluster_size();
		unsigned cta_x = ptx_cluster_ctaid_x();
		unsigned cta_y = ptx_cluster_ctaid_y();
		unsigned ncta_x = ptx_cluster_nctaid_x();
		unsigned ncta_y = ptx_cluster_nctaid_y();

        if (cg_size != {{EXPECTED_SIZE}} || ptx_size != {{EXPECTED_SIZE}}) {
            atomicExch(err, 1);
            return;
        }
		if (ncta_x != {{CLUSTER_X}} || ncta_y != {{CLUSTER_Y}} || cta_x >= ncta_x || cta_y >= ncta_y) { atomicExch(err, 4); return; }
        if (cg_rank != ptx_rank) {
            atomicExch(err, 2);
            return;
        }
        if (ptx_rank != cta_x + cta_y * ncta_x) {
            atomicExch(err, 3);
            return;
        }
        atomicAdd(&ranks[ptx_rank], 1);
    }
}

int main() {
    const int cluster_x = {{CLUSTER_X}};
    const int cluster_y = {{CLUSTER_Y}};
    const int cluster_size = {{EXPECTED_SIZE}};
    unsigned *d_ranks = NULL;
    unsigned *d_err = NULL;

    if (cudaMalloc(&d_ranks, sizeof(unsigned) * cluster_size) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_ranks failed\n");
        return 2;
    }
    if (cudaMalloc(&d_err, sizeof(unsigned)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_err failed\n");
        return 2;
    }
    if (cudaMemset(d_ranks, 0, sizeof(unsigned) * cluster_size) != cudaSuccess) {
        fprintf(stderr, "cudaMemset d_ranks failed\n");
        return 3;
    }
    if (cudaMemset(d_err, 0, sizeof(unsigned)) != cudaSuccess) {
        fprintf(stderr, "cudaMemset d_err failed\n");
        return 3;
    }

    rank_geometry_kernel<<<dim3(cluster_x, cluster_y, 1), 32>>>(d_ranks, d_err);
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        fprintf(stderr, "rank_geometry_kernel launch failed: %s\n", cudaGetErrorString(launch));
        return 8;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        return 4;
    }

    unsigned h_err = 0;
    if (cudaMemcpy(&h_err, d_err, sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_err failed\n");
        return 5;
    }
    if (h_err != 0) {
        fprintf(stderr, "kernel reported geometry mismatch code %u\n", h_err);
        return 6;
    }

    unsigned h_ranks[cluster_size];
    if (cudaMemcpy(h_ranks, d_ranks, sizeof(h_ranks), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_ranks failed\n");
        return 5;
    }
    for (int i = 0; i < cluster_size; i++) {
        if (h_ranks[i] != 1) {
            fprintf(stderr, "rank %d observed %u times, want 1\n", i, h_ranks[i]);
            return 7;
        }
    }

    cudaFree(d_ranks);
    cudaFree(d_err);
    return 0;
}
`

// runRankGeometry compiles and runs a fixture verifying that BlockRank and
// ClusterSize agree with the launched cluster dimensions, gated by the process
// environment.
func runRankGeometry(t *testing.T, clusterSize int) error {
	t.Helper()
	return runRankGeometryEnv(t, os.Getenv, clusterSize)
}

// runRankGeometryEnv is runRankGeometry gated by getenv. The native gate is
// checked first, and a closed gate returns errNativeGateClosed.
func runRankGeometryEnv(t *testing.T, getenv func(string) string, clusterSize int) error {
	t.Helper()
	if err := requireNativeLock(getenv); err != nil {
		return err
	}
	target := clusterTarget()
	if err := Check(target); err != nil {
		return fmt.Errorf("check %s: %w", target, err)
	}
	var nvcc string
	if os.Getenv("CUDA_CLUSTER_PREBUILT_DIR") == "" {
		nvcc, _ = cudatest.NVCC(t)
	}

	src := rankGeometrySource(clusterSize)

	dir := t.TempDir()
	bin, err := clusterFixtureBinary(t, nvcc, target, dir, "geom", src)
	if err != nil {
		return err
	}

	runCtx, runCancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer runCancel()
	run := exec.CommandContext(runCtx, bin)
	out, err := run.CombinedOutput()
	retainClusterRun(t, dir, "geom", run.Args, out)
	if runCtx.Err() != nil {
		return fmt.Errorf("fixture timed out: %s", out)
	}
	if err != nil {
		return fmt.Errorf("rank geometry fixture failed: %w: %s", err, out)
	}
	return nil
}

func rankGeometrySource(clusterSize int) string {
	if clusterSize < 4 || clusterSize > 8 || clusterSize%2 != 0 {
		return ""
	}
	clusterX := 2
	clusterY := clusterSize / clusterX
	src := strings.ReplaceAll(rankGeometryTemplate, "{{BLOCK_RANK}}", strings.ReplaceAll(EmitBlockRank("%0"), "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{CLUSTER_SIZE}}", strings.ReplaceAll(EmitClusterSize("%0"), "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{EXPECTED_SIZE}}", strconv.Itoa(clusterSize))
	cta, _ := EmitClusterCTAID("%0", DimX)
	ctaY, _ := EmitClusterCTAID("%0", DimY)
	nctaX, _ := EmitClusterNCTAID("%0", DimX)
	nctaY, _ := EmitClusterNCTAID("%0", DimY)
	src = strings.ReplaceAll(src, "{{CTAID_X}}", strings.ReplaceAll(cta, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{CTAID_Y}}", strings.ReplaceAll(ctaY, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{NCTAID_X}}", strings.ReplaceAll(nctaX, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{NCTAID_Y}}", strings.ReplaceAll(nctaY, "%cluster", "%%cluster"))
	src = strings.ReplaceAll(src, "{{CLUSTER_Y}}", strconv.Itoa(clusterY))
	src = strings.ReplaceAll(src, "{{CLUSTER_X}}", strconv.Itoa(clusterX))
	return src
}

// TestRankGeometrySourceUsesEmitters pins each accessor in the geometry
// fixture to the emitter for its own register and axis, so that reading y from
// the x register, or swapping two accessors, fails on the host.
func TestRankGeometrySourceUsesEmitters(t *testing.T) {
	src := rankGeometrySource(4)
	escape := func(ptx string, err error) string {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return strings.ReplaceAll(ptx, "%cluster", "%%cluster")
	}
	for _, tc := range []struct {
		accessor string
		emitted  string
	}{
		{"ptx_block_rank", escape(EmitBlockRank("%0"), nil)},
		{"ptx_cluster_size", escape(EmitClusterSize("%0"), nil)},
		{"ptx_cluster_ctaid_x", escape(EmitClusterCTAID("%0", DimX))},
		{"ptx_cluster_ctaid_y", escape(EmitClusterCTAID("%0", DimY))},
		{"ptx_cluster_nctaid_x", escape(EmitClusterNCTAID("%0", DimX))},
		{"ptx_cluster_nctaid_y", escape(EmitClusterNCTAID("%0", DimY))},
	} {
		_, body, ok := strings.Cut(src, "unsigned "+tc.accessor+"() {")
		if !ok {
			t.Errorf("rank geometry source has no accessor %s", tc.accessor)
			continue
		}
		body, _, _ = strings.Cut(body, "\n}")
		if !strings.Contains(body, `asm("`+tc.emitted+`"`) {
			t.Errorf("%s body %q does not read %q", tc.accessor, body, tc.emitted)
		}
	}
	if strings.Contains(src, "{{") || strings.Contains(src, "}}") {
		t.Error("rank geometry source has an unfilled placeholder")
	}
}

// TestNativeGateClosedIsPending requires a closed native gate to report
// ErrHardwarePending: nothing has examined a device, which is not the same as
// a refusal or a configuration error.
func TestNativeGateClosedIsPending(t *testing.T) {
	closed := func(string) string { return "" }
	if err := requireNativeLock(closed); !errors.Is(err, ErrHardwarePending) {
		t.Fatalf("requireNativeLock with closed gate = %v, want ErrHardwarePending", err)
	}
}

// TestNativeHarnessEvaluatesTheGate is the positive control for the pending
// skip: with the gate open and no inherited lock, the harness must reach the
// lock check and fail, not report pending or skip.
func TestNativeHarnessEvaluatesTheGate(t *testing.T) {
	open := func(key string) string {
		if key == "CUDA_CLUSTER_NATIVE" {
			return "1"
		}
		return ""
	}
	total, err := reduceOnDevice(t, open, 4, MutantNone)
	if total != 0 || err == nil || errors.Is(err, ErrHardwarePending) || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Errorf("reduceOnDevice with open gate and no lock = (%d, %v), want (0, cudalock.ErrNotInherited)", total, err)
	}
	if err := runRankGeometryEnv(t, open, 4); err == nil || errors.Is(err, ErrHardwarePending) || !errors.Is(err, cudalock.ErrNotInherited) {
		t.Errorf("runRankGeometryEnv with open gate and no lock = %v, want cudalock.ErrNotInherited", err)
	}
	var sub *testing.T
	t.Run("runner", func(t *testing.T) {
		sub = t
		if _, err := runReductionEnv(t, open, 4, MutantNone); !errors.Is(err, cudalock.ErrNotInherited) {
			t.Errorf("runReductionEnv with open gate and no lock = %v, want cudalock.ErrNotInherited", err)
		}
	})
	if sub != nil && sub.Skipped() {
		t.Error("runReductionEnv skipped with the gate open")
	}
}

func TestGeometryEmittersHaveLiteralRegisterSpellings(t *testing.T) {
	if got := EmitClusterSize("%r1"); got != "mov.u32 %r1, %cluster_nctarank;" {
		t.Fatalf("EmitClusterSize = %q", got)
	}
	for _, tc := range []struct {
		name string
		fn   func(string, Dim) (string, error)
		reg  string
	}{
		{"CTAID", EmitClusterCTAID, "%cluster_ctaid."},
		{"NCTAID", EmitClusterNCTAID, "%cluster_nctaid."},
	} {
		for _, axis := range []struct {
			dim  Dim
			name string
		}{{DimX, "x"}, {DimY, "y"}, {DimZ, "z"}} {
			got, err := tc.fn("%r7", axis.dim)
			if err != nil {
				t.Fatal(err)
			}
			want := "mov.u32 %r7, " + tc.reg + axis.name + ";"
			if got != want {
				t.Errorf("EmitCluster%s(%s) = %q, want %q", tc.name, axis.name, got, want)
			}
		}
	}
}

func TestExpectedReduction(t *testing.T) {
	for _, tc := range []struct {
		size int
		want uint32
	}{{2, 114}, {4, 628}, {8, 2856}} {
		if got := ExpectedReduction(tc.size); got != tc.want {
			t.Errorf("ExpectedReduction(%d) = %d, want %d", tc.size, got, tc.want)
		}
	}
}

func TestRankGeometrySourceUsesRequestedClusterShape(t *testing.T) {
	src := rankGeometrySource(4)
	for _, want := range []string{
		"__cluster_dims__(2, 2, 1)",
		"dim3(cluster_x, cluster_y, 1)",
		"cudaGetLastError()",
		"cta_x + cta_y * ncta_x",
		"ptx_cluster_ctaid_y()",
		"unsigned cta_y = ptx_cluster_ctaid_y();",
		"if (ncta_x != 2 || ncta_y != 2 || cta_x >= ncta_x || cta_y >= ncta_y)",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("rank geometry source lacks %q", want)
		}
	}
	if strings.Contains(src, `launch failed: %s\\n`) {
		t.Fatal("rank geometry launch error prints a literal backslash-n")
	}
	if rankGeometrySource(6) == "" || !strings.Contains(rankGeometrySource(6), "const int cluster_y = 3;") {
		t.Fatal("rank geometry source ignored the requested size")
	}
	if rankGeometrySource(3) != "" {
		t.Fatal("rank geometry source accepted a shape it cannot represent")
	}
}
