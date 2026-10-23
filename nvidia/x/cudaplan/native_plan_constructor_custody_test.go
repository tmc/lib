package cudaplan_test

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
	"unicode"
)

// setupNativePlanOwners runs admission before any device initialization. The
// child lease and device belong to the same chain as every later acquisition.
func setupNativePlanOwners(owners *workflowNativeOwners, admit func() error, acquire, open func() (func() error, error)) error {
	if err := admit(); err != nil {
		return fmt.Errorf("inherit GPU lease: %w", err)
	}
	release, err := acquire()
	if release != nil {
		owners.add(workflowLease, "GPU lease", release)
	}
	retainNativePlanError(owners, err)
	if err != nil {
		return fmt.Errorf("acquire GPU lease: %w", err)
	}
	closeDevice, err := open()
	if closeDevice != nil {
		owners.add(workflowDevice, "device", closeDevice)
	}
	retainNativePlanError(owners, err)
	if err != nil {
		return fmt.Errorf("open device: %w", err)
	}
	return nil
}

func retainNativePlanError(owners *workflowNativeOwners, err error) {
	var owner io.Closer
	if errors.As(err, &owner) {
		owners.add(workflowCompletion, "acquisition error owner", owner.Close)
	}
}

func closeNativePlanOwners(owners *workflowNativeOwners) error {
	if err := owners.close(); err != nil {
		workflowRetainedOwners.Lock()
		workflowRetainedOwners.owners = append(workflowRetainedOwners.owners, owners)
		workflowRetainedOwners.Unlock()
		return err
	}
	return nil
}

type nativePlanClientError struct {
	cause error
	close func() error
}

func (e *nativePlanClientError) Error() string { return e.cause.Error() }
func (e *nativePlanClientError) Unwrap() error { return e.cause }
func (e *nativePlanClientError) Close() error  { return e.close() }

func TestNativePlanClientSetup(t *testing.T) {
	failed := errors.New("injected failure")
	for _, stage := range []string{"none", "admit", "acquire", "open"} {
		t.Run(stage, func(t *testing.T) {
			var calls []string
			owners := &workflowNativeOwners{}
			step := func(name string) error {
				calls = append(calls, name)
				if stage == name {
					return failed
				}
				return nil
			}
			release := func(name string) func() error { return func() error { calls = append(calls, name); return nil } }
			err := setupNativePlanOwners(owners,
				func() error { return step("admit") },
				func() (func() error, error) {
					if err := step("acquire"); err != nil {
						return nil, &nativePlanClientError{err, release("pending lease")}
					}
					return release("lease"), nil
				},
				func() (func() error, error) {
					if err := step("open"); err != nil {
						return nil, err
					}
					return release("device"), nil
				},
			)
			if (stage == "none") != (err == nil) || err != nil && !errors.Is(err, failed) {
				t.Fatalf("setup = %v", err)
			}
			if err := owners.close(); err != nil {
				t.Fatal(err)
			}
			want := map[string][]string{
				"none":    {"admit", "acquire", "open", "device", "lease"},
				"admit":   {"admit"},
				"acquire": {"admit", "acquire", "pending lease"},
				"open":    {"admit", "acquire", "open", "lease"},
			}[stage]
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("calls = %v, want %v", calls, want)
			}
			n := len(calls)
			if err := owners.close(); err != nil || len(calls) != n {
				t.Fatalf("repeated close = %v, calls %v", err, calls)
			}
		})
	}
}

func TestNativePlanClientPartialStream(t *testing.T) {
	cause := errors.New("restore failed")
	for _, nonnil := range []bool{false, true} {
		t.Run(fmt.Sprint(nonnil), func(t *testing.T) {
			var calls []string
			owners := &workflowNativeOwners{}
			owners.add(workflowLease, "lease", func() error { calls = append(calls, "lease"); return nil })
			owners.add(workflowDevice, "device", func() error { calls = append(calls, "device"); return nil })
			create := func() (*custodyStream, error) {
				if !nonnil {
					return nil, cause
				}
				return &custodyStream{calls: &calls, closeErr: cause}, cause
			}
			stream, err := create()
			if stream != nil {
				owners.ownStream(stream)
			}
			if !errors.Is(err, cause) {
				t.Fatalf("acquisition = %v", err)
			}
			err = owners.close()
			want := []string{"device", "lease"}
			if nonnil {
				want = []string{"drain", "stream"}
				if !errors.Is(err, cause) {
					t.Fatalf("close = %v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("calls = %v, want %v", calls, want)
			}
			if nonnil {
				stream.closeErr = nil
				calls = nil
				if err := owners.close(); err != nil {
					t.Fatal(err)
				}
				if want := []string{"stream", "device", "lease"}; !reflect.DeepEqual(calls, want) {
					t.Fatalf("retry = %v, want %v", calls, want)
				}
			}
		})
	}
}

func TestNativePlanClientFailureOrder(t *testing.T) {
	failed := errors.New("injected cleanup failure")
	for _, stage := range []string{"none", "drain", "error owner", "plan", "buffer", "stream", "device", "lease"} {
		t.Run(stage, func(t *testing.T) {
			var calls []string
			refuse := true
			release := func(name string) func() error {
				return func() error {
					calls = append(calls, name)
					if refuse && name == stage {
						return failed
					}
					return nil
				}
			}
			owners := &workflowNativeOwners{drain: release("drain")}
			owners.add(workflowLease, "lease", release("lease"))
			owners.add(workflowDevice, "device", release("device"))
			owners.add(workflowStream, "stream", release("stream"))
			owners.add(workflowBuffer, "buffer", release("buffer"))
			owners.add(workflowPlan, "plan", release("plan"))
			retainNativePlanError(owners, fmt.Errorf("wrapped: %w", &nativePlanClientError{failed, release("error owner")}))
			err := closeNativePlanOwners(owners)
			order := []string{"drain", "error owner", "plan", "buffer", "stream", "device", "lease"}
			stop := len(order)
			for i, name := range order {
				if stage == name {
					stop = i + 1
				}
			}
			if !reflect.DeepEqual(calls, order[:stop]) {
				t.Fatalf("calls = %v, want %v", calls, order[:stop])
			}
			if stage == "none" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				if !errors.Is(err, failed) {
					t.Fatalf("close = %v", err)
				}
				workflowRetainedOwners.Lock()
				found := false
				for _, retained := range workflowRetainedOwners.owners {
					found = found || retained == owners
				}
				workflowRetainedOwners.Unlock()
				if !found {
					t.Fatal("failed owner chain was not retained")
				}
			}
			calls = nil
			refuse = false
			if err := owners.close(); err != nil {
				t.Fatal(err)
			}
			var want []string
			if stage != "none" {
				want = order[stop-1:]
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("retry = %v, want %v", calls, want)
			}
			calls = nil
			if err := owners.close(); err != nil || len(calls) != 0 {
				t.Fatalf("repeated close = %v, calls %v", err, calls)
			}
		})
	}
}

// The guard binds the CPU owner controls to the real native acquisition route.
func TestNativePlanClientRouting(t *testing.T) {
	read := func(path, name string) string {
		t.Helper()
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == name {
				var b bytes.Buffer
				if err := format.Node(&b, token.NewFileSet(), fn); err != nil {
					t.Fatal(err)
				}
				return strings.Map(func(r rune) rune {
					if unicode.IsSpace(r) {
						return -1
					}
					return r
				}, b.String())
			}
		}
		t.Fatalf("missing %s in %s", name, path)
		return ""
	}
	sequence := func(body string, parts ...string) {
		t.Helper()
		for _, part := range parts {
			i := strings.Index(body, part)
			if i < 0 {
				t.Fatalf("missing ordered route %q", part)
			}
			body = body[i+len(part):]
		}
	}
	body := read("plan_test.go", "TestNativeMatMulReference")
	sequence(body, `ifos.Getenv("CUDA_COMPILER_TEST")!="1"`, `dev,owners:=requireNativePlanDevice(t)`, `retainNativePlanError(owners,err)`, `iferr!=nil{t.Fatal(err)}`,
		`stream,err:=dev.NewStream()`, `ifstream!=nil{owners.ownStream(stream)}`, `check(err)`,
		`nativePlan,err:=cudaplan.Compile(snap,cudaplan.WithDevice(dev))`, `ifnativePlan!=nil{owners.add(workflowPlan,"nativeplan",nativePlan.Close)}`, `check(err)`,
		`buf,err:=cudamem.Alloc[float32](dev,n)`, `ifbuf.Ptr()!=0{owners.add(workflowBuffer,"matrixbuffer",buf.Free)}`, `check(err)`,
		`nativeComp,err:=nativePlan.Execute(stream,nativeBindings)`, `owners.retainCompletion(nativeComp)`, `check(err)`)
	native := body[strings.Index(body, "dev,owners:="):]
	if strings.Contains(native, "defer") || strings.Contains(native, "_=") {
		t.Fatal("independent cleanup or discarded native result")
	}
	gate := read("native_gate_test.go", "requireNativePlanDevice")
	sequence(gate, `t.Cleanup(func(){iferr:=closeNativePlanOwners(owners);err!=nil{t.Error(err)}})`, `setupNativePlanOwners(owners,`, `cudalock.Inherited(os.Getenv)`, `cudalock.Acquire(`, `cuda.NewDevice(0)`)
	if strings.Contains(gate, "cudatest.") {
		t.Fatal("independent cudatest lease cleanup")
	}
	setup := read("native_plan_constructor_custody_test.go", "setupNativePlanOwners")
	sequence(setup, `iferr:=admit();err!=nil`, `release,err:=acquire()`, `owners.add(workflowLease,`, `retainNativePlanError(owners,err)`, `iferr!=nil`, `closeDevice,err:=open()`, `owners.add(workflowDevice,`)
	// Keep the source files readable directly by the future 0-vendor selector.
	if _, err := os.Stat("agreement_integration_test.go"); err != nil {
		t.Fatal(err)
	}
}
