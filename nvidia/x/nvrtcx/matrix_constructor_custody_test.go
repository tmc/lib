package nvrtcx_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode"
)

const (
	matrixErrorOwner = iota
	matrixBuffer
	matrixModule
	matrixStreamOwner
	matrixDevice
	matrixLease
)

// matrixOwners drains once before cleanup and stops at the first failure.
// Successful entries are removed so an explicit retry resumes at the failure.
// Device and child lease remain live until every dependent is released.
type matrixOwners struct {
	drain     func() error
	quiesced  bool
	resources [6][]func() error
}

var retainedMatrixOwners struct {
	sync.Mutex
	owners []*matrixOwners
}

func newMatrixOwners(t *testing.T) *matrixOwners {
	owners := &matrixOwners{}
	t.Cleanup(func() {
		if err := closeMatrixOwners(owners); err != nil {
			t.Error(err)
		}
	})
	return owners
}

func (o *matrixOwners) add(kind int, close func() error) {
	o.resources[kind] = append(o.resources[kind], close)
}
func (o *matrixOwners) addError(err error) {
	var owner io.Closer
	if errors.As(err, &owner) {
		o.add(matrixErrorOwner, owner.Close)
	}
}

func (o *matrixOwners) ownStream(stream interface {
	Synchronize(context.Context) error
	Close() error
}) {
	o.drain = func() error {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		return stream.Synchronize(ctx)
	}
	o.add(matrixStreamOwner, stream.Close)
}

func (o *matrixOwners) close() error {
	if !o.quiesced && o.drain != nil {
		if err := o.drain(); err != nil {
			return fmt.Errorf("drain matrix stream: %w", err)
		}
		o.quiesced = true
	}
	for kind := range o.resources {
		for len(o.resources[kind]) > 0 {
			n := len(o.resources[kind]) - 1
			if err := o.resources[kind][n](); err != nil {
				return fmt.Errorf("close matrix resource %d: %w", kind, err)
			}
			o.resources[kind] = o.resources[kind][:n]
		}
	}
	return nil
}

// matrixAllocationClose remembers a successful native free inside the context
// callback, before run can append a restoration error. A retry still qualifies
// context entry/restoration, but never frees successfully released storage twice.
func matrixAllocationClose(run func(func() error) error, free func() error) func() error {
	freed := false
	return func() error {
		return run(func() error {
			if freed {
				return nil
			}
			if err := free(); err != nil {
				return err
			}
			freed = true
			return nil
		})
	}
}

func closeMatrixOwners(owners *matrixOwners) error {
	if err := owners.close(); err != nil {
		retainedMatrixOwners.Lock()
		retainedMatrixOwners.owners = append(retainedMatrixOwners.owners, owners)
		retainedMatrixOwners.Unlock()
		return err
	}
	return nil
}

func setupMatrixLease(owners *matrixOwners, admit func() error, acquire func() (func() error, error)) error {
	if err := admit(); err != nil {
		return fmt.Errorf("inherit matrix GPU lease: %w", err)
	}
	release, err := acquire()
	if release != nil {
		owners.add(matrixLease, release)
	}
	owners.addError(err)
	if err != nil {
		return fmt.Errorf("acquire matrix GPU lease: %w", err)
	}
	return nil
}

type matrixClientError struct {
	cause error
	close func() error
}

func (e *matrixClientError) Error() string { return e.cause.Error() }
func (e *matrixClientError) Unwrap() error { return e.cause }
func (e *matrixClientError) Close() error  { return e.close() }

type matrixClientStream struct {
	drain, close func() error
	closed       bool
}

func (s *matrixClientStream) Synchronize(context.Context) error {
	if s.closed {
		panic("drain after stream close")
	}
	return s.drain()
}
func (s *matrixClientStream) Close() error {
	if s.closed {
		panic("repeat successful stream close")
	}
	err := s.close()
	s.closed = err == nil
	return err
}

func TestMatrixClientLease(t *testing.T) {
	failed := errors.New("injected lease failure")
	for _, stage := range []string{"none", "admit", "acquire"} {
		t.Run(stage, func(t *testing.T) {
			var calls []string
			owners := &matrixOwners{}
			err := setupMatrixLease(owners,
				func() error {
					calls = append(calls, "admit")
					if stage == "admit" {
						return failed
					}
					return nil
				},
				func() (func() error, error) {
					calls = append(calls, "acquire")
					if stage == "acquire" {
						return nil, &matrixClientError{failed, func() error { calls = append(calls, "pending lease"); return nil }}
					}
					return func() error { calls = append(calls, "lease"); return nil }, nil
				},
			)
			if (stage == "none") != (err == nil) || err != nil && !errors.Is(err, failed) {
				t.Fatalf("setup = %v", err)
			}
			if err := owners.close(); err != nil {
				t.Fatal(err)
			}
			want := map[string][]string{"none": {"admit", "acquire", "lease"}, "admit": {"admit"}, "acquire": {"admit", "acquire", "pending lease"}}[stage]
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

func TestMatrixClientPartialStream(t *testing.T) {
	// The fake acquisition returns the actual owner together with an error.
	// Routing checks below bind registration-before-check to runTileMatrix.
	failed := errors.New("restore failed")
	for _, stage := range []string{"none", "drain", "error owner", "buffer", "module", "stream", "device", "lease"} {
		t.Run(stage, func(t *testing.T) {
			var calls []string
			refuse := true
			release := func(name string) func() error {
				return func() error {
					calls = append(calls, name)
					if refuse && stage == name {
						return failed
					}
					return nil
				}
			}
			owners := &matrixOwners{}
			owners.add(matrixLease, release("lease"))
			owners.add(matrixDevice, release("device"))
			owners.add(matrixModule, release("module"))
			owners.add(matrixBuffer, release("buffer"))
			create := func() (*matrixClientStream, error) {
				return &matrixClientStream{drain: release("drain"), close: release("stream")}, failed
			}
			stream, err := create()
			if stream != nil {
				owners.ownStream(stream)
			}
			if !errors.Is(err, failed) {
				t.Fatalf("acquisition = %v", err)
			}
			owners.addError(fmt.Errorf("load: %w", &matrixClientError{failed, release("error owner")}))
			err = closeMatrixOwners(owners)
			order := []string{"drain", "error owner", "buffer", "module", "stream", "device", "lease"}
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
				retainedMatrixOwners.Lock()
				found := false
				for _, retained := range retainedMatrixOwners.owners {
					found = found || retained == owners
				}
				retainedMatrixOwners.Unlock()
				if !found {
					t.Fatal("failed owners not retained")
				}
			}
			refuse = false
			calls = nil
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

func TestMatrixClientAllocationRetry(t *testing.T) {
	freeFailure := errors.New("native free failed")
	restoreFailure := errors.New("restore failed after callback")
	for _, stage := range []string{"none", "free", "restore", "both"} {
		t.Run(stage, func(t *testing.T) {
			refuse := true
			frees, freed, contexts, devices, leases := 0, 0, 0, 0, 0
			owners := &matrixOwners{}
			owners.add(matrixLease, func() error { leases++; return nil })
			owners.add(matrixDevice, func() error { devices++; return nil })
			owners.add(matrixBuffer, matrixAllocationClose(
				func(fn func() error) error {
					contexts++
					err := fn()
					if refuse && (stage == "restore" || stage == "both") {
						err = errors.Join(err, restoreFailure)
					}
					return err
				},
				func() error {
					frees++
					if refuse && (stage == "free" || stage == "both") {
						return freeFailure
					}
					freed++
					return nil
				},
			))
			err := closeMatrixOwners(owners)
			if frees != 1 || contexts != 1 {
				t.Fatalf("first close: frees=%d contexts=%d", frees, contexts)
			}
			if stage == "none" {
				if err != nil || freed != 1 || devices != 1 || leases != 1 {
					t.Fatalf("close: err=%v freed=%d device=%d lease=%d", err, freed, devices, leases)
				}
			} else {
				if err == nil || devices != 0 || leases != 0 {
					t.Fatalf("failure released parents: err=%v device=%d lease=%d", err, devices, leases)
				}
				if (stage == "free" || stage == "both") && !errors.Is(err, freeFailure) {
					t.Fatalf("lost free cause: %v", err)
				}
				if (stage == "restore" || stage == "both") && !errors.Is(err, restoreFailure) {
					t.Fatalf("lost restore cause: %v", err)
				}
				if stage == "restore" {
					if freed != 1 {
						t.Fatalf("successful native free lost: %d", freed)
					}
					// A second restore failure must still retain device and lease without
					// issuing another native free; success below qualifies final release.
					if err := owners.close(); !errors.Is(err, restoreFailure) {
						t.Fatalf("second restoration = %v", err)
					}
					if frees != 1 || contexts != 2 || devices != 0 || leases != 0 {
						t.Fatalf("unqualified retry: frees=%d contexts=%d device=%d lease=%d", frees, contexts, devices, leases)
					}
				} else if freed != 0 {
					t.Fatalf("failed free counted as success: %d", freed)
				}
				retainedMatrixOwners.Lock()
				found := false
				for _, retained := range retainedMatrixOwners.owners {
					found = found || retained == owners
				}
				retainedMatrixOwners.Unlock()
				if !found {
					t.Fatal("failed allocation owner not retained")
				}
			}
			refuse = false
			if err := owners.close(); err != nil {
				t.Fatal(err)
			}
			wantFrees := 1
			if stage == "free" || stage == "both" {
				wantFrees = 2
			}
			if frees != wantFrees || freed != 1 || devices != 1 || leases != 1 {
				t.Fatalf("qualified retry: frees=%d freed=%d device=%d lease=%d", frees, freed, devices, leases)
			}
			beforeFrees, beforeContexts := frees, contexts
			if err := owners.close(); err != nil || frees != beforeFrees || contexts != beforeContexts {
				t.Fatalf("repeated close: err=%v frees=%d contexts=%d", err, frees, contexts)
			}
		})
	}
}

func TestMatrixClientNoStream(t *testing.T) {
	var calls []string
	owners := &matrixOwners{}
	owners.add(matrixLease, func() error { calls = append(calls, "lease"); return nil })
	owners.add(matrixDevice, func() error { calls = append(calls, "device"); return nil })
	owners.addError(errors.New("creation failed without owner"))
	if err := owners.close(); err != nil {
		t.Fatal(err)
	}
	if want := []string{"device", "lease"}; !reflect.DeepEqual(calls, want) {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
	if owners.quiesced {
		t.Fatal("no-stream cleanup claimed a drain")
	}
}

func TestMatrixClientRouting(t *testing.T) {
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
	body := read("matrix_physical_test.go", "runTileMatrix")
	sequence(body, `ifos.Getenv("CUDA_TILE_TEST")!="1"`, `owners:=newMatrixOwners(t)`, `owners.addError(err)`, `iferr!=nil{t.Fatal(err)}`,
		`err:=setupMatrixLease(owners,`, `cudalock.Inherited(os.Getenv)`, `cudalock.Acquire(`, `iferr!=nil{t.Fatal(err)}`, `requireTile(t)`, `dev,err:=cuda.NewDevice(0)`, `ifdev!=nil{owners.add(matrixDevice,dev.Close)}`, `check(err)`,
		`compileTileMatrixWithOwners(t,int(major),int(minor),mixed,owners)`, `cm.LoadOwnedOutput(dev,nvrtcx.OutputTileIR)`, `ifmod!=nil{owners.add(matrixModule,mod.Close)}`, `check(err)`,
		`cm.LoadOwnedOutput(dev,nvrtcx.OutputPTX)`, `ifptxMod!=nil{owners.add(matrixModule,ptxMod.Close)}`, `check(err)`,
		`stream,err:=dev.NewStream()`, `ifstream!=nil{owners.ownStream(stream)}`, `check(err)`,
		`p,err=driver.MemAllocBytes(`, `ifp!=0{owners.add(matrixBuffer,matrixAllocationClose(dev.WithContext,p.Free))}`, `check(err)`)
	if strings.Contains(body, "cleanup(") || strings.Contains(body, "t.Cleanup(") || strings.Contains(body, "cudatest.") {
		t.Fatal("independent resource or lease cleanup")
	}
	compiler := read("matrix_physical_test.go", "compileTileMatrixWithOwners")
	sequence(compiler, `nvrtcx.Compile(source,opts...)`, `ifowners!=nil{owners.addError(err)}`, `iferr!=nil{t.Fatal(err)}`)
	wrapper := read("matrix_physical_test.go", "compileTileMatrix")
	if !strings.Contains(wrapper, `returncompileTileMatrixWithOwners(t,major,minor,mixed,nil)`) {
		t.Fatal("compiler-only route changed")
	}
	if strings.Contains(read("matrix_physical_test.go", "requireTile"), "cudalock") {
		t.Fatal("compiler-only admission requires GPU lease")
	}
	lease := read("matrix_constructor_custody_test.go", "setupMatrixLease")
	sequence(lease, `iferr:=admit();err!=nil`, `release,err:=acquire()`, `owners.add(matrixLease,release)`, `owners.addError(err)`, `iferr!=nil`)
	allocation := read("matrix_constructor_custody_test.go", "matrixAllocationClose")
	sequence(allocation, `freed:=false`, `returnfunc()error{returnrun(func()error{`, `iffreed{returnnil}`, `iferr:=free();err!=nil{returnerr}`, `freed=true`, `returnnil`)

	cleanup := read("matrix_constructor_custody_test.go", "newMatrixOwners")
	if !strings.Contains(cleanup, `t.Cleanup(func(){iferr:=closeMatrixOwners(owners);err!=nil{t.Error(err)}})`) {
		t.Fatal("retaining cleanup route missing")
	}
}
