// Package abistub builds C libraries that stand in for a vendor library in
// native ABI tests of the generated bindings.
//
// A stub starts with [Prelude] and records each argument it receives with
// its macros. A test loads the stub through the package's own Load, calls a
// wrapper, and compares [Recorder.Recorded] with the arguments it passed,
// bit for bit. The stub returns the value set by [Recorder.Reset], so the
// same call also checks error conversion.
package abistub

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"unsafe"
)

// Prelude declares the record a stub keeps and the macros that fill it.
// Each function starts with BEGIN and returns RET:
//
//	R(x)    records an unsigned integer, pointer or handle, zero-extended
//	S(x)    records a signed integer, sign-extended
//	F32(x)  records a float's bit pattern
//	F64(x)  records a double's bit pattern
const Prelude = `
#include <stddef.h>
#include <stdint.h>
#include <string.h>

uint64_t stubRec[64];
int stubN;
int stubRet;

#define BEGIN (stubN = 0)
#define RET stubRet
#define R(x) (stubRec[stubN++] = (uint64_t)(x))
#define S(x) (stubRec[stubN++] = (uint64_t)(int64_t)(x))
#define F32(x) do { float f_ = (x); uint32_t u_; memcpy(&u_, &f_, 4); R(u_); } while (0)
#define F64(x) do { double f_ = (x); uint64_t u_; memcpy(&u_, &f_, 8); R(u_); } while (0)
`

// Build compiles csrc, with flags, as a shared library and returns its
// path. It skips tb when there is no C compiler.
func Build(tb testing.TB, csrc string, flags ...string) string {
	tb.Helper()
	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		tb.Skip("no C compiler")
	}
	dir := tb.TempDir()
	src := filepath.Join(dir, "stub.c")
	if err := os.WriteFile(src, []byte(csrc), 0o644); err != nil {
		tb.Fatal(err)
	}
	path := filepath.Join(dir, "libstub.so")
	args := append([]string{"-shared", "-fPIC", "-Wno-deprecated-declarations", "-o", path, src}, flags...)
	if runtime.GOOS == "darwin" {
		arch := runtime.GOARCH
		if arch == "amd64" {
			arch = "x86_64"
		}
		args = append([]string{"-arch", arch}, args...)
	}
	if out, err := exec.Command(cc, args...).CombinedOutput(); err != nil {
		tb.Fatalf("cc: %v\n%s", err, out)
	}
	return path
}

// Child runs the top-level test named test in a child process of the test
// binary, with the environment variable env set to lib, and fails t if the
// child fails. A test that loads a stub runs there, so the stub never
// becomes another test's library.
func Child(t *testing.T, test, env, lib string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+test+"$", "-test.v", "-test.count=1")
	cmd.Env = append(os.Environ(), env+"="+lib)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child: %v\n%s", err, out)
	}
	if testing.Verbose() {
		t.Logf("child output:\n%s", out)
	}
}

// A Recorder reads the record of a loaded stub.
type Recorder struct {
	lookup func(string) (uintptr, error)
	rec    *[64]uint64
	n      *int32
	ret    *int32
}

// NewRecorder returns a Recorder for the stub whose data symbols lookup
// resolves.
func NewRecorder(lookup func(string) (uintptr, error)) (*Recorder, error) {
	r := &Recorder{lookup: lookup}
	for name, dst := range map[string]*unsafe.Pointer{
		"stubRec": (*unsafe.Pointer)(unsafe.Pointer(&r.rec)),
		"stubN":   (*unsafe.Pointer)(unsafe.Pointer(&r.n)),
		"stubRet": (*unsafe.Pointer)(unsafe.Pointer(&r.ret)),
	} {
		p, err := r.symbol(name)
		if err != nil {
			return nil, err
		}
		*dst = p
	}
	return r, nil
}

func (r *Recorder) symbol(name string) (unsafe.Pointer, error) {
	addr, err := r.lookup(name)
	if err == nil && addr == 0 {
		err = fmt.Errorf("address is zero")
	}
	if err != nil {
		return nil, fmt.Errorf("stub symbol %s: %w", name, err)
	}
	// addr is C data, not a Go pointer.
	return *(*unsafe.Pointer)(unsafe.Pointer(&addr)), nil
}

// Reset clears the record and sets the value each stub function returns.
func (r *Recorder) Reset(ret int32) {
	*r.n = 0
	*r.ret = ret
}

// Recorded returns the words the last stub call recorded.
func (r *Recorder) Recorded() []uint64 {
	return slices.Clone(r.rec[:*r.n])
}

// Uint64s returns the first n elements of the stub's uint64_t array name.
func (r *Recorder) Uint64s(name string, n int) ([]uint64, error) {
	p, err := r.symbol(name)
	if err != nil {
		return nil, err
	}
	return slices.Clone(unsafe.Slice((*uint64)(p), n)), nil
}

// CUDAInclude returns the CUDA toolkit include directory holding header,
// or "" if there is none. It looks in $CUDA_HOME, $CUDA_PATH and
// /usr/local/cuda, then in /usr, where distribution packages such as
// NCCL's install their headers. A stub built against the real header with
// -DSTUB_REAL_HEADER has its prototypes and layouts checked by the compiler.
func CUDAInclude(header string) string {
	for _, dir := range []string{os.Getenv("CUDA_HOME"), os.Getenv("CUDA_PATH"), "/usr/local/cuda", "/usr"} {
		if dir == "" {
			continue
		}
		inc := filepath.Join(dir, "include")
		if _, err := os.Stat(filepath.Join(inc, header)); err == nil {
			return inc
		}
	}
	return ""
}
