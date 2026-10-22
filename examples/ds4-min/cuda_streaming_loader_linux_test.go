//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/gguf"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaResidentTestAllocation struct {
	bytes int
	free  func() error
}

func (a *cudaResidentTestAllocation) Len() int    { return a.bytes }
func (a *cudaResidentTestAllocation) Free() error { return a.free() }

func TestCUDAStreamingResidentOwnerFailures(t *testing.T) {
	for _, fault := range []string{"allocate", "stage", "copy", "drain", "free"} {
		t.Run(fault, func(t *testing.T) {
			f := newFoundationCloseFixture(t)
			path := filepath.Join(t.TempDir(), "weights")
			if err := os.WriteFile(path, make([]byte, 4096), 0600); err != nil {
				t.Fatal(err)
			}
			file, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			f.r.file, f.r.model = file, make([]byte, 4096)
			f.r.gguf = &gguf.File{Tensors: []gguf.Tensor{{Name: "norm.weight", Dimensions: []uint64{4}, TypeID: 0}}}
			cause := errors.New("resident fault")
			fail := true
			deviceFree, stageFree, copied, drained := 0, 0, false, false
			f.r.residentOps = &cudaResidentOperations{
				allocate: func(n int) (cudaResidentAllocation, error) {
					free := func() error { deviceFree++; return nil }
					mem := &cudaResidentTestAllocation{n, free}
					if fault == "allocate" {
						return mem, cause
					}
					return mem, nil
				},
				stage: func(int) ([]byte, func() error, error) {
					free := func() error {
						stageFree++
						if fail && fault == "free" {
							return cause
						}
						return nil
					}
					if fault == "stage" {
						return nil, free, cause
					}
					return make([]byte, 4096), free, nil
				},
				run: func(fn func() error) error { return fn() },
				copy: func(cudaResidentAllocation, int, []byte) error {
					copied = true
					if fault == "copy" {
						return cause
					}
					return nil
				},
				drain: func() error {
					drained = true
					if fault == "drain" {
						return cause
					}
					return nil
				},
				drop: func(int, uint64, uint64) error { return nil },
			}
			if err := f.r.loadResidentSet(); !errors.Is(err, cause) {
				t.Fatal(err)
			}
			if deviceFree != 0 || f.r.residentClose == nil {
				t.Fatal("resident allocation owner discarded")
			}
			if fault != "allocate" && f.r.residentStageClose == nil {
				t.Fatal("staging owner discarded")
			}
			if fault == "free" && (!copied || !drained) {
				t.Fatal("stage freed before transfer drain")
			}
			if fault == "free" {
				if err := f.r.Close(); !errors.Is(err, cause) || deviceFree != 0 || f.calls["file"] != 0 {
					t.Fatal("dependent owner released on stage failure", err)
				}
			}
			fail = false
			if err := f.r.Close(); err != nil {
				t.Fatal(err)
			}
			if deviceFree != 1 || (fault != "allocate" && stageFree == 0) || f.r.residentClose != nil || f.r.residentStageClose != nil {
				t.Fatal("resident retry lost owners")
			}
		})
	}
}

func TestCUDAStreamingResidentInvalidAllocation(t *testing.T) {
	for _, size := range []int{-1, 0, 4095, 4097} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			f := newFoundationCloseFixture(t)
			f.r.model = make([]byte, 4096)
			f.r.gguf = &gguf.File{Tensors: []gguf.Tensor{{Name: "norm.weight", Dimensions: []uint64{4}, TypeID: 0}}}
			fail := true
			freed := 0
			unexpected := func() { t.Fatal("invalid allocation used") }
			f.r.residentOps = &cudaResidentOperations{
				allocate: func(int) (cudaResidentAllocation, error) {
					if size < 0 {
						return nil, nil
					}
					return &cudaResidentTestAllocation{size, func() error {
						freed++
						if fail {
							return errors.New("free fault")
						}
						return nil
					}}, nil
				},
				stage: func(int) ([]byte, func() error, error) { unexpected(); return nil, nil, nil },
				run:   func(func() error) error { unexpected(); return nil },
				copy:  func(cudaResidentAllocation, int, []byte) error { unexpected(); return nil },
				drain: func() error { unexpected(); return nil },
				drop:  func(int, uint64, uint64) error { unexpected(); return nil },
			}
			if err := f.r.loadResidentSet(); err == nil {
				t.Fatal("invalid allocation admitted")
			}
			if size >= 0 {
				if f.r.residentClose == nil {
					t.Fatal("invalid allocation owner discarded")
				}
				if err := f.r.Close(); err == nil || f.calls["file"] != 0 {
					t.Fatal("failed free released dependencies")
				}
				fail = false
			}
			if err := f.r.Close(); err != nil {
				t.Fatal(err)
			}
			if size >= 0 && freed != 2 {
				t.Fatal("allocation cleanup not retried")
			}
		})
	}
}

func TestCUDAStreamingFoundationSkipsWarm(t *testing.T) {
	f := newFoundationCloseFixture(t)
	file, dev, stream := f.r.file, f.r.device, f.r.workStream
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	cause := errors.New("stop after streaming")
	loaded := false
	ops := cudaFoundationConstructorOperations{
		open: func(string) (*os.File, error) { return file, nil }, stat: func(*os.File) (os.FileInfo, error) { return info, nil },
		direct: func(string, os.FileInfo) (*os.File, int) { return nil, 0 }, read: func(*os.File, int64) (*gguf.File, error) { return new(gguf.File), nil },
		validate: func(*gguf.File, *os.File, int64) error { return nil }, mapModel: func(*os.File, int64) ([]byte, error) { return []byte{1}, nil },
		device: func() (*cuda.Device, error) { return dev, nil }, describe: func(*cudaFoundation) error { return nil }, stream: func(*cuda.Device) (*cuda.Stream, error) { return stream, nil },
		warm: func(*cudaFoundation, cudaRuntimeOptions) error { t.Fatal("whole model warm reached"); return nil },
		streaming: func(r *cudaFoundation, o streamingOptions, n uint64) error {
			loaded = true
			if o.budget != "6" {
				t.Fatal(o)
			}
			r.streamingExperts = &cudaStreamingExperts{}
			return nil
		},
		probe: func(*cudaFoundation) error { return cause }, close: f.r.closeOps,
	}
	_, err = newCUDAFoundationWithOperations("not-opened", cudaRuntimeOptions{streaming: streamingOptions{enabled: true, budget: "6"}}, ops)
	if !loaded || !errors.Is(err, cause) {
		t.Fatal(loaded, err)
	}
}

func TestCUDAStreamingResidentCopyAdmission(t *testing.T) {
	f := newFoundationCloseFixture(t)
	ops := f.r.residentOperations()
	mem := &cudaResidentTestAllocation{bytes: 1, free: func() error { return nil }}
	if err := ops.copy(mem, 0, []byte{1}); err == nil {
		t.Fatal("invalid resident allocation owner admitted")
	}
	for _, offset := range []int{-1, 0, 1, int(^uint(0) >> 1)} {
		if err := ops.copy(cudamem.Slice[byte]{}, offset, []byte{1}); err == nil {
			t.Fatal("invalid resident copy bounds admitted", offset)
		}
	}
}
