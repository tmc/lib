package nvvmx

import (
	"bytes"
	"errors"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

var errInjected = errors.New("injected")

type fakeProgram struct {
	fail   string
	inputs [][]byte
	lazy   []bool
	output []byte
	closes int
}

func (p *fakeProgram) add(ptr unsafe.Pointer, size uint, lazy bool) error {
	p.inputs = append(p.inputs, unsafe.Slice((*byte)(ptr), int(size)))
	p.lazy = append(p.lazy, lazy)
	if p.fail == "add" || lazy && p.fail == "lazy" {
		return errInjected
	}
	return nil
}
func (p *fakeProgram) AddModuleTo(ptr unsafe.Pointer, size uint, _ string) error {
	return p.add(ptr, size, false)
}
func (p *fakeProgram) LazyAddModuleTo(ptr unsafe.Pointer, size uint, _ string) error {
	return p.add(ptr, size, true)
}
func (p *fakeProgram) Compile([]string) error {
	if p.fail == "compile" {
		return errInjected
	}
	return nil
}
func (p *fakeProgram) GetCompiledResult() ([]byte, error) {
	if p.fail == "output" {
		return nil, errInjected
	}
	return p.output, nil
}
func (p *fakeProgram) GetProgramLog() (string, error) {
	if p.fail == "log" {
		return "diagnostic", errInjected
	}
	return "diagnostic", nil
}
func (p *fakeProgram) close() error {
	p.closes++
	if p.fail == "close" {
		return errInjected
	}
	clear(p.output)
	return nil
}

func TestCompileLifecycle(t *testing.T) {
	for _, fail := range []string{"", "create", "add", "lazy", "compile", "output", "empty", "log", "close"} {
		t.Run(fail, func(t *testing.T) {
			p := &fakeProgram{fail: fail, output: []byte{1, 0, 2}}
			if fail == "empty" {
				p.output = nil
			}
			inputs := []Input{{Name: "module", Data: []byte("ir")}, {Name: "libdevice", Data: []byte("bitcode"), Lazy: true}}
			r, err := compile(inputs, nil, func() (program, error) {
				if fail == "create" {
					return nil, errInjected
				}
				return p, nil
			})
			if (err != nil) != (fail != "") {
				t.Fatalf("error = %v", err)
			}
			if fail == "create" {
				if r != nil || p.closes != 0 {
					t.Fatal("created resources on failure")
				}
				return
			}
			if r.Log != "diagnostic" || p.closes != 1 {
				t.Fatalf("log = %q, closes = %d", r.Log, p.closes)
			}
			if fail != "" && !strings.Contains(err.Error(), r.Log) {
				t.Fatalf("missing diagnostic: %v", err)
			}
			if fail != "" && fail != "empty" && !errors.Is(err, errInjected) {
				t.Fatalf("lost error: %v", err)
			}
			if fail == "close" {
				if r.resources.h == nil || r.resources.pins == nil {
					t.Fatal("released resources after failed destroy")
				}
				p.fail = ""
				var wg sync.WaitGroup
				for range 8 {
					wg.Go(func() {
						if err := r.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				wg.Wait()
				if p.closes != 2 {
					t.Fatalf("closes = %d", p.closes)
				}
			}
			if r.resources.h != nil || r.resources.pins != nil {
				t.Fatal("resources retained after successful destroy")
			}
			if fail == "" {
				if !bytes.Equal(r.PTX, []byte{1, 0, 2}) || r.LTOIR != nil {
					t.Fatalf("output = %+v", r)
				}
				inputs[0].Data[0] = 'X'
				if string(p.inputs[0]) != "ir" || !p.lazy[1] {
					t.Fatal("input ownership or lazy selection lost")
				}
			}
		})
	}
}

func TestCompileValidation(t *testing.T) {
	valid := []Input{{Name: "module", Data: []byte("ir")}}
	for _, tt := range []struct {
		name    string
		inputs  []Input
		options []string
	}{
		{"none", nil, nil},
		{"name", []Input{{Data: []byte("ir")}}, nil},
		{"nul name", []Input{{Name: "x\x00y", Data: []byte("ir")}}, nil},
		{"empty", []Input{{Name: "x"}}, nil},
		{"lazy", []Input{{Name: "x", Data: []byte("ir"), Lazy: true}}, nil},
		{"empty option", valid, []string{""}},
		{"nul option", valid, []string{"-g\x00"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := compile(tt.inputs, tt.options, func() (program, error) { t.Fatal("native call before validation"); return nil, nil })
			if r != nil || err == nil {
				t.Fatalf("result = %v, error = %v", r, err)
			}
		})
	}
}

func TestCompileLTO(t *testing.T) {
	p := &fakeProgram{output: []byte{0, 1, 0}}
	r, err := compile([]Input{{Name: "ir", Data: []byte("ir")}}, []string{"-gen-lto"}, func() (program, error) { return p, nil })
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(r.LTOIR, []byte{0, 1, 0}) || r.PTX != nil {
		t.Fatalf("result = %+v", r)
	}
}
