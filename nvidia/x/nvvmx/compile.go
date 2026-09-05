package nvvmx

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvvm"
)

// Input is a named NVVM IR module. Lazy inputs must contain bitcode and are
// loaded only for symbols required by non-lazy inputs. Compile copies Data;
// callers must not change it concurrently with Compile.
type Input struct {
	Name string
	Data []byte
	Lazy bool
}

// Result contains copied compiler output and diagnostics. Exactly one of PTX
// and LTOIR is set on success. Output must not be used if Compile returns an
// error. A non-nil result accompanying an error retains available diagnostics.
type Result struct {
	PTX, LTOIR []byte
	Log        string
	resources  *resources
}

type resources struct {
	mu   sync.Mutex
	h    program
	pins *runtime.Pinner
}

// Close retries failed native destruction. Successful Compile already released
// its resources. Close is nil-safe, idempotent, and safe for concurrent calls.
// When destruction fails, the result retains the program and input pins until
// a later Close succeeds.
func (r *Result) Close() error {
	if r == nil || r.resources == nil {
		return nil
	}
	s := r.resources
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.h == nil {
		return nil
	}
	if err := s.h.close(); err != nil {
		return fmt.Errorf("nvvmx: destroy: %w", err)
	}
	s.h = nil
	s.pins.Unpin()
	s.pins = nil
	return nil
}

type program interface {
	AddModuleTo(unsafe.Pointer, uint, string) error
	LazyAddModuleTo(unsafe.Pointer, uint, string) error
	Compile([]string) error
	GetCompiledResult() ([]byte, error)
	GetProgramLog() (string, error)
	close() error
}

type nativeProgram struct{ nvvm.Program }

func (p *nativeProgram) close() error { return nvvm.DestroyProgram(&p.Program) }

// Compile synchronously compiles inputs using native NVVM options. At least one
// input must be non-lazy. The -gen-lto option selects LTOIR instead of PTX;
// unsupported IR and options are diagnosed by the loaded library. No library
// installation, implicit libdevice lookup, or caching is performed.
func Compile(inputs []Input, options ...string) (*Result, error) {
	return compile(inputs, options, func() (program, error) {
		p, err := nvvm.CreateProgram()
		if err != nil {
			return nil, err
		}
		return &nativeProgram{p}, nil
	})
}

func compile(inputs []Input, options []string, create func() (program, error)) (result *Result, err error) {
	if len(inputs) == 0 {
		return nil, errors.New("nvvmx: no inputs")
	}
	lto := false
	for _, option := range options {
		if option == "" || strings.ContainsRune(option, 0) {
			return nil, errors.New("nvvmx: invalid option")
		}
		lto = lto || option == "-gen-lto"
	}
	nonlazy := false
	for i, input := range inputs {
		if input.Name == "" || strings.ContainsRune(input.Name, 0) || len(input.Data) == 0 {
			return nil, fmt.Errorf("nvvmx: invalid input %d", i)
		}
		nonlazy = nonlazy || !input.Lazy
	}
	if !nonlazy {
		return nil, errors.New("nvvmx: all inputs are lazy")
	}
	owned := make([]Input, len(inputs))
	pins := new(runtime.Pinner)
	transferred := false
	defer func() {
		if !transferred {
			pins.Unpin()
		}
	}()
	for i, input := range inputs {
		owned[i] = input
		owned[i].Data = append([]byte(nil), input.Data...)
		pins.Pin(&owned[i].Data[0])
	}
	h, err := create()
	if err != nil {
		return nil, fmt.Errorf("nvvmx: create: %w", err)
	}
	result = &Result{resources: &resources{h: h, pins: pins}}
	transferred = true
	defer func() {
		var logErr error
		result.Log, logErr = h.GetProgramLog()
		if logErr != nil {
			err = errors.Join(err, fmt.Errorf("nvvmx: log: %w", logErr))
		}
		err = errors.Join(err, result.Close())
		if err != nil {
			err = fmt.Errorf("nvvmx: compile: %w\ncompiler log:\n%s", err, result.Log)
		}
	}()
	for _, input := range owned {
		add := h.AddModuleTo
		if input.Lazy {
			add = h.LazyAddModuleTo
		}
		if err := add(unsafe.Pointer(&input.Data[0]), uint(len(input.Data)), input.Name); err != nil {
			return result, fmt.Errorf("add %q: %w", input.Name, err)
		}
	}
	if err := h.Compile(append([]string(nil), options...)); err != nil {
		return result, fmt.Errorf("compile: %w", err)
	}
	data, err := h.GetCompiledResult()
	if err != nil {
		return result, fmt.Errorf("output: %w", err)
	}
	if len(data) == 0 {
		return result, errors.New("empty compiled output")
	}
	data = append([]byte(nil), data...)
	if lto {
		result.LTOIR = data
	} else {
		result.PTX = data
	}
	return result, nil
}
