package nvfatbinx

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvfatbin"
)

// Kind identifies an input image format. The zero value is invalid.
type Kind uint8

const (
	CUBIN Kind = iota + 1
	PTX
	LTOIR
	TileIR
)

// Input describes one image. Arch is the numerical architecture string, such
// as "80" or "100a", not "sm_80". TileIR has no Arch. JITOptions is the native
// command-line option string for PTX, LTOIR, or TileIR; CUBIN has no JIT options.
// Build copies Data. Do not mutate it concurrently with Build.
type Input struct {
	Kind       Kind
	Name       string
	Arch       string
	Data       []byte
	JITOptions string
}

// Result owns a fatbinary. Data remains valid after Close. If Build returns an
// error, Data must not be used and any non-nil Result must be closed to retry
// failed cleanup. Successful Build already released all native resources.
type Result struct {
	Data      []byte
	resources *resources
}

type resources struct {
	mu   sync.Mutex
	h    handle
	pins *runtime.Pinner
}

// Close retries native cleanup. It is nil-safe, idempotent and safe for
// concurrent calls. Inputs remain pinned until destruction succeeds. Do not
// unload the native library while Build or cleanup remains in progress.
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
		return fmt.Errorf("nvfatbinx: destroy: %w", err)
	}
	s.h = nil
	s.pins.Unpin()
	s.pins = nil
	return nil
}

type handle interface {
	add(Input) error
	Get() ([]byte, error)
	close() error
}

type nativeHandle struct{ nvfatbin.Handle }

func (h *nativeHandle) close() error { return nvfatbin.Destroy(&h.Handle) }
func (h *nativeHandle) add(input Input) error {
	p, n := unsafe.Pointer(&input.Data[0]), uint(len(input.Data))
	switch input.Kind {
	case CUBIN:
		return h.AddCubin(p, n, input.Arch, input.Name)
	case PTX:
		return h.AddPTX(p, n, input.Arch, input.Name, input.JITOptions)
	case LTOIR:
		return h.AddLTOIR(p, n, input.Arch, input.Name, input.JITOptions)
	case TileIR:
		return h.AddTileIR(p, n, input.Name, input.JITOptions)
	default:
		return errors.New("unsupported input kind")
	}
}

// Build synchronously packages inputs with native nvFatbin options. It neither
// links nor loads the result. Unknown options and incompatible images return
// native errors, annotated with the failing stage and input. Optional formats
// fail with the raw package's SymbolError when the loaded library lacks support;
// they are never silently omitted or converted to another format.
func Build(inputs []Input, options ...string) (*Result, error) {
	return build(inputs, options, func(options []string) (handle, error) {
		h, err := nvfatbin.Create(options)
		if err != nil {
			return nil, err
		}
		return &nativeHandle{h}, nil
	})
}

func build(inputs []Input, options []string, create func([]string) (handle, error)) (result *Result, err error) {
	if len(inputs) == 0 {
		return nil, errors.New("nvfatbinx: no inputs")
	}
	for _, option := range options {
		if option == "" || strings.ContainsRune(option, 0) {
			return nil, errors.New("nvfatbinx: invalid option")
		}
	}
	for i, input := range inputs {
		if input.Name == "" || len(input.Data) == 0 || strings.ContainsRune(input.Name, 0) || strings.ContainsRune(input.Arch, 0) || strings.ContainsRune(input.JITOptions, 0) {
			return nil, fmt.Errorf("nvfatbinx: invalid input %d", i)
		}
		switch input.Kind {
		case CUBIN, PTX, LTOIR:
			if input.Arch == "" {
				return nil, fmt.Errorf("nvfatbinx: input %q has no architecture", input.Name)
			}
			if input.Kind == CUBIN && input.JITOptions != "" {
				return nil, errors.New("nvfatbinx: CUBIN does not accept JIT options")
			}
		case TileIR:
			if input.Arch != "" {
				return nil, errors.New("nvfatbinx: TileIR does not accept an architecture")
			}
		default:
			return nil, fmt.Errorf("nvfatbinx: unsupported input kind %d", input.Kind)
		}
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
		if input.Kind == PTX && owned[i].Data[len(owned[i].Data)-1] != 0 {
			owned[i].Data = append(owned[i].Data, 0)
		}
		pins.Pin(&owned[i].Data[0])
	}
	h, err := create(append([]string(nil), options...))
	if err != nil {
		return nil, fmt.Errorf("nvfatbinx: create: %w", err)
	}
	result = &Result{resources: &resources{h: h, pins: pins}}
	transferred = true
	defer func() { err = errors.Join(err, result.Close()) }()
	for _, input := range owned {
		if err := h.add(input); err != nil {
			return result, fmt.Errorf("nvfatbinx: add %q (%s): %w", input.Name, input.Arch, err)
		}
	}
	data, err := h.Get()
	if err != nil {
		return result, fmt.Errorf("nvfatbinx: get: %w", err)
	}
	if len(data) == 0 {
		return result, errors.New("nvfatbinx: empty fatbinary")
	}
	result.Data = append([]byte(nil), data...)
	return result, nil
}
