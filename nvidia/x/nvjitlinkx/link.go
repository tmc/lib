package nvjitlinkx

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvjitlink"
)

// Input is one named device-code input. Kind must describe Data's format.
// Link copies Data before passing it to the native linker. The caller must not
// mutate Data concurrently with Link.
type Input struct {
	Name string
	Kind nvjitlink.InputType
	Data []byte
}

// Result owns linked output and diagnostics. Exactly one of CUBIN or PTX is set
// after successful linking. A non-nil Result may accompany an error so diagnostics
// remain available; its output must not be used when Link reports an error.
type Result struct {
	resources *linkResources
	CUBIN     []byte
	PTX       []byte
	InfoLog   string
	ErrorLog  string
}

type linkResources struct {
	mu   sync.Mutex
	h    linkHandle
	pins *runtime.Pinner
}

// Close retries native destruction after a cleanup failure. It is safe to call
// repeatedly or concurrently. Successful Link already closed its native resources.
// A Result returned with an error must be closed if its initial cleanup failed;
// until then it retains the linker and input pins.
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
		return err
	}
	s.h = nil
	s.pins.Unpin()
	s.pins = nil
	return nil
}

type linkHandle interface {
	AddData(nvjitlink.InputType, unsafe.Pointer, uint, string) error
	Complete() error
	GetLinkedCubin() ([]byte, error)
	GetLinkedPtx() ([]byte, error)
	GetErrorLog() (string, error)
	GetInfoLog() (string, error)
	close() error
}

type nativeHandle struct{ nvjitlink.Handle }

func (h *nativeHandle) close() error { return nvjitlink.Destroy(&h.Handle) }

// Link synchronously links inputs using native nvJitLink options, including a
// required -arch= target. The -ptx option selects PTX and requires -lto; otherwise
// CUBIN is returned. LTOIR inputs require -lto. Toolkit compatibility and unknown
// options are checked by the loaded linker and reported with its diagnostic log.
// Link does not install a library or cache results.
func Link(inputs []Input, options ...string) (*Result, error) {
	return link(inputs, options, func(options []string) (linkHandle, error) {
		h, err := nvjitlink.Create(options)
		if err != nil {
			return nil, err
		}
		return &nativeHandle{h}, nil
	})
}

func link(inputs []Input, options []string, create func([]string) (linkHandle, error)) (result *Result, err error) {
	if len(inputs) == 0 {
		return nil, errors.New("nvjitlinkx: no inputs")
	}
	ptx, lto, arch := false, false, false
	for _, option := range options {
		if option == "" || strings.ContainsRune(option, 0) {
			return nil, errors.New("nvjitlinkx: invalid option")
		}
		ptx = ptx || option == "-ptx"
		lto = lto || option == "-lto"
		arch = arch || strings.HasPrefix(option, "-arch=") && len(option) > len("-arch=")
	}
	if !arch {
		return nil, errors.New("nvjitlinkx: missing -arch target")
	}
	if ptx && !lto {
		return nil, errors.New("nvjitlinkx: PTX output requires -lto")
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
		if input.Name == "" || strings.ContainsRune(input.Name, 0) || len(input.Data) == 0 {
			return nil, fmt.Errorf("nvjitlinkx: invalid input %d", i)
		}
		switch input.Kind {
		case nvjitlink.InputPtx, nvjitlink.InputCubin, nvjitlink.InputFatbin, nvjitlink.InputObject, nvjitlink.InputLibrary, nvjitlink.InputIndex:
		case nvjitlink.InputLtoir:
			if !lto {
				return nil, fmt.Errorf("nvjitlinkx: LTOIR input %q requires -lto", input.Name)
			}
		default:
			return nil, fmt.Errorf("nvjitlinkx: unsupported input kind %d", input.Kind)
		}
		owned[i] = input
		owned[i].Data = append([]byte(nil), input.Data...)
		if input.Kind == nvjitlink.InputPtx && owned[i].Data[len(owned[i].Data)-1] != 0 {
			owned[i].Data = append(owned[i].Data, 0)
		}
		pins.Pin(&owned[i].Data[0])
	}
	h, err := create(append([]string(nil), options...))
	if err != nil {
		return nil, fmt.Errorf("nvjitlinkx: create: %w", err)
	}
	result = &Result{resources: &linkResources{h: h, pins: pins}}
	transferred = true
	// This defer runs before the input pins are released.
	defer func() {
		var logErr, infoErr error
		result.ErrorLog, logErr = h.GetErrorLog()
		result.InfoLog, infoErr = h.GetInfoLog()
		err = errors.Join(err, logErr, infoErr)
		if closeErr := result.Close(); closeErr != nil {
			err = errors.Join(err, fmt.Errorf("nvjitlinkx: destroy: %w", closeErr))
		}
		if err != nil {
			err = fmt.Errorf("nvjitlinkx: link: %w\nlinker log:\n%s", err, result.ErrorLog)
		}
	}()
	for _, input := range owned {
		if err := h.AddData(input.Kind, unsafe.Pointer(&input.Data[0]), uint(len(input.Data)), input.Name); err != nil {
			return result, fmt.Errorf("add %q: %w", input.Name, err)
		}
	}
	if err := h.Complete(); err != nil {
		return result, fmt.Errorf("complete: %w", err)
	}
	var data []byte
	if ptx {
		data, err = h.GetLinkedPtx()
	} else {
		data, err = h.GetLinkedCubin()
	}
	if err != nil {
		return result, fmt.Errorf("output: %w", err)
	}
	if len(data) == 0 {
		return result, errors.New("empty linked output")
	}
	data = append([]byte(nil), data...)
	if ptx {
		result.PTX = data
	} else {
		result.CUBIN = data
	}
	return result, nil
}
