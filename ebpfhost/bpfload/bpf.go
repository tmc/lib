package bpfload

import (
	"errors"
	"fmt"
	"strings"
	"sync"
)

var ErrUnsupported = errors.New("bpf: unsupported platform")
var ErrClosed = errors.New("bpf: closed descriptor")

// ProgramSpec describes a raw Linux BPF program. Type and ExpectedAttachType
// use the upstream linux/bpf.h numeric identifiers. Zero ExpectedAttachType
// leaves the attach type unspecified. License defaults to GPL.
type ProgramSpec struct {
	Type               uint32
	ExpectedAttachType uint32
	Instructions       []byte
	License            string
	// LogSize is bounded to 16 MiB; zero requests 64 KiB.
	LogSize int
}

// LoadError retains the kernel verifier output when loading fails.
type LoadError struct {
	Err error
	Log string
}

func (e *LoadError) Error() string { return fmt.Sprintf("bpf load: %v", e.Err) }
func (e *LoadError) Unwrap() error { return e.Err }

// Object owns a kernel descriptor. Its zero value is closed. Close may be
// called repeatedly and is serialized with map operations and inspection.
type Object struct {
	closeErr error
	mu       sync.Mutex
	fd       int
	open     bool
}

func newObject(fd int) *Object { return &Object{fd: fd, open: true} }
func (o *Object) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.open {
		return o.closeErr
	}
	o.open = false
	o.closeErr = closeFD(o.fd)
	return o.closeErr
}

// MapInfo is the stable leading portion of Linux bpf_map_info.
type MapInfo struct {
	BTFID, BTFKeyTypeID, BTFValueTypeID, BTFVmlinuxValueTypeID uint32
	Type, ID, KeySize, ValueSize, MaxEntries, Flags            uint32
	Name                                                       string
}

// MapSpec describes an ordinary fixed-size Linux BPF map. Type must be hash
// (1), array (2), LRU hash (9), or LPM trie (11). Other layouts are refused.
type MapSpec struct {
	Type, KeySize, ValueSize, MaxEntries, Flags uint32
	Name                                        string
}

func validateMap(s MapSpec) error {
	switch s.Type {
	case 1, 2, 9, 11:
	default:
		return errors.New("bpf: unsupported map value layout")
	}
	if s.KeySize == 0 || s.ValueSize == 0 || s.MaxEntries == 0 {
		return errors.New("bpf: zero map dimension")
	}
	if len(s.Name) > 15 {
		return errors.New("bpf: map name exceeds 15 bytes")
	}
	for _, c := range s.Name {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_') {
			return errors.New("bpf: invalid map name")
		}
	}
	return nil
}

func validateSpec(s ProgramSpec) (ProgramSpec, error) {
	if s.Type == 0 {
		return s, errors.New("bpf: missing program type")
	}
	if len(s.Instructions) == 0 || len(s.Instructions)%8 != 0 || len(s.Instructions) > 8*1000000 {
		return s, errors.New("bpf: invalid instruction size")
	}
	if s.License == "" {
		s.License = "GPL"
	}
	if strings.ContainsRune(s.License, 0) {
		return s, errors.New("bpf: license contains nul")
	}
	if s.LogSize == 0 {
		s.LogSize = 64 << 10
	}
	if s.LogSize < 0 || s.LogSize > 16<<20 {
		return s, errors.New("bpf: invalid verifier log size")
	}
	return s, nil
}
