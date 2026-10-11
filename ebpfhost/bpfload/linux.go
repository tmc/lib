//go:build linux && (amd64 || arm64)

package bpfload

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

func call(command uintptr, attr unsafe.Pointer, size uintptr) (int, error) {
	r, _, e := syscall.Syscall(sysBPF, command, uintptr(attr), size)
	if e != 0 {
		return -1, e
	}
	return int(r), nil
}
func closeFD(fd int) error { return syscall.Close(fd) }
func pointer(b []byte) uint64 {
	if len(b) == 0 {
		return 0
	}
	return uint64(uintptr(unsafe.Pointer(&b[0])))
}
func logText(b []byte) string {
	if n := bytes.IndexByte(b, 0); n >= 0 {
		b = b[:n]
	}
	return string(b)
}

// Load loads a program and returns its owned descriptor and verifier log.
// It does not attach the program or retry a refused load.
func Load(spec ProgramSpec) (*Object, string, error) {
	s, err := validateSpec(spec)
	if err != nil {
		return nil, "", err
	}
	license := append([]byte(s.License), 0)
	log := make([]byte, s.LogSize)
	attr := struct {
		Type, Count                 uint32
		Instructions, License       uint64
		LogLevel, LogSize           uint32
		LogBuffer                   uint64
		KernelVersion, Flags        uint32
		Name                        [16]byte
		IfIndex, ExpectedAttachType uint32
	}{Type: s.Type, Count: uint32(len(s.Instructions) / 8), Instructions: pointer(s.Instructions), License: pointer(license), LogLevel: 1, LogSize: uint32(len(log)), LogBuffer: pointer(log), ExpectedAttachType: s.ExpectedAttachType}
	fd, err := call(5, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	runtime.KeepAlive(s.Instructions)
	runtime.KeepAlive(license)
	runtime.KeepAlive(log)
	text := logText(log)
	if err != nil {
		return nil, text, &LoadError{Err: err, Log: text}
	}
	return newObject(fd), text, nil
}

// LoadBTF asks the kernel to validate and load a raw BTF blob. ELF extraction
// is the caller's responsibility. Successful loading is not GPU qualification.
func LoadBTF(data []byte, logSize int) (*Object, string, error) {
	if len(data) == 0 || len(data) > 64<<20 {
		return nil, "", errors.New("bpf: invalid btf size")
	}
	if logSize == 0 {
		logSize = 64 << 10
	}
	if logSize < 0 || logSize > 16<<20 {
		return nil, "", errors.New("bpf: invalid verifier log size")
	}
	log := make([]byte, logSize)
	attr := struct {
		Data, Log               uint64
		Size, LogSize, LogLevel uint32
	}{pointer(data), pointer(log), uint32(len(data)), uint32(len(log)), 1}
	fd, err := call(18, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	runtime.KeepAlive(data)
	runtime.KeepAlive(log)
	text := logText(log)
	if err != nil {
		return nil, text, &LoadError{Err: err, Log: text}
	}
	return newObject(fd), text, nil
}

// CreateMap creates a fixed-size map with an owned descriptor. Kernel size,
// flag, privilege, and resource-limit refusals are returned without retry.
func CreateMap(s MapSpec) (*Object, error) {
	if err := validateMap(s); err != nil {
		return nil, err
	}
	attr := struct {
		Type, KeySize, ValueSize, MaxEntries, Flags, InnerFD, NUMANode uint32
		Name                                                           [16]byte
	}{Type: s.Type, KeySize: s.KeySize, ValueSize: s.ValueSize, MaxEntries: s.MaxEntries, Flags: s.Flags}
	copy(attr.Name[:], s.Name)
	fd, err := call(0, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return nil, fmt.Errorf("bpf create map: %w", err)
	}
	return newObject(fd), nil
}

// CreateRingMap creates a BPF ring-buffer map. capacity must be a power of two
// and a multiple of the system page size. The map has no keys or values.
func CreateRingMap(capacity int) (*Object, error) {
	if capacity < os.Getpagesize() || capacity > 64<<20 || capacity&(capacity-1) != 0 || capacity%os.Getpagesize() != 0 {
		return nil, errors.New("bpf: invalid ring capacity")
	}
	attr := struct{ Type, KeySize, ValueSize, MaxEntries, Flags uint32 }{Type: 27, MaxEntries: uint32(capacity)}
	fd, err := call(0, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return nil, fmt.Errorf("bpf create ring: %w", err)
	}
	return newObject(fd), nil
}

// CreateStackMap creates a bounded user-stack trace map. Capturing stack IDs
// must avoid BPF_F_REUSE_STACKID and BPF_F_FAST_STACK_CMP when stable frame
// ownership is required. Lookup returns depth uint64 addresses, padded by zero.
func CreateStackMap(capacity, depth int) (*Object, error) {
	if capacity <= 0 || capacity > 1024 || depth <= 0 || depth > 8 {
		return nil, errors.New("bpf: invalid stack map limits")
	}
	attr := struct{ Type, KeySize, ValueSize, MaxEntries, Flags uint32 }{Type: 7, KeySize: 4, ValueSize: uint32(depth * 8), MaxEntries: uint32(capacity)}
	fd, err := call(0, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	if err != nil {
		return nil, fmt.Errorf("bpf create stack map: %w", err)
	}
	return newObject(fd), nil
}

// File duplicates the object's descriptor with close-on-exec set. The caller
// owns the returned file; closing either handle does not invalidate the other.
func (o *Object) File() (*os.File, error) {
	if o == nil {
		return nil, ErrClosed
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.open {
		return nil, ErrClosed
	}
	fd, _, errno := syscall.Syscall(syscall.SYS_FCNTL, uintptr(o.fd), syscall.F_DUPFD_CLOEXEC, 0)
	if errno != 0 {
		return nil, fmt.Errorf("bpf duplicate descriptor: %w", errno)
	}
	return os.NewFile(fd, "bpf"), nil
}

// Pin gives the object a persistent bpffs path. It never replaces an existing
// pin. The caller owns removal of the path; Close only closes this descriptor.
func (o *Object) Pin(path string) error {
	if o == nil {
		return ErrClosed
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.open {
		return ErrClosed
	}
	p, err := syscall.ByteSliceFromString(path)
	if err != nil || path == "" {
		return errors.New("bpf: invalid pin path")
	}
	attr := struct {
		Path      uint64
		FD, Flags uint32
	}{Path: pointer(p), FD: uint32(o.fd)}
	_, err = call(6, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	runtime.KeepAlive(p)
	if err != nil {
		return fmt.Errorf("bpf pin: %w", err)
	}
	return nil
}

// OpenMap opens a pinned map read-only when readOnly is true, and verifies
// that the object is a map before returning its owned descriptor.
func OpenMap(path string, readOnly bool) (*Object, MapInfo, error) {
	p, err := syscall.ByteSliceFromString(path)
	if err != nil || path == "" {
		return nil, MapInfo{}, errors.New("bpf: invalid pin path")
	}
	attr := struct {
		Path      uint64
		FD, Flags uint32
	}{Path: pointer(p)}
	if readOnly {
		attr.Flags = 8
	}
	fd, err := call(7, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	runtime.KeepAlive(p)
	if err != nil {
		return nil, MapInfo{}, fmt.Errorf("bpf open map: %w", err)
	}
	o := newObject(fd)
	info, err := o.MapInfo()
	if err != nil {
		o.Close()
		return nil, MapInfo{}, err
	}
	return o, info, nil
}

// MapInfo queries the kernel for map metadata. The descriptor must refer to
// a map; kernel rejection is returned rather than inferred from a pin name.
func (o *Object) MapInfo() (MapInfo, error) {
	if o == nil {
		return MapInfo{}, ErrClosed
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.open {
		return MapInfo{}, ErrClosed
	}
	return mapInfo(o.fd)
}

// InspectMap queries a borrowed map descriptor. The caller must keep the
// descriptor open until this call returns. It does not take ownership.
func InspectMap(fd int) (MapInfo, error) { return mapInfo(fd) }
func mapInfo(fd int) (MapInfo, error) {
	kind, err := os.Readlink(fmt.Sprintf("/proc/self/fd/%d", fd))
	if err != nil {
		return MapInfo{}, fmt.Errorf("bpf identify map: %w", err)
	}
	if kind != "anon_inode:bpf-map" {
		return MapInfo{}, errors.New("bpf: descriptor is not a map")
	}
	info := struct {
		Type, ID, KeySize, ValueSize, MaxEntries, Flags uint32
		Name                                            [16]byte
		IfIndex, BTFVmlinuxValueTypeID                  uint32
		NetnsDev, NetnsIno                              uint64
		BTFID, BTFKeyTypeID, BTFValueTypeID, Padding    uint32
		MapExtra                                        uint64
	}{}
	attr := struct {
		FD, Size uint32
		Info     uint64
	}{uint32(fd), uint32(unsafe.Sizeof(info)), uint64(uintptr(unsafe.Pointer(&info)))}
	_, err = call(15, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	runtime.KeepAlive(&info)
	if err != nil {
		return MapInfo{}, fmt.Errorf("bpf map info: %w", err)
	}
	// Programs have a different info layout. Nonzero key/value sizes are
	// required because this API only supports ordinary fixed-size maps.
	if (info.KeySize == 0 || info.ValueSize == 0) && info.Type != 27 {
		return MapInfo{}, errors.New("bpf: object is not a supported fixed-size map")
	}
	return MapInfo{Type: info.Type, ID: info.ID, KeySize: info.KeySize, ValueSize: info.ValueSize, MaxEntries: info.MaxEntries, Flags: info.Flags, Name: logText(info.Name[:]), BTFID: info.BTFID, BTFKeyTypeID: info.BTFKeyTypeID, BTFValueTypeID: info.BTFValueTypeID, BTFVmlinuxValueTypeID: info.BTFVmlinuxValueTypeID}, nil
}

// Lookup reads a fixed-size map value. key and value must exactly match the
// kernel sizes. Per-CPU maps are refused because values require CPU padding.
func (o *Object) Lookup(key, value []byte) error { return o.mapOp(1, key, value, 0) }

// Update writes a fixed-size map value. flags uses Linux BPF_ANY (0),
// BPF_NOEXIST (1), or BPF_EXIST (2).
func (o *Object) Update(key, value []byte, flags uint64) error {
	if flags > 2 {
		return errors.New("bpf: invalid update flags")
	}
	return o.mapOp(2, key, value, flags)
}
func (o *Object) mapOp(cmd uintptr, key, value []byte, flags uint64) error {
	if o == nil {
		return ErrClosed
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.open {
		return ErrClosed
	}
	info, err := mapInfo(o.fd)
	if err != nil {
		return err
	}
	switch info.Type {
	case 1, 2, 9, 11:
	case 7:
		if cmd != 1 {
			return errors.New("bpf: stack map update refused")
		}
	default:
		return errors.New("bpf: unsupported map value layout")
	}
	if uint64(len(key)) != uint64(info.KeySize) || uint64(len(value)) != uint64(info.ValueSize) {
		return errors.New("bpf: map key or value size mismatch")
	}
	attr := struct {
		FD, Padding       uint32
		Key, Value, Flags uint64
	}{uint32(o.fd), 0, pointer(key), pointer(value), flags}
	_, err = call(cmd, unsafe.Pointer(&attr), unsafe.Sizeof(attr))
	runtime.KeepAlive(key)
	runtime.KeepAlive(value)
	if err != nil {
		return fmt.Errorf("bpf map operation: %w", err)
	}
	return nil
}
