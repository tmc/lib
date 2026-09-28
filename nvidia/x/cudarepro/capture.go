package cudarepro

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// Argument describes a scalar value or a device slice. For a buffer, Address
// and Size describe the slice, not necessarily its whole allocation. Value is
// the host ABI representation of a scalar; embedded pointers are unsupported.
type Argument struct {
	Name    string
	Type    string
	Buffer  bool
	Address uint64
	Size    uint64
	Value   []byte
}

// Allocation is one complete backing allocation at capture time.
type Allocation struct {
	Address uint64
	Data    []byte
}

// Launch is a self-contained isolated launch. ID identifies this capture,
// not a CUPTI launch. Stream is diagnostic metadata, never reused for replay.
// Sources and Options are optional provenance; Artifact is the executed image.
type Launch struct {
	Version     int
	ID          string
	Artifact    *cudakernel.Artifact
	Grid, Block [3]uint32
	SharedBytes uint32
	Stream      uint64
	CallSite    string
	Arguments   []Argument
	Allocations []Allocation
	Sources     map[string]string
	Options     []string
}

// Capture snapshots one launch without submitting it. limit is the maximum
// total allocation bytes to copy, and must be positive. Only native CUBINs are
// accepted so replay does not silently recompile different code. Artifact.Schema
// must be the complete, accurate ABI supplied by a trusted producer. The caller
// must keep all allocations live and exclude concurrent writes until it returns.
func Capture(stream *cuda.Stream, artifact *cudakernel.Artifact, grid, block [3]uint32, shared uint32, limit uint64, args ...Argument) (*Launch, error) {
	if stream == nil || stream.Device() == nil {
		return nil, fmt.Errorf("cudarepro: nil stream")
	}
	if artifact == nil || artifact.Kind != cudakernel.KindCUBIN {
		return nil, fmt.Errorf("cudarepro: capture requires a native CUBIN")
	}
	if err := artifact.Validate(); err != nil {
		return nil, err
	}
	if limit == 0 {
		return nil, fmt.Errorf("cudarepro: zero capture limit")
	}
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return nil, err
	}
	l := &Launch{Version: 1, ID: hex.EncodeToString(id[:]), Artifact: artifact, Grid: grid, Block: block, SharedBytes: shared, Arguments: append([]Argument(nil), args...)}
	_, file, line, _ := runtime.Caller(1)
	l.CallSite = fmt.Sprintf("%s:%d", file, line)
	err := stream.Device().WithContext(func() error {
		return stream.WithHandle(func(h driver.Stream) error {
			status, err := h.IsCapturing()
			if err != nil {
				return err
			}
			if status != driver.StreamCaptureStatusNone {
				return fmt.Errorf("cudarepro: cannot snapshot during graph capture")
			}
			if err := h.Synchronize(); err != nil {
				return err
			}
			l.Stream = uint64(h)
			var total uint64
			seen := make(map[uint64]bool)
			for i, a := range l.Arguments {
				l.Arguments[i].Value = append([]byte(nil), a.Value...)
				if !a.Buffer || a.Address == 0 {
					continue
				}
				base, size, err := driver.DevicePtr(a.Address).MemGetAddressRange()
				if err != nil {
					return fmt.Errorf("cudarepro: %s allocation: %w", a.Name, err)
				}
				b, n := uint64(base), uint64(size)
				if a.Address < b || a.Address-b > n || a.Size > n-(a.Address-b) {
					return fmt.Errorf("cudarepro: %s exceeds allocation", a.Name)
				}
				if seen[b] {
					continue
				}
				if n > limit-total || n > uint64(int(^uint(0)>>1)) {
					return fmt.Errorf("cudarepro: allocation exceeds capture limit")
				}
				data := make([]byte, int(n))
				if n != 0 {
					var pin runtime.Pinner
					pin.Pin(&data[0])
					err = base.CopyToHostPtr(unsafe.Pointer(&data[0]), size)
					pin.Unpin()
					if err != nil {
						return err
					}
				}
				seen[b] = true
				total += n
				l.Allocations = append(l.Allocations, Allocation{b, data})
			}
			return nil
		})
	})
	if err != nil {
		return nil, err
	}
	if err := l.Validate(); err != nil {
		return nil, err
	}
	return l, nil
}

// Validate checks the image digest, launch dimensions and allocation mapping.
func (l *Launch) Validate() error {
	if l == nil || l.Version != 1 || l.ID == "" {
		return fmt.Errorf("cudarepro: invalid capture version or identity")
	}
	if l.Artifact == nil || l.Artifact.Kind != cudakernel.KindCUBIN {
		return fmt.Errorf("cudarepro: native CUBIN required")
	}
	if err := l.Artifact.Validate(); err != nil {
		return err
	}
	for i := range l.Grid {
		if l.Grid[i] == 0 || l.Block[i] == 0 {
			return fmt.Errorf("cudarepro: zero launch dimension")
		}
	}
	for i, b := range l.Allocations {
		n := uint64(len(b.Data))
		if b.Address == 0 || n == 0 || b.Address+n < b.Address {
			return fmt.Errorf("cudarepro: invalid allocation")
		}
		for _, other := range l.Allocations[:i] {
			if b.Address < other.Address+uint64(len(other.Data)) && other.Address < b.Address+n {
				return fmt.Errorf("cudarepro: overlapping allocation records")
			}
		}
	}
	if len(l.Artifact.Schema.Params) != len(l.Arguments) {
		return fmt.Errorf("cudarepro: argument count differs from captured ABI")
	}
	for i, a := range l.Arguments {
		p := l.Artifact.Schema.Params[i]
		size := len(a.Value)
		if a.Buffer {
			size = 8
		}
		if p.ByteWidth != size || p.IsPointer != a.Buffer || p.Name != a.Name {
			return fmt.Errorf("cudarepro: argument %s differs from captured ABI", a.Name)
		}
	}
	names := make(map[string]bool)
	for _, a := range l.Arguments {
		if a.Name == "" || names[a.Name] {
			return fmt.Errorf("cudarepro: missing or repeated argument name")
		}
		names[a.Name] = true
		if !a.Buffer {
			if len(a.Value) == 0 || a.Address != 0 || a.Size != 0 {
				return fmt.Errorf("cudarepro: invalid scalar %s", a.Name)
			}
			continue
		}
		if len(a.Value) != 0 {
			return fmt.Errorf("cudarepro: buffer %s has scalar data", a.Name)
		}
		if a.Address == 0 && a.Size == 0 {
			continue
		}
		if l.allocation(a.Address, a.Size) < 0 {
			return fmt.Errorf("cudarepro: unmapped buffer %s", a.Name)
		}
	}
	return nil
}

func (l *Launch) allocation(address, size uint64) int {
	for i, b := range l.Allocations {
		n := uint64(len(b.Data))
		if address >= b.Address && address-b.Address < n && size <= n-(address-b.Address) {
			return i
		}
	}
	return -1
}
