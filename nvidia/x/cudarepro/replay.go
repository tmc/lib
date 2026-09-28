package cudarepro

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Write validates and writes a capture as JSON, including allocation contents.
func (l *Launch) Write(w io.Writer) error {
	if err := l.Validate(); err != nil {
		return err
	}
	return json.NewEncoder(w).Encode(l)
}

// Read reads a capture with an explicit encoded byte limit. Use captures from
// trusted producers only: replay executes their native GPU code.
func Read(r io.Reader, limit int64) (*Launch, error) {
	if limit <= 0 || limit == int64(^uint64(0)>>1) {
		return nil, fmt.Errorf("cudarepro: invalid read limit")
	}
	b, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("cudarepro: capture exceeds read limit")
	}
	var l Launch
	if err := json.Unmarshal(b, &l); err != nil {
		return nil, err
	}
	return &l, l.Validate()
}

// Replay runs an isolated launch on fresh allocations and returns each buffer
// argument's output bytes. before, when nonnil, receives relocated addresses
// before submission; save that record to attribute a sanitizer report from this
// replay. It must not mutate the record. Aliases retain their original offsets.
// Replay does not recreate concurrent work, mutable globals or external state.
func (l *Launch) Replay(dev *cuda.Device, before func(*Launch) error) (result map[string][]byte, err error) {
	if err := l.Validate(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudarepro: nil device")
	}
	// Supported hosts and CUDA scalar argument ABI are little-endian, 64-bit.
	if unsafe.Sizeof(uintptr(0)) != 8 {
		return nil, fmt.Errorf("cudarepro: replay requires a 64-bit host")
	}
	raw, err := l.Artifact.Load(dev)
	if err != nil {
		if raw != 0 {
			err = errors.Join(err, dev.WithContext(raw.Close))
		}
		return nil, err
	}
	mod, err := cuda.OwnModule(dev, raw)
	if err != nil {
		return nil, errors.Join(err, dev.WithContext(raw.Close))
	}
	defer func() { err = errors.Join(err, mod.Close()) }()
	kernel, err := mod.Function(l.Artifact.EntryPoint)
	if err != nil {
		return nil, err
	}

	err = dev.WithContext(func() error {
		f, err := raw.Function(l.Artifact.EntryPoint)
		if err != nil {
			return err
		}
		for i, a := range l.Arguments {
			_, size, err := f.ParamInfo(uint(i))
			if err != nil {
				return err
			}
			got := len(a.Value)
			if a.Buffer {
				got = 8
			}
			if size != uint(got) {
				return fmt.Errorf("cudarepro: argument %s has %d bytes, artifact requires %d", a.Name, got, size)
			}
		}

		return nil
	})
	if err != nil {
		return nil, err
	}
	stream, err := dev.NewStream()
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, stream.Close()) }()
	buffers := make([]cudamem.Slice[byte], len(l.Allocations))
	defer func() {
		for _, b := range buffers {
			err = errors.Join(err, b.Free())
		}
	}()
	defer func() { err = errors.Join(err, stream.Synchronize(context.Background())) }()
	relocated := *l
	relocated.Arguments = append([]Argument(nil), l.Arguments...)
	relocated.Allocations = append([]Allocation(nil), l.Allocations...)
	relocated.Stream = uint64(stream.Raw())
	for i, b := range l.Allocations {
		buffers[i], err = cudamem.Alloc[byte](dev, len(b.Data))
		if err != nil {
			return nil, err
		}
		if err = buffers[i].CopyFromHost(b.Data); err != nil {
			return nil, err
		}
		relocated.Allocations[i].Address = uint64(buffers[i].Ptr())
	}
	values := make([][]byte, len(l.Arguments))
	ptrs := make([]unsafe.Pointer, len(values))
	for i, a := range l.Arguments {
		if a.Buffer {
			address := uint64(0)
			if a.Address != 0 {
				j := l.allocation(a.Address, a.Size)
				address = uint64(buffers[j].Ptr()) + (a.Address - l.Allocations[j].Address)
			}
			relocated.Arguments[i].Address = address
			values[i] = make([]byte, 8)
			binary.LittleEndian.PutUint64(values[i], address)
		} else {
			values[i] = append([]byte(nil), a.Value...)
		}
		ptrs[i] = unsafe.Pointer(&values[i][0])
	}
	if before != nil {
		if err := before(&relocated); err != nil {
			return nil, err
		}
	}
	if err = stream.WithHandle(func(h driver.Stream) error { return kernel.Launch(l.Grid, l.Block, l.SharedBytes, h, ptrs) }); err != nil {
		return nil, err
	}
	if err = stream.Synchronize(context.Background()); err != nil {
		return nil, err
	}
	result = make(map[string][]byte)
	for _, a := range l.Arguments {
		if !a.Buffer {
			continue
		}
		data := make([]byte, int(a.Size))
		if a.Size != 0 {
			j := l.allocation(a.Address, a.Size)
			offset := int(a.Address - l.Allocations[j].Address)
			if err = buffers[j].Slice(offset, offset+len(data)).CopyToHost(data); err != nil {
				return nil, err
			}
		}
		result[a.Name] = data
	}
	return result, nil
}
