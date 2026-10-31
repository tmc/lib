package model

import (
	"bytes"
	"encoding/binary"
	"fmt"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

// A MappedSafetensors is a safetensors file held as one byte slice, usually
// a read-only memory mapping. [MappedSafetensors.TensorBytes] returns
// subslices of it, so packing reads tensor bytes in place with no
// intermediate heap copy.
//
// On Grace and Vera systems the same mapping, registered with
// cuMemHostRegister, is what [PackedTensorBundle.SourceTMADesc] can address
// directly. This package does not register memory or touch the device.
type MappedSafetensors struct {
	*safetensors.File
	data  []byte
	base  int64
	unmap func() error
}

// MapSafetensors maps the named file read-only. On Linux and Darwin it uses
// mmap; elsewhere it reads the file into memory.
func MapSafetensors(name string) (*MappedSafetensors, error) {
	data, unmap, err := mapFile(name)
	if err != nil {
		return nil, fmt.Errorf("map safetensors: %w", err)
	}
	m, err := ViewSafetensors(data)
	if err != nil {
		unmap()
		return nil, fmt.Errorf("map safetensors %s: %w", name, err)
	}
	m.unmap = unmap
	return m, nil
}

// ViewSafetensors parses data, a complete safetensors file, without copying
// it. data must not be modified while the view is in use.
func ViewSafetensors(data []byte) (*MappedSafetensors, error) {
	f, err := safetensors.NewFile(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	base := 8 + int64(binary.LittleEndian.Uint64(data))
	return &MappedSafetensors{File: f, data: data, base: base}, nil
}

// TensorBytes returns the named tensor's bytes as a subslice of the mapping.
// The slice must not be written and is invalid after Close.
func (m *MappedSafetensors) TensorBytes(name string) ([]byte, error) {
	info, ok := m.Info(name)
	if !ok {
		return nil, fmt.Errorf("tensor bytes: no tensor %q", name)
	}
	start := m.base + info.Offset
	return m.data[start : start+info.Size : start+info.Size], nil
}

// Close releases the mapping.
func (m *MappedSafetensors) Close() error {
	if m.unmap == nil {
		return nil
	}
	u := m.unmap
	m.unmap, m.data = nil, nil
	return u()
}

// PackMapped packs the named rank-2 tensor of m directly from the mapping.
func (l *StreamLoader) PackMapped(m *MappedSafetensors, name string) (PackedTensorBundle, error) {
	info, ok := m.Info(name)
	if !ok {
		return PackedTensorBundle{}, fmt.Errorf("pack mapped: no tensor %q", name)
	}
	if len(info.Shape) != 2 {
		return PackedTensorBundle{}, fmt.Errorf("pack mapped %s: shape %v is not rank 2", name, info.Shape)
	}
	raw, err := m.TensorBytes(name)
	if err != nil {
		return PackedTensorBundle{}, fmt.Errorf("pack mapped: %w", err)
	}
	return l.PackTensorBytes(name, info.Shape[0], info.Shape[1], info.Dtype, raw)
}

// A StreamStage is the work one [StreamStep] does.
type StreamStage uint8

const (
	// StageCopy moves a chunk from the mapping into a staging slot on
	// stream 0 (host-to-device or C2C).
	StageCopy StreamStage = iota + 1
	// StageRepack runs the repack kernel over a staged chunk on stream 1.
	StageRepack
)

func (s StreamStage) String() string {
	switch s {
	case StageCopy:
		return "copy"
	case StageRepack:
		return "repack"
	}
	return fmt.Sprintf("StreamStage(%d)", uint8(s))
}

// A StreamStep is one operation of a [StreamPlan]. It may start only after
// every step in Wait has finished.
type StreamStep struct {
	Stage  StreamStage
	Stream int // 0 for copies, 1 for repacks
	Slot   int // staging slot
	Tensor string
	// Offset and Bytes locate the chunk within the tensor's data.
	Offset, Bytes int64
	Wait          []int
}

// A StreamPlan overlaps copying checkpoint chunks into a ring of staging
// slots on stream 0 with repacking them on stream 1. It is a schedule
// description: event record and wait pairs on two CUDA streams implement
// it, and [StreamPlan.Validate] and [StreamPlan.Makespan] model it on the
// host.
type StreamPlan struct {
	Slots     int
	SlotBytes int64
	Steps     []StreamStep
}

// PlanStream splits the tensors into chunks of whole rows that fit
// slotBytes and schedules them across slots staging slots. A rank-2 tensor
// is split at multiples of 8 rows; any other tensor must fit one slot.
func PlanStream(tensors []safetensors.TensorInfo, slotBytes int64, slots int) (StreamPlan, error) {
	if slots < 2 {
		return StreamPlan{}, fmt.Errorf("plan stream: need at least 2 slots to overlap, have %d", slots)
	}
	if slotBytes <= 0 {
		return StreamPlan{}, fmt.Errorf("plan stream: slot bytes %d is not positive", slotBytes)
	}
	p := StreamPlan{Slots: slots, SlotBytes: slotBytes}
	lastRepack := make([]int, slots)
	for i := range lastRepack {
		lastRepack[i] = -1
	}
	n := 0
	for _, t := range tensors {
		chunk := t.Size
		if len(t.Shape) == 2 && t.Shape[0] > 0 {
			rowBytes := t.Size / int64(t.Shape[0])
			rowsPer := slotBytes / rowBytes / 8 * 8
			if rowsPer == 0 {
				return StreamPlan{}, fmt.Errorf("plan stream: 8 rows of %s (%d bytes) exceed slot bytes %d", t.Name, 8*rowBytes, slotBytes)
			}
			chunk = min(t.Size, rowsPer*rowBytes)
		}
		if chunk > slotBytes {
			return StreamPlan{}, fmt.Errorf("plan stream: %s (%d bytes) exceeds slot bytes %d", t.Name, t.Size, slotBytes)
		}
		for off := int64(0); off < t.Size; off += chunk {
			slot := n % slots
			cp := StreamStep{Stage: StageCopy, Stream: 0, Slot: slot, Tensor: t.Name, Offset: off, Bytes: min(chunk, t.Size-off)}
			if lastRepack[slot] >= 0 {
				cp.Wait = []int{lastRepack[slot]}
			}
			p.Steps = append(p.Steps, cp)
			rp := cp
			rp.Stage, rp.Stream, rp.Wait = StageRepack, 1, []int{len(p.Steps) - 1}
			p.Steps = append(p.Steps, rp)
			lastRepack[slot] = len(p.Steps) - 1
			n++
		}
	}
	return p, p.Validate()
}

// Validate checks that the plan is race-free: every repack waits for the
// copy that filled its slot with its chunk, every copy into a slot waits
// for the repack that last read it, waits point backwards, and each
// tensor's chunks tile its data.
func (p StreamPlan) Validate() error {
	filled := make(map[int]int)  // slot -> copy step whose data it holds
	reading := make(map[int]int) // slot -> last repack step
	covered := make(map[string]int64)
	for i, s := range p.Steps {
		for _, w := range s.Wait {
			if w < 0 || w >= i {
				return fmt.Errorf("stream plan: step %d waits on step %d, which is not earlier", i, w)
			}
		}
		if s.Slot < 0 || s.Slot >= p.Slots {
			return fmt.Errorf("stream plan: step %d uses slot %d of %d", i, s.Slot, p.Slots)
		}
		if s.Bytes <= 0 || s.Bytes > p.SlotBytes {
			return fmt.Errorf("stream plan: step %d moves %d bytes into a %d-byte slot", i, s.Bytes, p.SlotBytes)
		}
		switch s.Stage {
		case StageCopy:
			if s.Stream != 0 {
				return fmt.Errorf("stream plan: copy step %d on stream %d, want 0", i, s.Stream)
			}
			if r, ok := reading[s.Slot]; ok && !waitsOn(p, i, r) {
				return fmt.Errorf("stream plan: copy step %d overwrites slot %d before repack step %d finishes", i, s.Slot, r)
			}
			if s.Offset != covered[s.Tensor] {
				return fmt.Errorf("stream plan: copy step %d starts %s at %d, want %d", i, s.Tensor, s.Offset, covered[s.Tensor])
			}
			covered[s.Tensor] += s.Bytes
			filled[s.Slot] = i
		case StageRepack:
			if s.Stream != 1 {
				return fmt.Errorf("stream plan: repack step %d on stream %d, want 1", i, s.Stream)
			}
			c, ok := filled[s.Slot]
			if !ok || p.Steps[c].Tensor != s.Tensor || p.Steps[c].Offset != s.Offset || !waitsOn(p, i, c) {
				return fmt.Errorf("stream plan: repack step %d does not wait for the copy of its chunk", i)
			}
			reading[s.Slot] = i
		default:
			return fmt.Errorf("stream plan: step %d has stage %v", i, s.Stage)
		}
	}
	return nil
}

// waitsOn reports whether step i is ordered after step j, either by an
// explicit wait chain or by program order on the same stream.
func waitsOn(p StreamPlan, i, j int) bool {
	seen := make(map[int]bool)
	var walk func(k int) bool
	walk = func(k int) bool {
		if k == j {
			return true
		}
		if seen[k] || k < j {
			return false
		}
		seen[k] = true
		for _, w := range p.Steps[k].Wait {
			if walk(w) {
				return true
			}
		}
		// Steps on one stream run in issue order.
		for prev := k - 1; prev >= j; prev-- {
			if p.Steps[prev].Stream == p.Steps[k].Stream {
				return walk(prev)
			}
		}
		return false
	}
	return walk(i)
}

// Makespan models the plan's duration in seconds when copies run at
// copyGBps and repacks at repackGBps, each stream executing its steps in
// order and honoring waits. It is a model, not a measurement.
func (p StreamPlan) Makespan(copyGBps, repackGBps float64) float64 {
	finish := make([]float64, len(p.Steps))
	var streamFree [2]float64
	for i, s := range p.Steps {
		start := streamFree[s.Stream]
		for _, w := range s.Wait {
			start = max(start, finish[w])
		}
		rate := copyGBps
		if s.Stage == StageRepack {
			rate = repackGBps
		}
		finish[i] = start + float64(s.Bytes)/(rate*1e9)
		streamFree[s.Stream] = finish[i]
	}
	var end float64
	for _, f := range finish {
		end = max(end, f)
	}
	return end
}
