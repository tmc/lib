package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// afmoeExpertTables owns three device pointer tables and holds leases on their
// original BF16 matrices. Closing tables before weights prevents an indirect
// kernel argument from outliving its underlying allocation.
type afmoeExpertTables struct {
	gate, up, down cudamem.Slice[uint64]
	dev            *cuda.Device
	hidden         int
	intermediate   int
	release        []func()
	matrices       []afmoeMemoryRange
}

func newAFMoEExpertTables(dev *cuda.Device, gate, up, down []deviceWeight, hidden, intermediate int) (tables *afmoeExpertTables, err error) {
	if dev == nil || hidden < 1 || hidden > 32768 || intermediate < 1 || intermediate > 32768 {
		return nil, fmt.Errorf("afmoe expert tables: invalid device or dimensions")
	}
	if len(gate) != 128 || len(up) != 128 || len(down) != 128 {
		return nil, fmt.Errorf("afmoe expert tables: require 128 gate, up and down matrices")
	}
	t := &afmoeExpertTables{dev: dev, hidden: hidden, intermediate: intermediate}
	defer func() {
		if err != nil {
			if cerr := t.Close(); cerr != nil {
				err = errors.Join(err, cerr)
				tables = t
			}
		}
	}()
	for i, item := range []struct {
		weights []deviceWeight
		dst     *cudamem.Slice[uint64]
	}{
		{gate, &t.gate}, {up, &t.up}, {down, &t.down},
	} {
		addresses := make([]uint64, 128)
		for expert, w := range item.weights {
			b, err := afmoeBF16Matrix(w, dev, hidden*intermediate)
			if err != nil {
				return nil, fmt.Errorf("afmoe expert tables: table %d expert %d: %w", i, expert, err)
			}
			release, err := b.Borrow(dev)
			if err != nil {
				return nil, fmt.Errorf("afmoe expert tables: lease expert %d: %w", expert, err)
			}
			t.release = append(t.release, release)
			addresses[expert] = uint64(b.Ptr())
			r, err := afmoeMemory(fmt.Sprintf("expert table %d matrix %d", i, expert), b.Ptr(), b.SizeBytes())
			if err != nil {
				return nil, err
			}
			t.matrices = append(t.matrices, r)
		}
		if *item.dst, err = cudamem.Alloc[uint64](dev, 128); err != nil {
			return nil, fmt.Errorf("afmoe expert tables: allocate table: %w", err)
		}
		if err := item.dst.CopyFromHost(addresses); err != nil {
			return nil, fmt.Errorf("afmoe expert tables: upload table: %w", err)
		}
	}
	if err := afmoeDisjointMemory(t.matrices); err != nil {
		return nil, err
	}
	return t, nil
}

func (t *afmoeExpertTables) Close() error {
	if t == nil {
		return nil
	}
	err := errors.Join(freeOwned(&t.gate), freeOwned(&t.up), freeOwned(&t.down))
	if err != nil {
		return fmt.Errorf("close afmoe expert tables: %w", err)
	}
	for _, release := range t.release {
		release()
	}
	*t = afmoeExpertTables{}
	return nil
}

func afmoeBF16Matrix(w deviceWeight, dev *cuda.Device, elements int) (cudamem.Slice[uint16], error) {
	b, ok := w.buf.(cudamem.Slice[uint16])
	if !ok || w.dtype != weightBF16 || elements < 1 || b.Len() != elements || b.Ptr() == 0 || b.Device() != dev {
		return cudamem.Slice[uint16]{}, fmt.Errorf("invalid BF16 matrix, size or device")
	}
	return b, nil
}

// emitDenseMLP emits the first dense layers' SwiGLU, with BF16 rounding after
// each linear, the SiLU and the gate/up multiplication. It uses the dense
// intermediate width (Mini6144), independently of the routed width (1024).
func (m *afmoeMoEKernels) emitDenseMLP(gate, up, down deviceWeight, input, activation, output cudamem.Slice[float32], hidden, intermediate int, emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.dev == nil || m.gateUpFn == 0 || m.downFn == 0 || emit == nil || hidden < 1 || hidden > 32768 || intermediate < 1 || intermediate > 32768 {
		return fmt.Errorf("afmoe dense mlp: invalid module, emitter or dimensions")
	}
	var matrices [3]cudamem.Slice[uint16]
	var ranges []afmoeMemoryRange
	for i, w := range []deviceWeight{gate, up, down} {
		b, err := afmoeBF16Matrix(w, m.dev, hidden*intermediate)
		if err != nil {
			return fmt.Errorf("afmoe dense mlp matrix %d: %w", i, err)
		}
		matrices[i] = b
		r, err := afmoeMemory(fmt.Sprintf("dense matrix %d", i), b.Ptr(), b.SizeBytes())
		if err != nil {
			return err
		}
		ranges = append(ranges, r)
	}
	for _, item := range []struct {
		name string
		s    cudamem.Slice[float32]
		n    int
	}{
		{"dense input", input, hidden}, {"dense activation", activation, intermediate}, {"dense output", output, hidden},
	} {
		if item.s.Ptr() == 0 || item.s.Len() != item.n || item.s.Device() != m.dev {
			return fmt.Errorf("afmoe dense mlp: invalid %s shape or device", item.name)
		}
		r, err := afmoeMemory(item.name, item.s.Ptr(), item.s.SizeBytes())
		if err != nil {
			return err
		}
		ranges = append(ranges, r)
	}
	if err := afmoeDisjointMemory(ranges); err != nil {
		return err
	}
	if err := emit(cudalaunch.Kernel(m.gateUpFn).Grid(uint32(intermediate), 1, 1).Block(128, 1, 1), input, matrices[0], matrices[1], activation, int32(hidden), int32(intermediate)); err != nil {
		return fmt.Errorf("emit afmoe dense gate/up: %w", err)
	}
	if err := emit(cudalaunch.Kernel(m.downFn).Grid(uint32(hidden), 1, 1).Block(128, 1, 1), activation, matrices[2], output, int32(intermediate), int32(hidden)); err != nil {
		return fmt.Errorf("emit afmoe dense down: %w", err)
	}
	return nil
}

// afmoeMoEBuffers is scratch for one token. Float buffers store BF16 values,
// except Weights stores FP32 routing weights. Every buffer must be distinct
// and on the tables' device.
type afmoeMoEBuffers struct {
	Input, Logits, Weights cudamem.Slice[float32]
	SharedAct, SharedOut   cudamem.Slice[float32]
	RoutedAct, RoutedOut   cudamem.Slice[float32]
	Output                 cudamem.Slice[float32]
	IDs, Fault             cudamem.Slice[int32]
}

// emitMoE emits one ordered scalar-token shared+routed MLP. emit must retain
// order on a single stream (or equivalent graph dependencies). After execution
// and stream synchronization, checkFault must succeed before Output is used.
// Emission does not initialize CUDA or run any kernel by itself.
func (m *afmoeMoEKernels) emitMoE(t *afmoeExpertTables, router deviceWeight, bias cudamem.Slice[float32], sharedGate, sharedUp, sharedDown deviceWeight, b *afmoeMoEBuffers, topK int, normalize bool, epsilon, scale float32, wide bool, emit func(*cudalaunch.Launcher, ...any) error) error {
	if m == nil || m.mod == 0 || m.dev == nil || emit == nil || t == nil || b == nil {
		return fmt.Errorf("afmoe emit moe: missing module, tables, buffers or emitter")
	}
	if m.dev != t.dev || t.dev == nil || t.hidden < 1 || t.intermediate < 1 || topK < 1 || topK > 128 || !afmoeFinite(epsilon) || epsilon <= 0 || !afmoeFinite(scale) || scale <= 0 {
		return fmt.Errorf("afmoe emit moe: invalid dimensions, device or route parameters")
	}
	for _, fn := range []driver.Function{m.routerLinearFn, m.routeFn, m.gateUpFn, m.downFn, m.selectedGateFn, m.selectedDownFn, m.combineFn, m.clearFaultFn} {
		if fn == 0 {
			return fmt.Errorf("afmoe emit moe: missing kernel")
		}
	}
	h, inter := t.hidden, t.intermediate
	routerBuffer, err := afmoeBF16Matrix(router, t.dev, 128*h)
	if err != nil {
		return fmt.Errorf("afmoe emit moe router: %w", err)
	}
	gateBuffer, err := afmoeBF16Matrix(sharedGate, t.dev, h*inter)
	if err != nil {
		return fmt.Errorf("afmoe emit moe shared gate: %w", err)
	}
	upBuffer, err := afmoeBF16Matrix(sharedUp, t.dev, h*inter)
	if err != nil {
		return fmt.Errorf("afmoe emit moe shared up: %w", err)
	}
	downBuffer, err := afmoeBF16Matrix(sharedDown, t.dev, h*inter)
	if err != nil {
		return fmt.Errorf("afmoe emit moe shared down: %w", err)
	}
	ranges := append([]afmoeMemoryRange(nil), t.matrices...)
	add := func(name string, ptr driver.DevicePtr, elements int, bytes uintptr, device *cuda.Device, expected int) error {
		if ptr == 0 || elements != expected || device != t.dev {
			return fmt.Errorf("afmoe emit moe: invalid %s size or device", name)
		}
		r, err := afmoeMemory(name, ptr, bytes)
		if err != nil {
			return err
		}
		ranges = append(ranges, r)
		return nil
	}
	for _, item := range []struct {
		name string
		s    cudamem.Slice[float32]
		n    int
	}{
		{"input", b.Input, h}, {"logits", b.Logits, 128}, {"route weights", b.Weights, topK},
		{"shared activation", b.SharedAct, inter}, {"shared output", b.SharedOut, h},
		{"routed activation", b.RoutedAct, topK * inter}, {"routed output", b.RoutedOut, topK * h},
		{"output", b.Output, h}, {"router bias", bias, 128},
	} {
		if err := add(item.name, item.s.Ptr(), item.s.Len(), item.s.SizeBytes(), item.s.Device(), item.n); err != nil {
			return err
		}
	}
	for _, item := range []struct {
		name string
		s    cudamem.Slice[int32]
		n    int
	}{
		{"route IDs", b.IDs, topK}, {"fault", b.Fault, 1},
	} {
		if err := add(item.name, item.s.Ptr(), item.s.Len(), item.s.SizeBytes(), item.s.Device(), item.n); err != nil {
			return err
		}
	}
	for _, item := range []struct {
		name string
		s    cudamem.Slice[uint64]
	}{
		{"gate table", t.gate}, {"up table", t.up}, {"down table", t.down},
	} {
		if err := add(item.name, item.s.Ptr(), item.s.Len(), item.s.SizeBytes(), item.s.Device(), 128); err != nil {
			return err
		}
	}
	for _, item := range []struct {
		name string
		s    cudamem.Slice[uint16]
	}{
		{"router matrix", routerBuffer}, {"shared gate", gateBuffer},
		{"shared up", upBuffer}, {"shared down", downBuffer},
	} {
		r, err := afmoeMemory(item.name, item.s.Ptr(), item.s.SizeBytes())
		if err != nil {
			return err
		}
		ranges = append(ranges, r)
	}
	if err := afmoeDisjointMemory(ranges); err != nil {
		return err
	}
	norm, accumulation := int32(0), int32(0)
	if normalize {
		norm = 1
	}
	if wide {
		accumulation = 1
	}
	for _, launch := range []struct {
		l    *cudalaunch.Launcher
		args []any
	}{
		{cudalaunch.Kernel(m.clearFaultFn).Grid(1, 1, 1).Block(1, 1, 1), []any{b.Fault}},
		{cudalaunch.Kernel(m.routerLinearFn).Grid(128, 1, 1).Block(128, 1, 1), []any{b.Input, routerBuffer, b.Logits, int32(h), int32(128)}},
		{cudalaunch.Kernel(m.routeFn).Grid(1, 1, 1).Block(1, 1, 1), []any{b.Logits, bias, b.IDs, b.Weights, b.Fault, int32(128), int32(topK), norm, epsilon, scale}},
		{cudalaunch.Kernel(m.gateUpFn).Grid(uint32(inter), 1, 1).Block(128, 1, 1), []any{b.Input, gateBuffer, upBuffer, b.SharedAct, int32(h), int32(inter)}},
		{cudalaunch.Kernel(m.downFn).Grid(uint32(h), 1, 1).Block(128, 1, 1), []any{b.SharedAct, downBuffer, b.SharedOut, int32(inter), int32(h)}},
		{cudalaunch.Kernel(m.selectedGateFn).Grid(uint32(inter), uint32(topK), 1).Block(128, 1, 1), []any{b.Input, t.gate, t.up, b.IDs, b.RoutedAct, int32(h), int32(inter), int32(topK), b.Fault}},
		{cudalaunch.Kernel(m.selectedDownFn).Grid(uint32(h), uint32(topK), 1).Block(128, 1, 1), []any{b.RoutedAct, t.down, b.IDs, b.RoutedOut, int32(inter), int32(h), int32(topK), b.Fault}},
		{cudalaunch.Kernel(m.combineFn).Grid(uint32((h+127)/128), 1, 1).Block(128, 1, 1), []any{b.SharedOut, b.RoutedOut, b.IDs, b.Weights, b.Output, int32(h), int32(topK), accumulation, b.Fault}},
	} {
		if err := emit(launch.l, launch.args...); err != nil {
			return fmt.Errorf("emit afmoe moe kernel: %w", err)
		}
	}
	return nil
}

func (b *afmoeMoEBuffers) checkFault() error {
	if b == nil || b.Fault.Len() != 1 || b.Fault.Ptr() == 0 {
		return fmt.Errorf("afmoe moe: missing fault buffer")
	}
	var fault [1]int32
	if err := b.Fault.CopyToHost(fault[:]); err != nil {
		return fmt.Errorf("read afmoe moe fault: %w", err)
	}
	if fault[0] != 0 {
		return fmt.Errorf("afmoe moe device fault: %d", fault[0])
	}
	return nil
}

type afmoeMemoryRange struct {
	name       string
	start, end uint64
}

func afmoeMemory(name string, ptr driver.DevicePtr, bytes uintptr) (afmoeMemoryRange, error) {
	start := uint64(ptr)
	if start == 0 || bytes == 0 || uint64(bytes) > ^uint64(0)-start {
		return afmoeMemoryRange{}, fmt.Errorf("afmoe memory: invalid %s pointer or size", name)
	}
	return afmoeMemoryRange{name: name, start: start, end: start + uint64(bytes)}, nil
}

func afmoeDisjointMemory(ranges []afmoeMemoryRange) error {
	for i, a := range ranges {
		for _, b := range ranges[:i] {
			if a.start < b.end && b.start < a.end {
				return fmt.Errorf("afmoe memory: %s overlaps %s", a.name, b.name)
			}
		}
	}
	return nil
}
