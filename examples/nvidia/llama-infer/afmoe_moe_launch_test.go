package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestAFMoEMemoryRanges(t *testing.T) {
	for _, test := range []struct {
		name string
		ptr  driver.DevicePtr
		size uintptr
	}{
		{"null", 0, 4},
		{"empty", 4, 0},
		{"overflow", driver.DevicePtr(^uint64(0) - 2), 4},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := afmoeMemory(test.name, test.ptr, test.size); err == nil {
				t.Fatal("accepted invalid range")
			}
		})
	}
	a, err := afmoeMemory("a", 100, 20)
	if err != nil {
		t.Fatal(err)
	}
	b, err := afmoeMemory("b", 120, 20)
	if err != nil {
		t.Fatal(err)
	}
	if err := afmoeDisjointMemory([]afmoeMemoryRange{a, b}); err != nil {
		t.Fatalf("adjacent ranges overlap: %v", err)
	}
	for _, overlap := range []afmoeMemoryRange{
		{name: "partial", start: 119, end: 140},
		{name: "contained", start: 101, end: 110},
		{name: "contains", start: 99, end: 121},
		{name: "identical", start: 100, end: 120},
	} {
		if err := afmoeDisjointMemory([]afmoeMemoryRange{a, overlap}); err == nil {
			t.Errorf("accepted %s alias", overlap.name)
		}
	}
}

func TestAFMoEMoERejectsUnpreparedLaunch(t *testing.T) {
	dev := new(cuda.Device)
	m := &afmoeMoEKernels{mod: 1, dev: dev, routerLinearFn: 1, routeFn: 2, gateUpFn: 3, downFn: 4, selectedGateFn: 5, selectedDownFn: 6, combineFn: 7, clearFaultFn: 8}
	emitted := false
	err := m.emitMoE(&afmoeExpertTables{dev: dev, hidden: 2048, intermediate: 1024}, deviceWeight{}, cudamem.Slice[float32]{}, deviceWeight{}, deviceWeight{}, deviceWeight{}, &afmoeMoEBuffers{}, 8, true, 1e-20, 2.826, false,
		func(*cudalaunch.Launcher, ...any) error { emitted = true; return nil })
	if err == nil || emitted {
		t.Fatal("unprepared layer emitted a kernel")
	}
	if _, err := newAFMoEExpertTables(nil, nil, nil, nil, 2048, 1024); err == nil {
		t.Error("accepted missing allocation device")
	}
	if _, err := newAFMoEExpertTables(dev, make([]deviceWeight, 32), make([]deviceWeight, 32), make([]deviceWeight, 32), 2048, 1024); err == nil {
		t.Error("silently accepted 32 experts for 128-expert tables")
	}
	if err := (*afmoeMoEBuffers)(nil).checkFault(); err == nil {
		t.Error("accepted missing fault result")
	}
}

func TestAFMoEDenseMLPRejectsUnpreparedLaunch(t *testing.T) {
	emitted := false
	emit := func(*cudalaunch.Launcher, ...any) error { emitted = true; return nil }
	for _, test := range []struct {
		name string
		m    *afmoeMoEKernels
		h, i int
	}{
		{"missing module", nil, 2048, 6144},
		{"negative dimensions", &afmoeMoEKernels{mod: 1, dev: new(cuda.Device), gateUpFn: 1, downFn: 2}, -1, 6144},
		{"missing matrices", &afmoeMoEKernels{mod: 1, dev: new(cuda.Device), gateUpFn: 1, downFn: 2}, 2048, 6144},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := test.m.emitDenseMLP(deviceWeight{}, deviceWeight{}, deviceWeight{}, cudamem.Slice[float32]{}, cudamem.Slice[float32]{}, cudamem.Slice[float32]{}, test.h, test.i, emit)
			if err == nil || emitted {
				t.Fatal("unprepared dense layer emitted a kernel")
			}
		})
	}
}
