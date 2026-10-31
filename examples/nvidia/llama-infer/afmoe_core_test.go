package main

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func afmoeCoreTestSlice[T any](t *testing.T, ptr driver.DevicePtr, n int) cudamem.Slice[T] {
	t.Helper()
	s, ok := fakeDeviceSliceLen[T](ptr, n)
	if !ok {
		t.Fatal("cannot construct host-only fake device slice")
	}
	return s
}

func afmoeCoreTestWeight(t *testing.T, ptr driver.DevicePtr, n int) deviceWeight {
	return deviceWeight{buf: afmoeCoreTestSlice[uint16](t, ptr, n), dtype: weightBF16}
}

func TestAfmoeCoreLookupRetainsOwner(t *testing.T) {
	lookupErr, unloadErr := errors.New("lookup failed"), errors.New("unload failed")
	for _, failClose := range []bool{false, true} {
		calls, closes := 0, 0
		m, err := lookupAfmoeCoreModule(driver.Module(11), func(string) (driver.Function, error) {
			calls++
			if calls == 3 {
				return 0, lookupErr
			}
			return driver.Function(12), nil
		}, func(*driver.Module) error {
			closes++
			if failClose {
				return unloadErr
			}
			return nil
		})
		if !errors.Is(err, lookupErr) || closes != 1 || calls != 3 {
			t.Fatalf("cleanup: error=%v closes=%d lookups=%d", err, closes, calls)
		}
		if !failClose {
			if m != nil {
				t.Fatal("retained successfully unloaded module")
			}
			continue
		}
		if !errors.Is(err, unloadErr) || m == nil || m.mod != 11 || m.embedding != 12 {
			t.Fatalf("lost owner: module=%v error=%v", m, err)
		}
		if err := m.close(func(*driver.Module) error { closes++; return nil }); err != nil || m.mod != 0 || closes != 2 {
			t.Fatal("retry did not release module")
		}
		if err := m.close(func(*driver.Module) error { t.Fatal("double close"); return nil }); err != nil {
			t.Fatal(err)
		}
	}
}

func TestAfmoeCoreLaunchContracts(t *testing.T) {
	m := &afmoeCoreModule{mod: 1, embedding: 2, linear: 3, residual: 4, argmax: 5}
	in := afmoeCoreTestSlice[float32](t, 0x10000, 4)
	out := afmoeCoreTestSlice[float32](t, 0x20000, 3)
	weight := afmoeCoreTestWeight(t, 0x100000, 12)
	var args []any
	emit := func(_ *cudalaunch.Launcher, a ...any) error { args = a; return nil }
	for _, round := range []bool{false, true} {
		if err := m.emitLinear(weight, in, out, 3, 4, round, emit); err != nil {
			t.Fatal(err)
		}
		want := int32(0)
		if round {
			want = 1
		}
		if len(args) != 6 || args[3] != int32(3) || args[4] != int32(4) || args[5] != want {
			t.Fatalf("linear arguments %v", args)
		}
	}
	embedOut := afmoeCoreTestSlice[float32](t, 0x30000, 4)
	if err := m.emitEmbedding(weight, embedOut, 2, 3, 4, float32(math.Sqrt(4)), emit); err != nil {
		t.Fatal(err)
	}
	if len(args) != 5 || args[2] != int32(2) || args[3] != int32(4) || args[4] != float32(2) {
		t.Fatalf("embedding arguments %v", args)
	}
	// Exact in-place residual output is safe; partial overlap is not.
	if err := m.emitResidual(in, in, embedOut, emit); err != nil {
		t.Fatal(err)
	}
	token := afmoeCoreTestSlice[int32](t, 0x40000, 1)
	if err := m.emitArgmax(out, token, emit); err != nil {
		t.Fatal(err)
	}
	if len(args) != 3 || args[2] != int32(3) {
		t.Fatalf("argmax arguments %v", args)
	}
	launchErr := errors.New("launch failed")
	if err := m.emitLinear(weight, in, out, 3, 4, true,
		func(*cudalaunch.Launcher, ...any) error { return launchErr }); !errors.Is(err, launchErr) {
		t.Fatalf("lost launch failure: %v", err)
	}
}

func TestAfmoeCoreRejectsInvalidLaunches(t *testing.T) {
	m := &afmoeCoreModule{mod: 1, embedding: 2, linear: 3, residual: 4, argmax: 5}
	in := afmoeCoreTestSlice[float32](t, 0x10000, 4)
	out := afmoeCoreTestSlice[float32](t, 0x20000, 3)
	w := afmoeCoreTestWeight(t, 0x100000, 12)
	emit := func(*cudalaunch.Launcher, ...any) error { t.Fatal("invalid launch reached emitter"); return nil }
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"linear rows", func() error { return m.emitLinear(w, in, out, 0, 4, true, emit) }},
		{"linear short weight", func() error { return m.emitLinear(afmoeCoreTestWeight(t, 0x100000, 11), in, out, 3, 4, true, emit) }},
		{"linear wrong storage", func() error { x := w; x.dtype = weightF32; return m.emitLinear(x, in, out, 3, 4, true, emit) }},
		{"linear partial alias", func() error { return m.emitLinear(w, in, afmoeCoreTestSlice[float32](t, 0x10004, 3), 3, 4, true, emit) }},
		{"embedding token", func() error { return m.emitEmbedding(w, in, 3, 3, 4, 2, emit) }},
		{"embedding negative token", func() error { return m.emitEmbedding(w, in, -1, 3, 4, 2, emit) }},
		{"embedding nan scale", func() error { return m.emitEmbedding(w, in, 0, 3, 4, float32(math.NaN()), emit) }},
		{"embedding wrong length", func() error { return m.emitEmbedding(w, out, 0, 3, 4, 2, emit) }},
		{"residual partial alias", func() error { return m.emitResidual(in, afmoeCoreTestSlice[float32](t, 0x10004, 4), in, emit) }},
		{"argmax output alias", func() error { return m.emitArgmax(out, afmoeCoreTestSlice[int32](t, 0x20000, 1), emit) }},
		{"argmax missing scalar", func() error { return m.emitArgmax(out, cudamem.Slice[int32]{}, emit) }},
		{"missing module", func() error { return (*afmoeCoreModule)(nil).emitLinear(w, in, out, 3, 4, true, emit) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.call(); err == nil {
				t.Fatal("accepted invalid launch")
			}
		})
	}
}

func TestAfmoeCoreContextBinding(t *testing.T) {
	contextErr := errors.New("context unavailable")
	entered := false
	m, err := bindAfmoeCoreModule(11,
		func(func() error) error { return contextErr },
		func(string) (driver.Function, error) { t.Fatal("lookup outside context"); return 0, nil },
		func(*driver.Module) error { t.Fatal("unload outside context"); return nil })
	if !errors.Is(err, contextErr) || m == nil || m.mod != 11 {
		t.Fatalf("lost owner before context entry: %v, %v", m, err)
	}
	if err := m.close(func(*driver.Module) error { t.Fatal("unload despite failed context"); return nil }); !errors.Is(err, contextErr) || m.mod != 11 {
		t.Fatalf("close context failure lost owner: %v, %v", m, err)
	}
	bound := func(f func() error) error {
		if entered {
			t.Fatal("nested context entry")
		}
		entered = true
		defer func() { entered = false }()
		return f()
	}
	m.run = bound
	if err := m.close(func(*driver.Module) error {
		if !entered {
			t.Fatal("unload outside context")
		}
		return nil
	}); err != nil || m.mod != 0 {
		t.Fatalf("context retry failed: %v", err)
	}

	lookupErr, unloadErr := errors.New("lookup failed"), errors.New("unload failed")
	m, err = bindAfmoeCoreModule(12, bound, func(string) (driver.Function, error) {
		if !entered {
			t.Fatal("lookup outside context")
		}
		return 0, lookupErr
	}, func(*driver.Module) error {
		if !entered {
			t.Fatal("cleanup outside context")
		}
		return unloadErr
	})
	if m == nil || m.mod != 12 || !errors.Is(err, lookupErr) || !errors.Is(err, unloadErr) {
		t.Fatalf("lost lookup/cleanup failure owner: %v, %v", m, err)
	}
	if err := m.close(func(*driver.Module) error { return nil }); err != nil || m.mod != 0 {
		t.Fatalf("lookup cleanup retry: %v", err)
	}
}
