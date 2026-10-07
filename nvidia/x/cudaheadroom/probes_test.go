package cudaheadroom

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

// Synthetic probes exercise the same arithmetic and query sequence as the
// public owned-kernel methods. They are not physical attribute observations.
func measuredProbes() probes {
	return probes{
		device: func(a driver.DeviceAttribute) (int, error) {
			switch a {
			case driver.DevAttrMaxSharedMemoryPerBlockOptin:
				return 101376, nil
			case driver.DevAttrReservedSharedMemoryPerBlock:
				return 1024, nil
			}
			return 0, errors.New("unexpected device query")
		},
		function: func(a driver.FunctionAttribute) (int, error) {
			switch a {
			case driver.FuncAttributeSharedMemoryMode:
				return int(driver.SharedMemoryModeDefault), nil
			case driver.FuncAttributeSharedSizeBytes:
				return 1024, nil
			case driver.FuncAttributeMaxDynamicSharedSizeBytes:
				return 48000, nil
			}
			return 0, errors.New("unexpected function query")
		},
	}
}

func TestRequiredProbeErrors(t *testing.T) {
	sentinel := errors.New("missing attribute")
	for _, tc := range []struct {
		name     string
		device   driver.DeviceAttribute
		function driver.FunctionAttribute
	}{
		{name: "mode", function: driver.FuncAttributeSharedMemoryMode},
		{name: "user", device: driver.DevAttrMaxSharedMemoryPerBlockOptin},
		{name: "static", function: driver.FuncAttributeSharedSizeBytes},
		{name: "dynamic", function: driver.FuncAttributeMaxDynamicSharedSizeBytes},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := measuredProbes()
			dev, fn := p.device, p.function
			p.device = func(a driver.DeviceAttribute) (int, error) {
				if tc.device != 0 && a == tc.device {
					return 0, sentinel
				}
				return dev(a)
			}
			p.function = func(a driver.FunctionAttribute) (int, error) {
				if tc.device == 0 && a == tc.function {
					return 0, sentinel
				}
				return fn(a)
			}
			if _, err := inspect(p); !errors.Is(err, sentinel) {
				t.Fatalf("lost required error %v", err)
			}
			p.set = func(driver.FunctionAttribute, int) error { t.Fatal("setter after failed query"); return nil }
			if _, err := preparePlan(p, testPlan(t, 512, 48000), 512); !errors.Is(err, sentinel) {
				t.Fatalf("prepare %v", err)
			}
		})
	}
	if _, err := inspect(probes{}); err == nil {
		t.Fatal("missing probe accepted")
	}
}

func TestModeAndReservation(t *testing.T) {
	p := measuredProbes()
	fn := p.function
	p.function = func(a driver.FunctionAttribute) (int, error) {
		if a == driver.FuncAttributeSharedMemoryMode {
			return int(driver.SharedMemoryModeDefault) + 1, nil
		}
		return fn(a)
	}
	if _, err := inspect(p); err == nil || !strings.Contains(err.Error(), "unsupported shared mode") {
		t.Fatalf("mode %v", err)
	}
	sentinel := errors.New("optional unavailable")
	for _, tc := range []struct {
		name     string
		value    int
		err      error
		observed bool
	}{
		{"absent", 0, sentinel, false}, {"negative", -1, nil, false}, {"zero", 0, nil, true}, {"nonzero", 1024, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := measuredProbes()
			dev := p.device
			p.device = func(a driver.DeviceAttribute) (int, error) {
				if a == driver.DevAttrReservedSharedMemoryPerBlock {
					return tc.value, tc.err
				}
				return dev(a)
			}
			b, err := inspect(p)
			if err != nil {
				t.Fatal(err)
			}
			if b.Ceiling != 48000 || (b.Reserved != nil) != tc.observed {
				t.Fatalf("reservation %+v", b)
			}
			if tc.observed {
				if *b.Reserved != int64(tc.value) || b.ReservationError != nil {
					t.Fatal("lost measured value")
				}
			} else if b.ReservationError == nil {
				t.Fatal("lost unavailable diagnostic")
			}
			if tc.err != nil && !errors.Is(b.ReservationError, sentinel) {
				t.Fatal("lost optional error")
			}
		})
	}
}

func TestPrepareSetterAndReadback(t *testing.T) {
	for _, tc := range []struct {
		name                              string
		setter                            bool
		setErr                            bool
		readback                          int
		changeMode, changeStatic, readErr bool
		wantErr                           bool
	}{
		{name: "missing_setter", readback: 48000, wantErr: true},
		{name: "setter_error", setter: true, setErr: true, readback: 48000, wantErr: true},
		{name: "low_readback", setter: true, readback: 50000, wantErr: true},
		{name: "successful_readback", setter: true, readback: 60000},
		{name: "readback_error", setter: true, readback: 60000, readErr: true, wantErr: true},
		{name: "changed_mode", setter: true, readback: 60000, changeMode: true, wantErr: true},
		{name: "changed_static", setter: true, readback: 60000, changeStatic: true, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := measuredProbes()
			fn := p.function
			set := false
			reads := 0
			sets := 0
			p.function = func(a driver.FunctionAttribute) (int, error) {
				if a == driver.FuncAttributeMaxDynamicSharedSizeBytes {
					reads++
					if set {
						if tc.readErr {
							return 0, errors.New("readback failed")
						}
						return tc.readback, nil
					}
				}
				if set && a == driver.FuncAttributeSharedMemoryMode && tc.changeMode {
					return int(driver.SharedMemoryModeDefault) + 1, nil
				}
				if set && a == driver.FuncAttributeSharedSizeBytes && tc.changeStatic {
					return 2048, nil
				}
				return fn(a)
			}
			if tc.setter {
				p.set = func(a driver.FunctionAttribute, value int) error {
					sets++
					if a != driver.FuncAttributeMaxDynamicSharedSizeBytes || value != 60000 {
						t.Fatalf("setter %v %d", a, value)
					}
					set = true
					if tc.setErr {
						return errors.New("setter failed")
					}
					return nil
				}
			}
			b, err := preparePlan(p, testPlan(t, 512, 60000), 512)
			if (err != nil) != tc.wantErr {
				t.Fatalf("prepare %+v %v", b, err)
			}
			if tc.setter && sets != 1 {
				t.Fatalf("setter count %d", sets)
			}
			if !tc.wantErr && (reads != 2 || b.FunctionDynamic != 60000 || b.Ceiling != 60000) {
				t.Fatalf("readback count %d budget %+v", reads, b)
			}
		})
	}
}

func TestPrepareWithoutIncreaseStillChecksFresh(t *testing.T) {
	p := measuredProbes()
	fn := p.function
	reads := 0
	p.function = func(a driver.FunctionAttribute) (int, error) {
		if a == driver.FuncAttributeMaxDynamicSharedSizeBytes {
			reads++
			if reads > 1 {
				return 47000, nil
			}
		}
		return fn(a)
	}
	p.set = func(driver.FunctionAttribute, int) error { t.Fatal("unnecessary setter"); return nil }
	if _, err := preparePlan(p, testPlan(t, 512, 48000), 512); err == nil || reads != 2 {
		t.Fatalf("fresh cap: %v reads %d", err, reads)
	}
	p = measuredProbes()
	if _, err := preparePlan(p, testPlan(t, 512, 48000), 512); err != nil {
		t.Fatalf("setter not needed: %v", err)
	}
}

func TestPrepareRefusesBeforeSetter(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		static, dynamic, other int64
	}{
		{"static_mismatch", 512, 60000, 0}, {"device_overflow", 512, 100353, 512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := measuredProbes()
			p.set = func(driver.FunctionAttribute, int) error { t.Fatal("setter before validation"); return nil }
			if _, err := preparePlan(p, testPlan(t, tc.static, tc.dynamic), tc.other); err == nil {
				t.Fatal("invalid plan accepted")
			}
		})
	}
}
