package nvml

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"unsafe"
)

func TestIsUnsupported(t *testing.T) {
	bind := errors.New("binding failed")
	for _, tt := range []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"unsupported", ErrNotSupported, true},
		{"missing", &SymbolError{Symbol: "old"}, true},
		{"permission", ErrNoPermission, false},
		{"lost", ErrGpuIsLost, false},
		{"binding", &SymbolError{Symbol: "bad", Err: bind}, false},
		{"wrapped", fmt.Errorf("query: %w", ErrNotSupported), true},
		{"wrapped missing", fmt.Errorf("query: %w", &SymbolError{Symbol: "old"}), true},
		{"joined missing second", errors.Join(&SymbolError{Err: bind}, &SymbolError{Symbol: "old"}), true},
		{"joined unsupported", errors.Join(ErrNoPermission, ErrNotSupported), true},
		{"joined operational", errors.Join(ErrNoPermission, ErrGpuIsLost), false},
		{"binding cause unsupported", &SymbolError{Err: ErrNotSupported}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsUnsupported(tt.err); got != tt.want {
				t.Fatalf("IsUnsupported = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestHelperVersions(t *testing.T) {
	for _, supported := range []uint32{0, 1} {
		got, err := gpmSupported(func(info *GpmSupport) error {
			if info.Version != GpmSupportVersion || info.IsSupportedDevice != 0 {
				t.Fatalf("gpm request = %+v", info)
			}
			info.IsSupportedDevice = supported
			return nil
		})
		if err != nil || got != (supported != 0) {
			t.Fatalf("support = %v, %v", got, err)
		}
	}
	got, err := fabricInfo(func(info *GpuFabricInfoV) error {
		if info.Version != GpuFabricInfoV3Version || info.CliqueId != 0 {
			t.Fatalf("fabric request = %+v", info)
		}
		info.CliqueId, info.Status, info.HealthMask = 17, ErrNotSupported, 31
		return nil
	})
	if err != nil || got.CliqueId != 17 || got.Status != ErrNotSupported || got.HealthMask != 31 {
		t.Fatalf("fabric result = %+v, %v", got, err)
	}
	for _, e := range []error{ErrNoPermission, ErrGpuIsLost, &SymbolError{Symbol: "old"}} {
		if _, err := gpmSupported(func(*GpmSupport) error { return e }); !errors.Is(err, e) {
			t.Fatalf("support error = %v, want %v", err, e)
		}
		if _, err := fabricInfo(func(*GpuFabricInfoV) error { return e }); !errors.Is(err, e) {
			t.Fatalf("fabric error = %v, want %v", err, e)
		}
	}
}

func TestProcessesUtilizationSizing(t *testing.T) {
	for _, tt := range []struct {
		name    string
		counts  []uint32
		errors  []error
		want    int
		wantErr bool
	}{
		{"growth and shrink", []uint32{1, 3, 2}, []error{ErrInsufficientSize, ErrInsufficientSize, nil}, 2, false},
		{"empty", []uint32{0}, []error{nil}, 0, false},
		{"size zero", []uint32{0}, []error{ErrInsufficientSize}, 0, true},
		{"size unchanged", []uint32{1, 1}, []error{ErrInsufficientSize, ErrInsufficientSize}, 0, true},
		{"huge", []uint32{helperMaxEntries + 1}, []error{ErrInsufficientSize}, 0, true},
		{"success overflow", []uint32{1}, []error{nil}, 0, true},
		{"filled overflow", []uint32{1, 2}, []error{ErrInsufficientSize, nil}, 0, true},
		{"never stable", []uint32{1, 2, 3, 4}, []error{ErrInsufficientSize, ErrInsufficientSize, ErrInsufficientSize, ErrInsufficientSize}, 0, true},
		{"permission", []uint32{0}, []error{ErrNoPermission}, 0, true},
		{"not found", []uint32{0}, []error{ErrNotFound}, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			got, err := processesUtilization(997, func(info *ProcessesUtilizationInfo, buf []ProcessUtilizationInfoV1) error {
				if calls >= len(tt.counts) {
					t.Fatal("unexpected retry")
				}
				if info.Version != ProcessesUtilizationInfoV1Version || info.LastSeenTimeStamp != 997 || info.ProcessSamplesCount != uint32(len(buf)) {
					t.Fatalf("request = %+v, buffer %d", info, len(buf))
				}
				if len(buf) == 0 {
					if info.ProcUtilArray != 0 {
						t.Fatal("non-null sizing pointer")
					}
				} else if info.ProcUtilArray != uintptr(unsafe.Pointer(&buf[0])) {
					t.Fatal("wrong sample pointer")
				}
				for i := range buf {
					buf[i].Pid = uint32(23 + i)
				}
				info.ProcessSamplesCount = tt.counts[calls]
				// Drivers may update request fields; the next call must restore
				// the caller's cursor and version.
				info.LastSeenTimeStamp, info.Version = 2, 0
				e := tt.errors[calls]
				calls++
				return e
			})
			if (err != nil) != tt.wantErr || len(got) != tt.want || calls != len(tt.counts) {
				t.Fatalf("result = %v, %v, calls %d", got, err, calls)
			}
			if tt.want > 0 && (got[0].Pid != 23 || got[1].Pid != 24) {
				t.Fatal("samples lost")
			}
			if tt.name == "permission" && !errors.Is(err, ErrNoPermission) {
				t.Fatal("permission lost")
			}
			if tt.name == "not found" && !errors.Is(err, ErrNotFound) {
				t.Fatal("not-found lost")
			}
			if tt.name == "never stable" && !errors.Is(err, ErrInsufficientSize) {
				t.Fatal("size error lost")
			}
		})
	}
}

func TestFieldValuesFor(t *testing.T) {
	got, err := fieldValuesFor([]uint32{71, 83}, func(n int32, first *FieldValue) error {
		values := unsafe.Slice(first, n)
		if values[0].FieldId != 71 || values[1].FieldId != 83 {
			t.Fatal("field order")
		}
		values[0].NvmlReturn, values[1].NvmlReturn = ErrNoPermission, Success
		values[1].Value = Value{Data: [1]uint64{99}}
		return nil
	})
	if err != nil || len(got) != 2 || got[0].NvmlReturn != ErrNoPermission || got[1].Value.Uint64() != 99 {
		t.Fatalf("fields = %+v, %v", got, err)
	}
	for _, ids := range [][]uint32{nil, make([]uint32, helperMaxEntries+1)} {
		_, err := fieldValuesFor(ids, func(int32, *FieldValue) error { t.Fatal("called on invalid/empty request"); return nil })
		if (err != nil) != (len(ids) > 0) {
			t.Fatalf("error = %v", err)
		}
	}
	_, err = fieldValuesFor([]uint32{1}, func(int32, *FieldValue) error { return ErrGpuIsLost })
	if !errors.Is(err, ErrGpuIsLost) {
		t.Fatal("lost-device error lost")
	}
}

func TestGpmMetricsHelper(t *testing.T) {
	ids := []GpmMetricId{GpmMetricSmUtil, GpmMetricDramBwUtil}
	got, err := gpmMetrics(11, 13, ids, func(info *GpmMetricsGetParams) error {
		if info.Version != GpmMetricsGetVersion || info.Sample1 != 11 || info.Sample2 != 13 || info.NumMetrics != 2 || info.Metrics[0].MetricId != uint32(ids[0]) || info.Metrics[1].MetricId != uint32(ids[1]) {
			t.Fatalf("request = %+v", info)
		}
		info.Metrics[0].NvmlReturn, info.Metrics[1].Value = ErrNotSupported, 43.5
		return nil
	})
	if err != nil || len(got) != 2 || got[0].NvmlReturn != ErrNotSupported || got[1].Value != 43.5 {
		t.Fatalf("metrics = %+v, %v", got, err)
	}
	for _, ids := range [][]GpmMetricId{nil, make([]GpmMetricId, 334), {-1}, {0}} {
		_, err := gpmMetrics(0, 0, ids, func(*GpmMetricsGetParams) error { t.Fatal("called on invalid/empty request"); return nil })
		if (err != nil) != (len(ids) > 0) {
			t.Fatalf("error = %v", err)
		}
	}
	for _, n := range []uint32{0, 2, 334} {
		_, err := gpmMetrics(11, 13, []GpmMetricId{1}, func(info *GpmMetricsGetParams) error { info.NumMetrics = n; return nil })
		if err == nil {
			t.Fatalf("accepted changed count %d", n)
		}
	}
	_, err = gpmMetrics(11, 13, []GpmMetricId{1}, func(*GpmMetricsGetParams) error { return ErrNoPermission })
	if !errors.Is(err, ErrNoPermission) {
		t.Fatal("permission lost")
	}
}

func TestGpuInstanceList(t *testing.T) {
	for _, tt := range []struct {
		name                string
		capacity, returned  uint32
		callErr, profileErr error
		want                []GpuInstance
		fail                bool
	}{
		{"shrinks", 3, 2, nil, nil, []GpuInstance{17, 19}, false},
		{"empty", 0, 0, nil, nil, nil, false},
		{"no active instances", 3, 0, nil, nil, []GpuInstance{}, false},
		{"too large", helperMaxEntries + 1, 0, nil, nil, nil, true},
		{"returned overflow", 2, 3, nil, nil, nil, true},
		{"call error", 2, 0, ErrNoPermission, nil, nil, true},
		{"profile error", 2, 0, nil, ErrNotSupported, nil, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			got, err := gpuInstanceList(29, func(id uint32) (GpuInstanceProfileInfoV2, error) {
				if id != 29 {
					t.Fatal("wrong profile ID")
				}
				return GpuInstanceProfileInfoV2{InstanceCount: tt.capacity}, tt.profileErr
			}, func(id uint32, first *GpuInstance, n *uint32) error {
				calls++
				if id != 29 || *n != tt.capacity {
					t.Fatal("instance request")
				}
				buf := unsafe.Slice(first, *n)
				for i := range buf {
					buf[i] = GpuInstance(17 + 2*i)
				}
				*n = tt.returned
				return tt.callErr
			})
			if (err != nil) != tt.fail || !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("instances = %v, %v", got, err)
			}
			wantCalls := 1
			if tt.capacity == 0 || tt.capacity > helperMaxEntries || tt.profileErr != nil {
				wantCalls = 0
			}
			if calls != wantCalls {
				t.Fatalf("calls = %d", calls)
			}
			if tt.callErr != nil && !errors.Is(err, tt.callErr) {
				t.Fatal("call error lost")
			}
			if tt.profileErr != nil && !errors.Is(err, tt.profileErr) {
				t.Fatal("profile error lost")
			}
		})
	}
}
