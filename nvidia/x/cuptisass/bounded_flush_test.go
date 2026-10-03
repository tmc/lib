package cuptisass

import (
	"errors"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestFlushBoundedStorage(t *testing.T) {
	valid := FlushLimits{2, 3, 4096, 16}
	for _, tt := range []struct {
		name    string
		limits  FlushLimits
		n, m    uint64
		wantErr bool
	}{
		{"valid", valid, 2, 3, false},
		{"zero records", FlushLimits{0, 3, 4096, 16}, 2, 3, true},
		{"zero instances", FlushLimits{2, 0, 4096, 16}, 2, 3, true},
		{"zero bytes", FlushLimits{2, 3, 0, 16}, 2, 3, true},
		{"zero names", FlushLimits{2, 3, 4096, 0}, 2, 3, true},
		{"records", valid, 3, 3, true},
		{"instances", valid, 2, 4, true},
		{"headers", FlushLimits{2, 3, 1, 16}, 2, 3, true},
		{"name reservation", FlushLimits{2, 3, 4096, 4096}, 2, 3, true},
		{"overflow", FlushLimits{^uint64(0), ^uint64(0), ^uint64(0), 1}, 1 << 62, 8, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := flushStorage(tt.n, tt.m, tt.limits); (err != nil) != tt.wantErr {
				t.Fatalf("error = %v", err)
			}
		})
	}
	// Exact accounting includes both value arrays and both record/header arrays.
	bytes := 3*uint64(unsafe.Sizeof([]Record{})) + 2*(uint64(unsafe.Sizeof(cupti.SassMetricsData{}))+uint64(unsafe.Sizeof([]cupti.SassMetricsInstanceValue{}))+uint64(unsafe.Sizeof(Record{}))+16) + 6*(uint64(unsafe.Sizeof(cupti.SassMetricsInstanceValue{}))+uint64(unsafe.Sizeof(InstanceValue{})))
	valid.MaxBytes = bytes
	if err := flushStorage(2, 3, valid); err != nil {
		t.Fatal(err)
	}
	valid.MaxBytes--
	if err := flushStorage(2, 3, valid); err == nil {
		t.Fatal("one-byte-short allocation admitted")
	}
}

func TestFlushBoundedNativeControls(t *testing.T) {
	for _, mode := range []string{"healthy", "oversized properties", "flush error", "record growth", "instance growth", "record shrink", "instance shrink", "name error", "pointer changed", "instance pointer changed", "invalid limits"} {
		t.Run(mode, func(t *testing.T) {
			queries, flushes, names, freed := 0, 0, 0, 0
			released := make(map[uintptr]int)
			limits := FlushLimits{2, 2, 4096, 8}
			if mode == "invalid limits" {
				limits.MaxBytes = 0
			}
			api := flushAPI{
				properties: func(p *cupti.SassMetricsGetDataPropertiesParams) error {
					queries++
					p.NumOfPatchedInstructionRecords = 2
					p.NumOfInstances = 2
					if mode == "oversized properties" {
						p.NumOfPatchedInstructionRecords = 3
					}
					return nil
				},
				flush: func(p *cupti.SassMetricsFlushDataParams, r []cupti.SassMetricsData, v [][]cupti.SassMetricsInstanceValue) error {
					flushes++
					for i := range r {
						r[i].FunctionName = uintptr(i + 1)
						v[i][0] = cupti.SassMetricsInstanceValue{MetricId: 7, Value: 42}
					}
					switch mode {
					case "flush error":
						return errors.New("native failure")
					case "record growth":
						p.NumOfPatchedInstructionRecords++
					case "instance growth":
						p.NumOfInstances++
					case "record shrink":
						p.NumOfPatchedInstructionRecords--
					case "instance shrink":
						p.NumOfInstances--
					case "instance pointer changed":
						r[0].PInstanceValues++
					case "pointer changed":
						p.PMetricsData++
					}
					return nil
				},
				free: func(ptr uintptr) { freed++; released[ptr]++ },
				name: func(_ uintptr, n uint64) (string, error) {
					names++
					if n != 8 {
						t.Fatal("name bound lost")
					}
					if mode == "name error" {
						return "", errors.New("unterminated")
					}
					return "kernel", nil
				},
			}
			records, err := new(Session).flushData(&limits, api)
			if mode == "healthy" {
				if err != nil || len(records) != 2 || records[1].FunctionName != "kernel" || records[1].Values[0].Value != 42 {
					t.Fatalf("records=%v error=%v", records, err)
				}
			} else if err == nil || records != nil {
				t.Fatalf("unsafe result admitted: %v %v", records, err)
			}
			if mode == "invalid limits" {
				if queries != 0 || flushes != 0 {
					t.Fatal("invalid limits reached native API")
				}
				return
			}
			if queries != 1 {
				t.Fatal("properties queried more than once")
			}
			if mode == "oversized properties" {
				if flushes != 0 || freed != 0 {
					t.Fatal("oversized properties reached allocation/flush")
				}
				return
			}
			if flushes != 1 || freed != 2 || released[1] != 1 || released[2] != 1 {
				t.Fatalf("flushes=%d freed=%d", flushes, freed)
			}
			if mode != "healthy" && mode != "name error" && names != 0 {
				t.Fatal("invalid native capacities reached name decoder")
			}
		})
	}
}

func TestBoundedFunctionName(t *testing.T) {
	for _, tt := range []struct {
		name  string
		bytes []byte
		bound uint64
		want  string
		fail  bool
	}{
		{"terminated", []byte{'a', 'b', 0}, 3, "ab", false},
		{"terminator at limit", []byte{'a', 'b', 0}, 2, "", true},
		{"empty", []byte{0}, 1, "", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var pin runtime.Pinner
			pin.Pin(&tt.bytes[0])
			defer pin.Unpin()
			got, err := boundedFunctionName(uintptr(unsafe.Pointer(&tt.bytes[0])), tt.bound)
			if got != tt.want || (err != nil) != tt.fail {
				t.Fatalf("got %q, %v", got, err)
			}
		})
	}
}
