package cudacensus

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
)

func TestSamplerCloseRetainsFailedHandles(t *testing.T) {
	a, b := errors.New("first free"), errors.New("second free")
	calls := map[nvml.GpmSample]int{}
	fail := true
	s := &gpmSampler{prev: 1, next: 2, want: []Counter{SMActive}, freeSample: func(h nvml.GpmSample) error {
		calls[h]++
		if fail {
			if h == 1 {
				return a
			}
			return b
		}
		return nil
	}, getSample: func(nvml.GpmSample, nvml.Device) error { t.Fatal("read after close started"); return nil }}
	err := s.Close()
	if !errors.Is(err, a) || !errors.Is(err, b) || s.prev != 1 || s.next != 2 {
		t.Fatalf("lost owners or errors: %+v %v", s, err)
	}
	if _, err := s.Read(make([]Reading, 1)); err == nil {
		t.Fatal("read while cleanup pending")
	}
	fail = false
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if calls[1] != 2 || calls[2] != 2 || s.prev != 0 || s.next != 0 {
		t.Fatalf("calls=%v handles=%d,%d", calls, s.prev, s.next)
	}
}

func TestSamplerPartialConstructionOwnership(t *testing.T) {
	for _, phase := range []string{"first", "second", "partial_second", "nil_handle", "success"} {
		t.Run(phase, func(t *testing.T) {
			allocationErr, freeErr := errors.New("allocation"), errors.New("free")
			n := 0
			failFree := true
			freed := map[nvml.GpmSample]int{}
			ops := samplerOpenOps{
				discover: func(nvml.Device, []Counter, StructVersions) (Support, error) { return Support{GPM: true}, nil },
				allocate: func() (nvml.GpmSample, error) {
					n++
					if phase == "first" && n == 1 {
						return 0, allocationErr
					}
					if n == 2 {
						switch phase {
						case "second":
							return 0, allocationErr
						case "partial_second":
							return 2, allocationErr
						case "nil_handle":
							return 0, nil
						}
					}
					return nvml.GpmSample(n), nil
				},
				free: func(h nvml.GpmSample) error {
					freed[h]++
					if h == 1 && failFree {
						return freeErr
					}
					return nil
				},
			}
			s, _, err := openSampler(0, []Counter{SMActive}, StructVersions{1, 1}, ops)
			if phase == "first" {
				if s != nil || !errors.Is(err, allocationErr) || len(freed) != 0 {
					t.Fatalf("s=%v err=%v frees=%v", s, err, freed)
				}
				return
			}
			if phase == "success" {
				if err != nil || s == nil {
					t.Fatalf("s=%v err=%v", s, err)
				}
				if err = s.Close(); !errors.Is(err, freeErr) {
					t.Fatal(err)
				}
			} else {
				if s == nil || err == nil || !errors.Is(err, freeErr) {
					t.Fatalf("lost partial owner: %v %v", s, err)
				}
				if phase != "nil_handle" && !errors.Is(err, allocationErr) {
					t.Fatal("lost allocation error")
				}
			}
			if s.prev != 1 || s.next != 0 {
				t.Fatalf("pending handles=%d,%d", s.prev, s.next)
			}
			failFree = false
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if freed[1] != 2 || freed[2] > 1 {
				t.Fatalf("cleanup calls=%v", freed)
			}
		})
	}
}
