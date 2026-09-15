package cuptirange

import (
	"errors"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestPreparedRanges(t *testing.T) {
	for _, test := range []struct {
		name    string
		failure string
	}{{"success", ""}, {"prepare fails", "prepare"}, {"trace changes", "trace"}, {"ignored region error", "ignored"}, {"pop fails", "pop"}, {"native wrong identity", "identity"}} {
		t.Run(test.name, func(t *testing.T) {
			c := &counterStore{host: 1, image: make([]byte, 32)}
			defer c.pin.Unpin()
			var events []string
			pass := uint(0)
			api := collectionAPI{
				config: func(p *cupti.RangeProfilerSetConfigParams) error {
					if p.Range != cupti.UserRange || p.NumNestingLevels != 3 {
						t.Fatal("wrong native mode")
					}
					return nil
				},
				start: func(*cupti.RangeProfilerStartParams) error { events = append(events, "start"); return nil },
				stop: func(p *cupti.RangeProfilerStopParams) error {
					events = append(events, "stop")
					pass++
					if pass == 2 {
						p.IsAllPassSubmitted = 1
					} else {
						p.IsAllPassSubmitted = 0
					}
					return nil
				},
				decode: func(*cupti.RangeProfilerDecodeDataParams) error { events = append(events, "decode"); return nil },
				info:   func(p *cupti.RangeProfilerGetCounterDataInfoParams) error { p.NumTotalRanges = 2; return nil },
			}
			sync := func() error { events = append(events, "sync"); return nil }
			replay := Replay{Prepare: func(uint) error {
				events = append(events, "prepare")
				if test.failure == "prepare" {
					return errors.New("preparation failed")
				}
				return nil
			}, Synchronize: sync,
				Run: func(p uint, s *Scope) error {
					events = append(events, "run")
					if test.failure == "ignored" {
						_ = s.Region("bad/name", func() error { return nil })
						return nil
					}
					name := "kernel"
					if test.failure == "trace" && p == 1 {
						name = "changed"
					}
					return s.Region(name, func() error { events = append(events, "work"); return nil })
				}}
			push := func(name string) error { events = append(events, "push "+name); return nil }
			pop := func() error {
				events = append(events, "pop")
				if test.failure == "pop" {
					return errors.New("pop failed")
				}
				return nil
			}
			names := [][]byte{append([]byte("operation#1"), 0), append([]byte("operation#1/kernel#1"), 0)}
			if test.failure == "identity" {
				names[1] = append([]byte("wrong"), 0)
			}
			var pin runtime.Pinner
			for _, name := range names {
				pin.Pin(&name[0])
			}
			defer pin.Unpin()
			info := func(p *cupti.RangeProfilerCounterDataGetRangeInfoParams) error {
				p.RangeName = uintptr(unsafe.Pointer(&names[p.RangeIndex][0]))
				return nil
			}
			_, records, err := c.captureReplayWithAPI([]byte{1}, Limits{MaxRanges: 4, MaxRangeTreeNodes: 4, MaxPasses: 2, MaxCounterBytes: 32}, CaptureOptions{Mode: Regions, MaxDepth: 3}, replay, true, api, push, pop, info)
			if test.failure != "" {
				if err == nil {
					t.Fatal("failure admitted")
				}
				if records != nil {
					t.Fatal("failed capture returned identities")
				}
				if test.failure == "prepare" && !reflect.DeepEqual(events, []string{"sync", "prepare", "sync"}) {
					t.Fatalf("preparation entered collection: %v", events)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			one := []string{"sync", "prepare", "sync", "start", "push operation#1", "run", "push kernel#1", "work", "pop", "sync", "pop", "stop"}
			want := append(append(append([]string{}, one...), one...), "decode")
			if !reflect.DeepEqual(events, want) {
				t.Fatalf("events=%v want=%v", events, want)
			}
			if len(records) != 2 || records[1].ParentID != "operation#1" || records[1].Name != "kernel" || records[1].Index != 1 {
				t.Fatalf("records=%+v", records)
			}
		})
	}
}

func TestScopeBoundsAndPanic(t *testing.T) {
	s := &Scope{active: true, mode: Regions, depth: 2, limit: 3, counts: make(map[string]int), push: func(string) error { return nil }, pop: func() error { return nil }}
	if err := s.region("operation", "operation", func() error {
		return s.Region("child", func() error {
			return s.Region("too-deep", func() error { t.Fatal("overflow callback executed"); return nil })
		})
	}); err == nil {
		t.Fatal("depth accepted")
	}
	if len(s.stack) != 0 {
		t.Fatal("unbalanced error")
	}
	pops := 0
	s = &Scope{active: true, mode: Regions, depth: 2, limit: 3, counts: make(map[string]int), push: func(string) error { return nil }, pop: func() error { pops++; return nil }}
	func() {
		defer func() {
			if recover() != "panic" {
				t.Error("panic not preserved")
			}
		}()
		_ = s.region("operation", "operation", func() error { return s.Region("child", func() error { panic("panic") }) })
	}()
	if pops != 2 || len(s.stack) != 0 {
		t.Fatal("unbalanced panic")
	}
}

func TestScopeOccurrenceIdentity(t *testing.T) {
	s := &Scope{active: true, mode: Regions, depth: 3, limit: 5, counts: make(map[string]int), push: func(string) error { return nil }, pop: func() error { return nil }}
	err := s.region("operation", "operation", func() error {
		for _, name := range []string{"a", "a", "a#1"} {
			if err := s.Region(name, func() error { return nil }); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if s.ranges[1].ID == s.ranges[2].ID || !strings.Contains(s.ranges[3].ID, "%23") {
		t.Fatalf("ambiguous IDs: %+v", s.ranges)
	}
	s.active = false
	if err := s.Region("late", func() error { return nil }); err == nil {
		t.Fatal("expired scope admitted")
	}
}
