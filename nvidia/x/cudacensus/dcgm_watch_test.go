package cudacensus

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

type fakeDCGMWatch struct {
	t                       *testing.T
	calls                   []string
	fail                    string
	err                     error
	groups, fields, watches int
}

func (s *fakeDCGMWatch) call(step string) error {
	s.calls = append(s.calls, step)
	if step == s.fail {
		return s.err
	}
	return nil
}
func (s *fakeDCGMWatch) NewGroup(name string, kind dcgm.GroupType) (dcgm.GpuGrp, error) {
	if name != "fixture" || kind != dcgm.GroupEmpty {
		s.t.Fatal("group arguments")
	}
	if err := s.call("group"); err != nil {
		return 0, err
	}
	s.groups++
	return 11, nil
}
func (s *fakeDCGMWatch) AddEntity(group dcgm.GpuGrp, kind dcgm.FieldEntityGroup, id dcgm.FieldEid) error {
	if group != 11 || kind != dcgm.FeGpu || id != 3 {
		s.t.Fatal("entity arguments")
	}
	return s.call("entity")
}
func (s *fakeDCGMWatch) NewFieldGroup(name string, fields []dcgm.FieldID) (dcgm.FieldGrp, error) {
	if name != "fixture" || !reflect.DeepEqual(fields, []dcgm.FieldID{dcgm.FieldProfSmActive}) {
		s.t.Fatal("field group arguments")
	}
	if err := s.call("fields"); err != nil {
		return 0, err
	}
	s.fields++
	return 22, nil
}
func (s *fakeDCGMWatch) Watch(group dcgm.GpuGrp, fields dcgm.FieldGrp, frequency, age time.Duration, samples int32) error {
	if group != 11 || fields != 22 || frequency != time.Millisecond || age != time.Second || samples != 2 {
		s.t.Fatal("watch arguments")
	}
	if err := s.call("watch"); err != nil {
		return err
	}
	s.watches++
	return nil
}

func TestWatchDCGMSetup(t *testing.T) {
	steps := []string{"group", "entity", "fields", "watch"}
	for _, fail := range append([]string{""}, steps...) {
		t.Run(fail, func(t *testing.T) {
			failure := errors.New("setup failed")
			s := &fakeDCGMWatch{t: t, fail: fail, err: failure}
			r, err := watchDCGM(s, 1, "fixture", 3, []Counter{SMActive, DMMATensorActive}, time.Millisecond, time.Second)
			if fail == "" {
				if err != nil || r == nil || s.groups != 1 || s.fields != 1 || s.watches != 1 {
					t.Fatalf("setup: %+v %v", s, err)
				}
			} else {
				if !errors.Is(err, failure) || r != nil {
					t.Fatalf("failure: %v %v", r, err)
				}
				for i, step := range steps {
					if step == fail && !reflect.DeepEqual(s.calls, steps[:i+1]) {
						t.Fatalf("continued after failure: %v", s.calls)
					}
				}
				if fail != "group" && s.groups != 1 {
					t.Fatal("lost session-owned group")
				}
				if fail == "watch" && s.fields != 1 {
					t.Fatal("lost session-owned field group")
				}
			}
		})
	}
}

func TestWatchDCGMValidationBeforeSetup(t *testing.T) {
	for _, name := range []string{"", "bad\x00name", strings.Repeat("x", dcgm.MaxStrLength)} {
		s := &fakeDCGMWatch{t: t}
		if _, err := watchDCGM(s, 1, name, 3, []Counter{SMActive}, time.Millisecond, time.Second); err == nil || len(s.calls) != 0 {
			t.Fatal("name validation")
		}
	}
	for _, freq := range []time.Duration{0, -1, time.Nanosecond, time.Microsecond + 1, 2 * time.Second} {
		s := &fakeDCGMWatch{t: t}
		if _, err := watchDCGM(s, 1, "fixture", 3, []Counter{SMActive}, freq, time.Second); err == nil || len(s.calls) != 0 {
			t.Fatal("cadence validation")
		}
	}
	s := &fakeDCGMWatch{t: t}
	if _, err := watchDCGM(s, 1, "fixture", 3, []Counter{DMMATensorActive}, time.Millisecond, time.Second); err == nil || len(s.calls) != 0 {
		t.Fatal("unmapped request validation")
	}
	if _, err := WatchDCGM(nil, "fixture", 3, []Counter{SMActive}, time.Millisecond, time.Second); err == nil {
		t.Fatal("nil session accepted")
	}
}
