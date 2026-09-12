package nccltest

import (
	"errors"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

func TestDrainRanksContinuesAfterFailures(t *testing.T) {
	selectFailure := errors.New("context unavailable")
	syncFailure := errors.New("stream failed")
	var selected []driver.Context
	var synced []driver.Stream
	states := []rankState{{}, {context: 1, stream: 11}, {context: 2, stream: 22}, {context: 3, stream: 33}}
	err := drainRanks(states, false, func(c driver.Context) error {
		selected = append(selected, c)
		if c == 1 {
			return selectFailure
		}
		return nil
	}, func(s driver.Stream) error {
		synced = append(synced, s)
		if s == 22 {
			return syncFailure
		}
		return nil
	})
	if !errors.Is(err, selectFailure) || !errors.Is(err, syncFailure) || !slices.Equal(selected, []driver.Context{1, 2, 3}) || !slices.Equal(synced, []driver.Stream{22, 33}) {
		t.Fatalf("selected=%v synced=%v err=%v", selected, synced, err)
	}
}

func TestDrainDefaultStreamAndRetainedGraph(t *testing.T) {
	calls := 0
	selectContext := func(driver.Context) error { return nil }
	sync := func(s driver.Stream) error {
		calls++
		if s != 0 {
			t.Fatal(s)
		}
		return nil
	}
	states := []rankState{{context: 1}}
	if err := drainRanks(states, false, selectContext, sync); err != nil || calls != 0 {
		t.Fatal(err, calls)
	}
	if err := drainRanks(states, true, selectContext, sync); err != nil || calls != 1 {
		t.Fatal(err, calls)
	}
	states[0].retainResources = true
	if err := drainRanks(states, true, func(driver.Context) error { t.Fatal("selected retained rank"); return nil }, sync); err == nil || calls != 1 {
		t.Fatal(err, calls)
	}
}

func TestReleaseCommunicatorDependencies(t *testing.T) {
	failure := errors.New("release failed")
	for _, fail := range []string{"", "send", "recv", "destroy"} {
		t.Run(fail, func(t *testing.T) {
			s := rankState{comm: 1, sendRegistration: 2, recvRegistration: 3}
			var calls []string
			err := releaseCommunicator(&s, func(c nccl.Comm, r nccl.Registration) error {
				if c != 1 {
					t.Fatal(c)
				}
				name := "send"
				if r == 3 {
					name = "recv"
				}
				calls = append(calls, name)
				if name == fail {
					return failure
				}
				return nil
			}, func(c nccl.Comm) error {
				if c != 1 {
					t.Fatal(c)
				}
				calls = append(calls, "destroy")
				if fail == "destroy" {
					return failure
				}
				return nil
			})
			want := []string{"send", "recv"}
			if fail == "" || fail == "destroy" {
				want = append(want, "destroy")
			}
			if !slices.Equal(calls, want) || (fail == "" && err != nil) || (fail != "" && !errors.Is(err, failure)) {
				t.Fatalf("calls=%v want=%v error=%v", calls, want, err)
			}
			if (s.comm != 0) != (fail != "") || (s.sendRegistration != 0) != (fail == "send") || (s.recvRegistration != 0) != (fail == "recv") {
				t.Fatalf("incorrect retained handles: %+v", s)
			}
		})
	}
}
