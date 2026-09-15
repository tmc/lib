package cuptirange

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cupticallback"
)

func launchEvent(symbol string) *cupticallback.Event {
	return &cupticallback.Event{Domain: cupti.CbDomainDriverApi, ID: uint32(cupti.DriverCBIDCuLaunchKernel), Api: &cupticallback.ApiData{Context: 7, CallbackSite: cupti.ApiEnter, FunctionName: "cuLaunchKernel", SymbolName: symbol}}
}

func TestKernelNamesWindowContextAndNativeOrdinal(t *testing.T) {
	k := &kernelNames{context: 7, limit: 4}
	k.callback(launchEvent("prepare"))
	for range 2 {
		if err := k.begin(); err != nil {
			t.Fatal(err)
		}
		wrongContext := launchEvent("other-context")
		wrongContext.Api.Context = 8
		k.callback(wrongContext)
		runtimeDuplicate := launchEvent("runtime")
		runtimeDuplicate.Domain = cupti.CbDomainRuntimeApi
		k.callback(runtimeDuplicate)
		exit := launchEvent("exit")
		exit.Api.CallbackSite = cupti.ApiExit
		k.callback(exit)
		k.callback(launchEvent("project"))
		k.callback(launchEvent("decode"))
		if err := k.end(); err != nil {
			t.Fatal(err)
		}
		k.callback(launchEvent("outside"))
	}
	// Evaluation index order is not used as launch order: native ordinals are.
	ranges := []RangeInfo{{Name: "1"}, {Name: "0"}}
	if err := k.resolve(ranges); err != nil {
		t.Fatal(err)
	}
	if ranges[0].Name != "decode" || ranges[0].NativeName != "1" || ranges[1].Name != "project" {
		t.Fatalf("ranges=%+v", ranges)
	}
}

func TestKernelNamesRejectsIncompleteAndChangedTrace(t *testing.T) {
	for _, tc := range []struct {
		name   string
		native []string
	}{
		{"missing", []string{"0"}}, {"extra", []string{"0", "1", "2"}},
		{"duplicate", []string{"0", "0"}}, {"not-ordinal", []string{"project", "decode"}},
		{"out-of-range", []string{"0", "2"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			k := &kernelNames{context: 7, limit: 4}
			_ = k.begin()
			k.callback(launchEvent("project"))
			k.callback(launchEvent("decode"))
			if err := k.end(); err != nil {
				t.Fatal(err)
			}
			var ranges []RangeInfo
			for _, n := range tc.native {
				ranges = append(ranges, RangeInfo{Name: n})
			}
			if err := k.resolve(ranges); err == nil {
				t.Fatal("accepted incomplete or ambiguous mapping")
			}
		})
	}
	k := &kernelNames{context: 7, limit: 4}
	_ = k.begin()
	k.callback(launchEvent("project"))
	if err := k.end(); err != nil {
		t.Fatal(err)
	}
	_ = k.begin()
	k.callback(launchEvent("decode"))
	if err := k.end(); err == nil {
		t.Fatal("accepted changed replay launch trace")
	}
}

func TestKernelNamesRejectsUnknownGraphAndCapacity(t *testing.T) {
	for _, name := range []string{"missing-symbol", "capacity", "graph"} {
		t.Run(name, func(t *testing.T) {
			k := &kernelNames{context: 7, limit: 1}
			_ = k.begin()
			switch name {
			case "missing-symbol":
				k.callback(launchEvent(""))
			case "capacity":
				k.callback(launchEvent("a"))
				k.callback(launchEvent("b"))
			case "graph":
				e := launchEvent("")
				e.Api.FunctionName = "cuGraphLaunch"
				e.ID = uint32(cupti.DriverCBIDCuGraphLaunch)
				k.callback(e)
			}
			if err := k.end(); err == nil {
				t.Fatal("accepted unproved identity trace")
			}
		})
	}
}

type retrySubscription struct {
	calls int
	fail  bool
}

func (s *retrySubscription) Unsubscribe() error {
	s.calls++
	if s.fail {
		return errors.New("unsubscribe")
	}
	return nil
}
func TestKernelNamesRetainsFailedSubscription(t *testing.T) {
	sub := &retrySubscription{fail: true}
	k := &kernelNames{subscription: sub}
	if err := k.Close(); err == nil || !strings.Contains(err.Error(), "unsubscribe") {
		t.Fatal(err)
	}
	if k.subscription != sub {
		t.Fatal("lost callback subscription")
	}
	sub.fail = false
	if err := k.Close(); err != nil {
		t.Fatal(err)
	}
	if err := k.Close(); err != nil || sub.calls != 2 {
		t.Fatalf("close err=%v calls=%d", err, sub.calls)
	}
}

func TestLegacyCaptureChecksIndependentLaunchCount(t *testing.T) {
	for _, count := range []uint{1, 2, 3} {
		t.Run(strconv.Itoa(int(count)), func(t *testing.T) {
			k := &kernelNames{context: 7, limit: 8}
			c := &counterStore{host: 1, image: make([]byte, 32), kernelNames: k}
			defer c.pin.Unpin()
			api := collectionAPI{
				config: func(*cupti.RangeProfilerSetConfigParams) error { return nil },
				start:  func(*cupti.RangeProfilerStartParams) error { return nil },
				stop:   func(p *cupti.RangeProfilerStopParams) error { p.IsAllPassSubmitted = 1; return nil },
				decode: func(*cupti.RangeProfilerDecodeDataParams) error { return nil },
				info:   func(p *cupti.RangeProfilerGetCounterDataInfoParams) error { p.NumTotalRanges = count; return nil },
			}
			replay := Replay{Synchronize: func() error { return nil }, Run: func(uint, *Scope) error {
				k.callback(launchEvent("a"))
				k.callback(launchEvent("b"))
				return nil
			}}
			_, records, err := c.captureReplayWithAPI([]byte{1}, Limits{MaxRanges: 8, MaxPasses: 2}, CaptureOptions{Mode: Kernels, MaxDepth: 1}, replay, false, api, nil, nil, nil)
			if (err == nil) != (count == 2) {
				t.Fatalf("count=%d err=%v", count, err)
			}
			if len(records) != 0 {
				t.Fatal("legacy capture unexpectedly queried identities")
			}
		})
	}
}
