package cuptirange

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestCollectGates(t *testing.T) {
	for _, mode := range []string{"success", "config", "start", "run", "stop", "incomplete", "unwritten completion", "decode", "dropped", "large loss", "info", "empty", "overflow", "capacity"} {
		t.Run(mode, func(t *testing.T) {
			c := counterStore{host: 1, image: make([]byte, 32)}
			defer c.pin.Unpin()
			c.pin.Pin(&c.image[0])
			failure := errors.New("native failure")
			decoded, inventoried := false, false
			runs := 0
			api := collectionAPI{
				config: func(p *cupti.RangeProfilerSetConfigParams) error {
					if p.ReplayMode != cupti.UserReplay || p.Range != cupti.AutoRange || p.ConfigSize != 3 || p.CounterDataImageSize != 32 {
						t.Error("wrong collection configuration")
					}
					if mode == "config" {
						return failure
					}
					return nil
				},
				start: func(*cupti.RangeProfilerStartParams) error {
					if mode == "start" {
						return failure
					}
					return nil
				},
				stop: func(p *cupti.RangeProfilerStopParams) error {
					if mode == "incomplete" {
						p.IsAllPassSubmitted = 0
					} else if mode != "unwritten completion" {
						p.IsAllPassSubmitted = 1
					}
					if mode == "stop" {
						return failure
					}
					return nil
				},
				decode: func(p *cupti.RangeProfilerDecodeDataParams) error {
					decoded = true
					if mode == "dropped" {
						p.NumOfRangeDropped = 1
					}
					if mode == "large loss" {
						p.NumOfRangeDropped = ^uint(0) - 7
					}
					if mode == "decode" {
						return failure
					}
					return nil
				},
				info: func(p *cupti.RangeProfilerGetCounterDataInfoParams) error {
					inventoried = true
					p.NumTotalRanges = 2
					if mode == "empty" {
						p.NumTotalRanges = 0
					}
					if mode == "capacity" {
						p.NumTotalRanges = 4
					}
					if mode == "overflow" {
						p.NumTotalRanges = 5
					}
					if mode == "info" {
						return failure
					}
					return nil
				},
			}
			count, err := c.collect([]byte{1, 2, 3}, 4, 2, func(uint) error {
				runs++
				if mode == "run" {
					return failure
				}
				return nil
			}, api)
			if mode == "success" {
				if err != nil || count != 2 {
					t.Fatalf("count=%d error=%v", count, err)
				}
			} else if err == nil || count != 0 {
				t.Fatalf("invalid collection accepted: count=%d error=%v", count, err)
			}
			if mode == "start" || mode == "stop" {
				call := "RangeProfilerStart"
				if mode == "stop" {
					call = "RangeProfilerStop"
				}
				if !errors.Is(err, failure) || !strings.Contains(err.Error(), call) {
					t.Fatalf("native call error lost identity/context: %v", err)
				}
			}
			switch mode {
			case "config", "start", "run", "stop", "incomplete", "unwritten completion":
				if decoded || inventoried {
					t.Fatal("failed replay reached decoding")
				}
			case "decode", "dropped", "large loss":
				if inventoried {
					t.Fatal("failed decode reached inventory")
				}
			}
			if mode == "unwritten completion" && runs != 1 {
				t.Fatalf("unwritten completion permitted %d runs", runs)
			}
			if mode == "incomplete" && runs != 2 {
				t.Fatalf("explicit incomplete flag ran %d passes, want budget 2", runs)
			}
			if mode == "success" && c.quality != (CollectionQuality{PassesStarted: 1, PassesStopped: 1, AllPassesSubmitted: true, Decoded: true}) {
				t.Fatalf("success quality=%+v", c.quality)
			}
			if mode == "dropped" && (!c.quality.Decoded || c.quality.DroppedRanges != 1) {
				t.Fatalf("lost drop count: %+v", c.quality)
			}
			if mode == "capacity" && (!c.quality.Decoded || c.quality.DroppedRanges != 0) {
				t.Fatalf("saturation invented native loss: %+v", c.quality)
			}
			if mode == "large loss" && (!c.quality.Decoded || c.quality.DroppedRanges != uint64(^uint(0)-7)) {
				t.Fatalf("large loss changed: %+v", c.quality)
			}
			if mode == "decode" && c.quality.Decoded {
				t.Fatal("failed decode represented as observed loss count")
			}
			if mode == "stop" && (c.quality.PassesStopped != 0 || c.quality.AllPassesSubmitted) {
				t.Fatalf("failed stop promoted progress: %+v", c.quality)
			}
		})
	}
}

func TestUserRangeCapacityIsNotAutomaticTruncation(t *testing.T) {
	c := counterStore{host: 1, image: make([]byte, 32)}
	defer c.pin.Unpin()
	api := collectionAPI{
		config: func(*cupti.RangeProfilerSetConfigParams) error { return nil },
		start:  func(*cupti.RangeProfilerStartParams) error { return nil },
		stop:   func(p *cupti.RangeProfilerStopParams) error { p.IsAllPassSubmitted = 1; return nil },
		decode: func(*cupti.RangeProfilerDecodeDataParams) error { return nil },
		info:   func(p *cupti.RangeProfilerGetCounterDataInfoParams) error { p.NumTotalRanges = 2; return nil },
	}
	count, err := c.collectReplay([]byte{1}, 2, 1, 1, cupti.UserRange, nil, func(uint) error { return nil }, api)
	if err != nil || count != 2 {
		t.Fatalf("user ranges count=%d error=%v", count, err)
	}
	// User capture subsequently reconciles every Scope identity; unlike automatic
	// selection it has an independent cardinality witness. Never invent native loss.
	if c.quality.DroppedRanges != 0 {
		t.Fatal("invented dropped-range count")
	}
}

func TestKernelReplayRunsHostOnce(t *testing.T) {
	for _, complete := range []bool{false, true} {
		t.Run(fmt.Sprint(complete), func(t *testing.T) {
			c := counterStore{host: 1, image: make([]byte, 32)}
			defer c.pin.Unpin()
			prepare, run := 0, 0
			api := collectionAPI{
				config: func(p *cupti.RangeProfilerSetConfigParams) error {
					if p.ReplayMode != cupti.KernelReplay {
						t.Fatalf("replay mode=%v", p.ReplayMode)
					}
					return nil
				},
				start: func(*cupti.RangeProfilerStartParams) error { return nil },
				stop: func(p *cupti.RangeProfilerStopParams) error {
					p.IsAllPassSubmitted = 0
					if complete {
						p.IsAllPassSubmitted = 1
					}
					return nil
				},
				decode: func(*cupti.RangeProfilerDecodeDataParams) error {
					if !complete {
						t.Fatal("incomplete replay decoded")
					}
					return nil
				},
				info: func(p *cupti.RangeProfilerGetCounterDataInfoParams) error { p.NumTotalRanges = 1; return nil },
			}
			_, err := c.collectReplayMode([]byte{1}, 4, 8, 1, cupti.AutoRange, cupti.KernelReplay, func(uint) error { prepare++; return nil }, func(uint) error { run++; return nil }, api)
			if (err == nil) != complete || prepare != 1 || run != 1 || c.quality.PassesStarted != 1 || c.quality.PassesStopped != 1 {
				t.Fatalf("complete=%v prepare=%d run=%d quality=%+v error=%v", complete, prepare, run, c.quality, err)
			}
		})
	}
}

func TestKernelReplayRejectsUnsupportedRanges(t *testing.T) {
	for _, test := range []struct {
		mode  cupti.ProfilerRange
		depth uint16
	}{{cupti.UserRange, 1}, {cupti.AutoRange, 2}} {
		c := counterStore{}
		if _, err := c.collectReplayMode([]byte{1}, 4, 8, test.depth, test.mode, cupti.KernelReplay, nil, func(uint) error { t.Fatal("ran rejected replay"); return nil }, collectionAPI{}); err == nil {
			t.Fatal("accepted unsupported replay")
		}
	}
}

func TestAutomaticRangeCapacityWitness(t *testing.T) {
	for _, test := range []struct {
		name                       string
		capacity, native, launches uint
		witness, want              bool
	}{
		{"no witness saturated", 2, 2, 0, false, false},
		{"exact witness saturated", 2, 2, 2, true, true},
		{"mismatch below capacity", 4, 2, 3, true, false},
		{"mismatch saturated", 2, 2, 3, true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			c := counterStore{host: 1, image: make([]byte, 32)}
			defer c.pin.Unpin()
			if test.witness {
				c.kernelNames = &kernelNames{passes: 1, expected: make([]kernelLaunch, test.launches)}
			}
			api := collectionAPI{
				config: func(*cupti.RangeProfilerSetConfigParams) error { return nil },
				start:  func(*cupti.RangeProfilerStartParams) error { return nil },
				stop:   func(p *cupti.RangeProfilerStopParams) error { p.IsAllPassSubmitted = 1; return nil },
				decode: func(*cupti.RangeProfilerDecodeDataParams) error { return nil },
				info:   func(p *cupti.RangeProfilerGetCounterDataInfoParams) error { p.NumTotalRanges = test.native; return nil },
			}
			count, err := c.collect([]byte{1}, test.capacity, 1, func(uint) error { return nil }, api)
			if (err == nil) != test.want {
				t.Fatalf("count=%d error=%v", count, err)
			}
			if test.want && count != test.native {
				t.Fatalf("count=%d", count)
			}
			if !c.quality.Decoded || c.quality.DroppedRanges != 0 {
				t.Fatalf("fabricated loss or lost decode evidence: %+v", c.quality)
			}
		})
	}
}
