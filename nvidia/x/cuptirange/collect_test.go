package cuptirange

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestCollectGates(t *testing.T) {
	for _, mode := range []string{"success", "config", "start", "run", "stop", "incomplete", "unwritten completion", "decode", "dropped", "large loss", "info", "empty", "overflow"} {
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
