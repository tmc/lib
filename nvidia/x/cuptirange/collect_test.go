package cuptirange

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestCollectGates(t *testing.T) {
	for _, mode := range []string{"success", "config", "start", "run", "stop", "incomplete", "decode", "dropped", "info", "empty", "overflow"} {
		t.Run(mode, func(t *testing.T) {
			c := counterStore{host: 1, image: make([]byte, 32)}
			defer c.pin.Unpin()
			c.pin.Pin(&c.image[0])
			failure := errors.New("native failure")
			decoded, inventoried := false, false
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
					if mode != "incomplete" {
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
			case "config", "start", "run", "stop", "incomplete":
				if decoded || inventoried {
					t.Fatal("failed replay reached decoding")
				}
			case "decode", "dropped":
				if inventoried {
					t.Fatal("failed decode reached inventory")
				}
			}
		})
	}
}
