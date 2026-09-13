package cuptirange

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestPlanFailures(t *testing.T) {
	for _, stage := range []string{"initialize", "add", "size", "empty", "oversize", "config", "passes", "zero passes", "success"} {
		t.Run(stage, func(t *testing.T) {
			failure := errors.New("native failure")
			closed := 0
			api := hostAPI{
				version: func() (uint32, error) { return 130200, nil },
				initialize: func(p *cupti.ProfilerHostInitializeParams) error {
					if stage == "initialize" {
						return failure
					}
					p.PHostObject = 1
					return nil
				},
				deinitialize: func(p *cupti.ProfilerHostDeinitializeParams) error {
					closed++
					if p.PHostObject != 1 {
						t.Error("wrong host")
					}
					return nil
				},
				add: func(p *cupti.ProfilerHostConfigAddMetricsParams) error {
					if p.NumMetrics != 1 || p.PpMetricNames == 0 {
						t.Error("missing metric storage")
					}
					if stage == "add" {
						return failure
					}
					return nil
				},
				size: func(p *cupti.ProfilerHostGetConfigImageSizeParams) error {
					p.ConfigImageSize = 16
					if stage == "empty" {
						p.ConfigImageSize = 0
					}
					if stage == "oversize" {
						p.ConfigImageSize = 65
					}
					if stage == "size" {
						return failure
					}
					return nil
				},
				config: func(p *cupti.ProfilerHostGetConfigImageParams) error {
					if p.ConfigImageSize != 16 || p.PConfigImage == 0 {
						t.Error("missing output storage")
					}
					if stage == "config" {
						return failure
					}
					return nil
				},
				passes: func(p *cupti.ProfilerHostGetNumOfPassesParams) error {
					p.NumOfPasses = 2
					if stage == "zero passes" {
						p.NumOfPasses = 0
					}
					if stage == "passes" {
						return failure
					}
					return nil
				},
			}
			p, err := newPlan("chip", nil, []string{"metric"}, 64, api)
			if stage == "success" {
				if err != nil {
					t.Fatal(err)
				}
				image, passes := p.Configuration()
				if len(image) != 16 || passes != 2 {
					t.Fatal("wrong configuration")
				}
				image[0] = 42
				again, _ := p.Configuration()
				if again[0] != 0 {
					t.Fatal("configuration aliases storage")
				}
				if err := p.Close(); err != nil {
					t.Fatal(err)
				}
			} else if err == nil || p != nil {
				t.Fatalf("plan=%v error=%v", p, err)
			}
			want := 1
			if stage == "initialize" {
				want = 0
			}
			if closed != want {
				t.Fatalf("deinitializations=%d want %d", closed, want)
			}
		})
	}
}

func TestPlanCleanupRetry(t *testing.T) {
	failure := errors.New("native failure")
	closes := 0
	api := hostAPI{
		version:    func() (uint32, error) { return 130200, nil },
		initialize: func(p *cupti.ProfilerHostInitializeParams) error { p.PHostObject = 9; return nil },
		add:        func(*cupti.ProfilerHostConfigAddMetricsParams) error { return failure },
		deinitialize: func(p *cupti.ProfilerHostDeinitializeParams) error {
			closes++
			if p.PHostObject != 9 {
				t.Error("lost native host")
			}
			if closes < 3 {
				return failure
			}
			return nil
		},
	}
	p, err := newPlan("chip", []byte{1, 2}, []string{"metric"}, 64, api)
	if p == nil || !errors.Is(err, failure) {
		t.Fatalf("plan=%v error=%v", p, err)
	}
	if p.names == nil || p.host != 9 {
		t.Fatal("cleanup failure lost ownership")
	}
	if err := p.Close(); !errors.Is(err, failure) {
		t.Fatalf("retry=%v", err)
	}
	if p.names == nil || p.host != 9 {
		t.Fatal("retry failure released inputs")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := p.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if closes != 3 || p.names != nil || p.host != 0 {
		t.Fatalf("closes=%d host=%d names retained=%v", closes, p.host, p.names != nil)
	}
}

func TestPlanRuntimeHostLayout(t *testing.T) {
	stop := errors.New("stop after host layout check")
	for _, tt := range []struct {
		name     string
		version  uint32
		queryErr error
		wantSize uint
	}{
		{"CUDA 12.9", 28, nil, 48},
		{"CUDA 13.0", 130001, nil, 48},
		// 13.1 extends the properties struct only, so initialization keeps
		// the 13.0 size while 13.2 and later grow to 56.
		{"CUDA 13.1", 130101, nil, 48},
		{"generated 13.2", 130200, nil, 56},
		{"CUDA 13.2", 130202, nil, 56},
		{"CUDA 13.3", 130301, nil, 56},
		{"CUDA 13.4", 130400, nil, 56},
		{"unknown version", 27, nil, 0},
		{"unreported 13.1 encoding", 130100, nil, 0},
		{"version failure", 0, stop, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			p, err := newPlan("chip", nil, []string{"metric"}, 64, hostAPI{
				version: func() (uint32, error) { return tt.version, tt.queryErr },
				initialize: func(p *cupti.ProfilerHostInitializeParams) error {
					calls++
					if tt.wantSize == 0 || p.StructSize != tt.wantSize {
						t.Fatalf("native initialization size %d, want %d", p.StructSize, tt.wantSize)
					}
					return stop
				},
			})
			if p != nil || err == nil {
				t.Fatalf("plan=%v err=%v", p, err)
			}
			if tt.wantSize == 0 && calls != 0 || tt.wantSize != 0 && calls != 1 {
				t.Fatalf("native calls=%d", calls)
			}
			if tt.queryErr != nil && !errors.Is(err, stop) {
				t.Fatal("lost version error")
			}
		})
	}
}
