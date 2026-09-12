package cuptirange

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestCatalogHostVersions(t *testing.T) {
	for _, version := range []uint32{27, 130001, 130200} {
		t.Run(fmt.Sprint(version), func(t *testing.T) {
			closed := 0
			input := []byte{1, 2, 3}
			c, err := newCatalog("chip", input, version, hostAPI{
				initialize: func(v *cupti.ProfilerHostInitializeParams) error {
					want := uint(48)
					if version == 130200 {
						want = 56
					}
					if v.StructSize != want || v.PCounterAvailabilityImage == 0 || v.ProfilerType != cupti.ProfilerTypeRangeProfiler {
						t.Fatalf("initialize=%+v", v)
					}
					v.PHostObject = 9
					return nil
				},
				deinitialize: func(v *cupti.ProfilerHostDeinitializeParams) error {
					closed++
					if v.PHostObject != 9 {
						t.Fatal("lost host")
					}
					return nil
				},
				add: func(*cupti.ProfilerHostConfigAddMetricsParams) error {
					t.Fatal("catalog configured metrics")
					return nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			input[0] = 8
			if c.plan.availability[0] != 1 || c.plan.names != nil || len(c.plan.config) != 0 || c.plan.passes != 0 {
				t.Fatal("catalog owns configuration or aliases input")
			}
			if err := c.Close(); err != nil {
				t.Fatal(err)
			}
			if err := c.Close(); err != nil || closed != 1 {
				t.Fatalf("close=%v calls=%d", err, closed)
			}
		})
	}
}

func TestCatalogFailureOwnership(t *testing.T) {
	want := errors.New("native failure")
	closes := 0
	c, err := newCatalog("chip", nil, 27, hostAPI{
		initialize: func(v *cupti.ProfilerHostInitializeParams) error { v.PHostObject = 3; return want },
		deinitialize: func(*cupti.ProfilerHostDeinitializeParams) error {
			closes++
			if closes == 1 {
				return want
			}
			return nil
		},
	})
	if c == nil || !errors.Is(err, want) || c.plan.host != 3 {
		t.Fatalf("catalog=%v err=%v", c, err)
	}
	if err := c.Close(); err != nil || c.plan.host != 0 || closes != 2 {
		t.Fatalf("close=%v calls=%d", err, closes)
	}
	if _, err := newCatalog("chip", nil, 999, hostAPI{initialize: func(*cupti.ProfilerHostInitializeParams) error { t.Fatal("unknown ABI called"); return nil }}); err == nil {
		t.Fatal("unknown version accepted")
	}
	if _, err := newCatalog("chip", nil, 27, hostAPI{initialize: func(*cupti.ProfilerHostInitializeParams) error { return nil }}); err == nil {
		t.Fatal("null host accepted")
	}
}

func ExampleNewCatalog() {
	_, err := NewCatalog("", nil)
	fmt.Println(err)
	// Output: cuptirange: invalid chip
}

func ExampleCatalog() {
	var c Catalog
	fmt.Println(c.Close())
	// Output: <nil>
}

func ExampleCatalog_Close() {
	var c *Catalog
	fmt.Println(c.Close())
	// Output: <nil>
}

func ExampleCatalog_BaseMetrics() {
	var c *Catalog
	_, err := c.BaseMetrics(cupti.MetricTypeCounter, 10000, 1024)
	fmt.Println(err)
	// Output: cuptirange: nil catalog
}

func ExampleCatalog_SubMetrics() {
	var c *Catalog
	_, err := c.SubMetrics("sm__cycles", cupti.MetricTypeCounter, 100, 1024)
	fmt.Println(err)
	// Output: cuptirange: nil catalog
}

func ExampleCatalog_Properties() {
	var c *Catalog
	_, err := c.Properties("sm__cycles", 4096)
	fmt.Println(err)
	// Output: cuptirange: nil catalog
}
