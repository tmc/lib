package cuptibench

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// validConfig is the minimum Config that validate accepts, so a test that
// expects a rejection is rejected for the reason it is testing.
func validConfig() Config {
	return Config{
		Mode:           Whole,
		Operations:     1,
		MaxConfigBytes: 1024,
		Metrics:        []string{"dram__bytes.sum"},
		Limits: cuptirange.Limits{
			MaxRanges:         8,
			MaxRangeTreeNodes: 8,
			MaxCounterBytes:   1024,
			MaxPasses:         4,
		},
	}
}

// TestCacheEvictedByConflictsWithAllowWarm pins the one combination that makes
// two contradictory claims about the same measurement.
func TestCacheEvictedByConflictsWithAllowWarm(t *testing.T) {
	c := validConfig()
	c.CacheEvictedBy = "4 MB streaming write between passes"
	c.AllowWarmCacheTraffic = true
	err := c.validate()
	if err == nil {
		t.Fatal("validate accepted a config labelling one measurement both cold and warm")
	}
	if !strings.Contains(err.Error(), "CacheEvictedBy") {
		t.Errorf("validate error does not name the field: %v", err)
	}
}

// TestEitherAloneValidates is the positive control: the conflict check must
// reject only the conflict, not either option on its own.
func TestEitherAloneValidates(t *testing.T) {
	for _, tt := range []struct {
		name string
		set  func(*Config)
	}{
		{"neither", func(c *Config) {}},
		{"evicted only", func(c *Config) { c.CacheEvictedBy = "4 MB streaming write" }},
		{"allow warm only", func(c *Config) { c.AllowWarmCacheTraffic = true }},
	} {
		c := validConfig()
		tt.set(&c)
		if err := c.validate(); err != nil {
			t.Errorf("%s: validate() = %v, want nil", tt.name, err)
		}
	}
}
