package inferbench

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestValidateRawSet(t *testing.T) {
	dir, campaign := rawSetFixture(t)
	if err := campaign.ValidateRawSet(dir); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRawSetRejectsCorruption(t *testing.T) {
	tests := []struct {
		name   string
		change func(string, *Campaign)
	}{
		{"missing interval", func(dir string, _ *Campaign) { os.Remove(filepath.Join(dir, "000000.json")) }},
		{"substituted interval", func(dir string, _ *Campaign) { os.WriteFile(filepath.Join(dir, "000000.json"), []byte("{}\n"), 0o600) }},
		{"extra interval", func(dir string, _ *Campaign) { os.WriteFile(filepath.Join(dir, "999999.json"), []byte("{}\n"), 0o600) }},
		{"derived receipt tampering", func(_ string, c *Campaign) { c.Runs[0].Samples[0].PeakMemoryBytes++ }},
		{"schedule mismatch", func(_ string, c *Campaign) {
			c.RawSet.Intervals[0], c.RawSet.Intervals[1] = c.RawSet.Intervals[1], c.RawSet.Intervals[0]
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir, campaign := rawSetFixture(t)
			test.change(dir, &campaign)
			if err := campaign.ValidateRawSet(dir); err == nil {
				t.Fatal("accepted corrupted raw set")
			}
		})
	}
}

func rawSetFixture(t *testing.T) (string, Campaign) {
	t.Helper()
	dir := t.TempDir()
	plan, samples := collectionPlan(t)
	if err := WriteJSONAtomic(filepath.Join(dir, "plan.json"), plan); err != nil {
		t.Fatal(err)
	}
	interval := 0
	campaign, err := Collect(plan, func(k ScheduledSample) (Sample, error) {
		return samples[k], nil
	}, func(raw IntervalReceipt) error {
		path := filepath.Join(dir, fmt.Sprintf("%06d.json", interval))
		interval++
		return WriteJSONAtomic(path, raw)
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := campaign.BindRawSet(dir); err != nil {
		t.Fatal(err)
	}
	return dir, campaign
}
