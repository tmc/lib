package inferbench

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
)

const PeakDeviceMemoryMetric = "peak-device-allocated-bytes-within-synchronized-interval-v1"

// RawSetManifest binds a campaign to every independently written interval.
type RawSetManifest struct {
	PlanSHA256 string                `json:"plan_sha256"`
	Intervals  []RawIntervalArtifact `json:"intervals"`
}

// RawIntervalArtifact identifies one raw interval file.
type RawIntervalArtifact struct {
	Index  int    `json:"index"`
	Arm    string `json:"arm"`
	ID     string `json:"id"`
	File   string `json:"file"`
	SHA256 string `json:"sha256"`
}

func (c *Campaign) validateRawManifest() error {
	plan := campaignPlan(*c)
	wantPlan, err := HashJSON(plan)
	if err != nil || c.RawSet.PlanSHA256 != wantPlan {
		return errors.New("raw set plan hash differs")
	}
	if len(c.RawSet.Intervals) != len(c.Order) {
		return errors.New("raw set does not cover every interval")
	}
	files := make(map[string]bool)
	for i, artifact := range c.RawSet.Intervals {
		want := c.Order[i]
		if artifact.Index != i || artifact.Arm != want.Arm || artifact.ID != want.ID ||
			artifact.File == "" || filepath.Base(artifact.File) != artifact.File || files[artifact.File] || !validSHA256(artifact.SHA256) {
			return errors.New("invalid raw interval manifest")
		}
		files[artifact.File] = true
	}
	return nil
}

// BindRawSet records the exact raw interval files belonging to c.
func (c *Campaign) BindRawSet(dir string) error {
	plan := campaignPlan(*c)
	planSHA256, err := HashJSON(plan)
	if err != nil {
		return fmt.Errorf("bind raw set: hash plan: %w", err)
	}
	manifest := RawSetManifest{PlanSHA256: planSHA256}
	for i, scheduled := range c.Order {
		name := fmt.Sprintf("%06d.json", i)
		sum, err := HashFile(filepath.Join(dir, name))
		if err != nil {
			return fmt.Errorf("bind raw set: interval %d: %w", i, err)
		}
		manifest.Intervals = append(manifest.Intervals, RawIntervalArtifact{Index: i, Arm: scheduled.Arm, ID: scheduled.ID, File: name, SHA256: sum})
	}
	c.RawSet = manifest
	return c.validateRawManifest()
}

// ValidateRawSet verifies the manifest, the preregistered plan, and every raw
// interval against the derived campaign.
func (c *Campaign) ValidateRawSet(dir string) error {
	if err := c.validate(true); err != nil {
		return err
	}
	planPath := filepath.Join(dir, "plan.json")
	var plan Campaign
	if err := readJSON(planPath, &plan); err != nil {
		return fmt.Errorf("raw set: plan: %w", err)
	}
	plan = campaignPlan(plan)
	if got, err := HashJSON(plan); err != nil || got != c.RawSet.PlanSHA256 || !reflect.DeepEqual(plan, campaignPlan(*c)) {
		return errors.New("raw set: preregistered plan differs")
	}
	wantNames := map[string]bool{"plan.json": true}
	for i, artifact := range c.RawSet.Intervals {
		path := filepath.Join(dir, artifact.File)
		got, err := HashFile(path)
		if err != nil || got != artifact.SHA256 {
			return fmt.Errorf("raw set: interval %d hash differs", i)
		}
		var raw IntervalReceipt
		if err := readJSON(path, &raw); err != nil {
			return fmt.Errorf("raw set: interval %d: %w", i, err)
		}
		run := runByArm(c.Runs, artifact.Arm)
		sample := -1
		if run != nil {
			sample = sampleIndex(run.Samples, artifact.ID)
		}
		if run == nil || sample < 0 || raw.SchemaVersion != SchemaVersion || raw.PlanSHA256 != c.RawSet.PlanSHA256 || raw.Index != i || raw.Arm != artifact.Arm || raw.Sample.ID != artifact.ID {
			return fmt.Errorf("raw set: interval %d differs from campaign", i)
		}
		if raw.Identity != run.Identity || !reflect.DeepEqual(raw.Workload, run.Workload) || raw.Evidence != run.Evidence {
			return fmt.Errorf("raw set: interval %d execution contract differs", i)
		}
		if !reflect.DeepEqual(raw.Sample, run.Samples[sample]) {
			return fmt.Errorf("raw set: interval %d sample differs", i)
		}
		wantNames[artifact.File] = true
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("raw set: read directory: %w", err)
	}
	for _, entry := range entries {
		if !entry.Type().IsRegular() || !wantNames[entry.Name()] {
			return fmt.Errorf("raw set: unexpected artifact %q", entry.Name())
		}
	}
	if len(entries) != len(wantNames) {
		return errors.New("raw set: missing or non-regular artifact")
	}
	return nil
}

func campaignPlan(c Campaign) Campaign {
	c.RawSet = RawSetManifest{}
	c.Runs = append([]Run(nil), c.Runs...)
	for i := range c.Runs {
		c.Runs[i].Samples = append([]Sample(nil), c.Runs[i].Samples...)
		for j := range c.Runs[i].Samples {
			c.Runs[i].Samples[j] = Sample{ID: c.Runs[i].Samples[j].ID}
		}
	}
	return c
}

func sampleIndex(samples []Sample, id string) int {
	for i := range samples {
		if samples[i].ID == id {
			return i
		}
	}
	return -1
}

func readJSON(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		return errors.New("trailing data")
	}
	return nil
}
