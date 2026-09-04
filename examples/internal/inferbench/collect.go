package inferbench

import (
	"errors"
	"fmt"
	"reflect"
)

// IntervalReceipt is the raw record written after one synchronized interval.
type IntervalReceipt struct {
	SchemaVersion int      `json:"schema_version"`
	PlanSHA256    string   `json:"plan_sha256"`
	Index         int      `json:"index"`
	Arm           string   `json:"arm"`
	Identity      Identity `json:"identity"`
	Workload      Workload `json:"workload"`
	Evidence      Evidence `json:"evidence"`
	Sample        Sample   `json:"sample"`
}

// Collect executes a preregistered campaign in its seeded order. The runs must
// contain sample IDs but no measured data. writeRaw must durably record each
// synchronized interval before Collect accepts the next one.
func Collect(c Campaign, execute func(ScheduledSample) (Sample, error), writeRaw func(IntervalReceipt) error) (Campaign, error) {
	c = cloneCampaign(c)
	if c.SchemaVersion != SchemaVersion || c.RandomSeed == 0 || len(c.Runs) < 2 {
		return Campaign{}, errors.New("inferbench: incomplete campaign plan")
	}
	if execute == nil || writeRaw == nil {
		return Campaign{}, errors.New("inferbench: missing campaign callback")
	}
	var workload Workload
	var identity Identity
	var evidence Evidence
	seenArms := make(map[string]bool)
	for i := range c.Runs {
		r := &c.Runs[i]
		if r.Arm == "" || seenArms[r.Arm] {
			return Campaign{}, fmt.Errorf("inferbench: invalid arm %q", r.Arm)
		}
		seenArms[r.Arm] = true
		if err := validateRunHeader(r); err != nil {
			return Campaign{}, fmt.Errorf("inferbench: arm %q: %w", r.Arm, err)
		}
		if len(r.Samples) < 20 {
			return Campaign{}, fmt.Errorf("inferbench: arm %q: only %d planned samples", r.Arm, len(r.Samples))
		}
		ids := make(map[string]bool)
		for _, s := range r.Samples {
			if s.ID == "" || ids[s.ID] {
				return Campaign{}, fmt.Errorf("inferbench: arm %q: invalid planned sample id", r.Arm)
			}
			ids[s.ID] = true
		}
		if i == 0 {
			workload, identity, evidence = r.Workload, r.Identity, r.Evidence
		} else if !reflect.DeepEqual(workload, r.Workload) {
			return Campaign{}, fmt.Errorf("inferbench: arm %q: unequal workload", r.Arm)
		} else if r.Identity != identity {
			return Campaign{}, fmt.Errorf("inferbench: arm %q: unequal identity", r.Arm)
		} else if r.Evidence != evidence {
			return Campaign{}, fmt.Errorf("inferbench: arm %q: unequal execution evidence", r.Arm)
		}
	}

	wantOrder := Schedule(c.Runs, c.RandomSeed)
	if !reflect.DeepEqual(c.Order, wantOrder) {
		return Campaign{}, errors.New("inferbench: campaign order was not preregistered from seed")
	}
	order := append([]ScheduledSample(nil), c.Order...)
	planSHA256, err := HashJSON(campaignPlan(c))
	if err != nil {
		return Campaign{}, fmt.Errorf("inferbench: hash campaign plan: %w", err)
	}
	for index, scheduled := range order {
		sample, err := execute(scheduled)
		if err != nil {
			return Campaign{}, fmt.Errorf("inferbench: collect %s/%s: %w", scheduled.Arm, scheduled.ID, err)
		}
		if sample.ID != scheduled.ID {
			return Campaign{}, fmt.Errorf("inferbench: collect %s/%s: returned sample %q", scheduled.Arm, scheduled.ID, sample.ID)
		}
		if err := validateSample(&sample, workload); err != nil {
			return Campaign{}, fmt.Errorf("inferbench: collect %s/%s: %w", scheduled.Arm, scheduled.ID, err)
		}
		run := runByArm(c.Runs, scheduled.Arm)
		if run == nil {
			return Campaign{}, fmt.Errorf("inferbench: missing arm %q", scheduled.Arm)
		}
		if err := writeRaw(IntervalReceipt{SchemaVersion: SchemaVersion, PlanSHA256: planSHA256, Index: index, Arm: scheduled.Arm, Identity: identity, Workload: workload, Evidence: evidence, Sample: sample}); err != nil {
			return Campaign{}, fmt.Errorf("inferbench: write raw interval: %w", err)
		}
		if !replaceSample(run.Samples, sample) {
			return Campaign{}, fmt.Errorf("inferbench: missing sample %s/%s", scheduled.Arm, scheduled.ID)
		}
	}
	if err := c.validate(false); err != nil {
		return Campaign{}, fmt.Errorf("inferbench: validate collected campaign: %w", err)
	}
	return c, nil
}

func cloneCampaign(c Campaign) Campaign {
	c.Order = append([]ScheduledSample(nil), c.Order...)
	c.Runs = append([]Run(nil), c.Runs...)
	for i := range c.Runs {
		c.Runs[i].Samples = append([]Sample(nil), c.Runs[i].Samples...)
	}
	return c
}

func replaceSample(samples []Sample, sample Sample) bool {
	for i := range samples {
		if samples[i].ID == sample.ID {
			samples[i] = sample
			return true
		}
	}
	return false
}

func runByArm(runs []Run, arm string) *Run {
	for i := range runs {
		if runs[i].Arm == arm {
			return &runs[i]
		}
	}
	return nil
}
