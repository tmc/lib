package cudarepro

import (
	"crypto/sha256"
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// Schedule records the original perturbation options and the retained subset
// of injection decisions. Reduction preserves options and decision order.
// An empty subset replays the pristine source, rather than skipping execution.
type Schedule struct {
	Options   cudaschedule.Options
	Decisions []cudaschedule.InjectionDecision
}

func cloneSchedule(s *Schedule) *Schedule {
	if s == nil {
		return nil
	}
	c := *s
	c.Decisions = slices.Clone(s.Decisions)
	for i := range c.Decisions {
		if p := c.Decisions[i].Site.BlockIndex; p != nil {
			v := *p
			c.Decisions[i].Site.BlockIndex = &v
		}
	}
	return &c
}

// Artifact returns the exact PTX a runner must load for this case. With no
// Schedule it returns Source unchanged. Recorded decisions are validated by
// the campaign producer's Replay API; source and options are never inferred.
func (c Case) Artifact() (string, error) {
	if c.Source == "" {
		return "", fmt.Errorf("replay case: missing source")
	}
	if c.Schedule == nil {
		return c.Source, nil
	}
	r, err := cudaschedule.Replay(c.Source, c.Schedule.Options, c.Schedule.Decisions)
	if err != nil {
		return "", fmt.Errorf("replay case: %w", err)
	}
	return r.PTX, nil
}

func checkScheduleRun(c Case, run cudaschedule.RunRecord) error {
	ptx, err := c.Artifact()
	if err != nil {
		return err
	}
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(ptx)))
	if run.ExpectedPTXDigest != digest || run.PTXDigest != digest {
		return fmt.Errorf("replay case: run does not bind the recorded injection artifact")
	}
	return nil
}
