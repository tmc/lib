package cudasearch

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"testing"
)

func TestProfileDecisionIdentity(t *testing.T) {
	base := freezeProfileActions(actionProfile()).decision
	if base == nil || base.ID == "" || base.SnapshotSHA256 == "" || len(base.ActiveRules) != 1 || base.ActiveRules[0] != "z" {
		t.Fatalf("incomplete decision: %+v", base)
	}
	if other := freezeProfileActions(actionProfile()).decision; other.ID != base.ID {
		t.Fatal("identical decisions differ")
	}
	for _, kind := range []string{"threshold", "unit", "rule", "snapshot", "quality", "time"} {
		t.Run(kind, func(t *testing.T) {
			cfg := actionProfile()
			switch kind {
			case "threshold":
				cfg.Priors[0].Minimum++
			case "unit":
				cfg.Priors[0].Unit = "cycles"
			case "rule":
				cfg.Priors[0].Rule = "other"
			case "snapshot":
				cfg.Snapshot.Metrics["reason"]++
			case "quality":
				cfg.MaxLosses++
			case "time":
				cfg.Time = cfg.Time.Add(1)
			}
			d := freezeProfileActions(cfg).decision
			if d.ID == "" || d.ID == base.ID {
				t.Fatalf("identity omits %s: %+v", kind, d)
			}
		})
	}
	encoded, err := json.Marshal(base)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ProfileDecision
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	id := decoded.ID
	decoded.ID = ""
	canonical, err := json.Marshal(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(canonical)); got != id {
		t.Fatalf("round-trip identity=%s want=%s", got, id)
	}
}

func TestProfileDecisionRetainsFallbackAndCopiesPolicy(t *testing.T) {
	cfg := actionProfile()
	d := freezeProfileActions(cfg).decision
	cfg.Priors[0].Minimum = 999
	if d.Priors[0].Minimum != 10 {
		t.Fatal("decision aliases policy")
	}
	cfg.MinCoverage = math.NaN()
	frozen := freezeProfileActions(cfg)
	if frozen.decision.ID == "" || frozen.decision.MinCoverage != "NaN" || frozen.decision.Fallback != frozen.fallback || frozen.fallback == "" {
		t.Fatalf("invalid policy disappeared: %+v", frozen.decision)
	}
}
