package cudarepro

import (
	"context"
	"crypto/sha256"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func reductionCase() Case {
	return Case{Source: "host fixture: immutable kernel source", Inputs: []uint64{64, 32, 16, 8}, Shape: []uint64{64, 32}, Grid: [3]uint32{8, 1, 1}, Block: [3]uint32{64, 1, 1}, SharedMem: 256}
}

func hostReplay(_ context.Context, c Case) (cudaschedule.RunRecord, error) {
	r := mismatchRecord(0)
	if len(c.Inputs) < 1 || c.Inputs[0] < 2 || c.Shape[0] < 2 || c.Grid[0] < 2 || c.Block[0] < 2 {
		r.Stdout = "PASS"
	}
	return r, nil
}

func TestReduce_ShrinksLaunchAndInputKeepingIdentity(t *testing.T) {
	original := reductionCase()
	r, err := Reduce(context.Background(), original, mismatchRecord(0), ReduceOptions{Repeats: 3, MaxCandidates: 100}, hostReplay)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Reduced.Inputs) >= len(original.Inputs) || r.Reduced.Grid[0] >= original.Grid[0] || r.Reduced.Block[0] >= original.Block[0] || r.Reduced.Shape[0] >= original.Shape[0] {
		t.Fatalf("did not shrink: %+v", r.Reduced)
	}
	if r.NoReduction != "" || r.BudgetExhausted {
		t.Fatalf("unexpected stopping: %+v", r)
	}
	for _, a := range r.Attempts {
		if a.Accepted && a.Measurement.Reproductions != 3 {
			t.Fatal("accepted failure loss")
		}
	}
	got, _ := hostReplay(context.Background(), r.Reduced)
	if ok, err := (Identity{Policy: "exact-uint64"}).Matches(got); err != nil || !ok {
		t.Fatalf("lost identity: %v, %v", ok, err)
	}
	if sha256.Sum256([]byte(original.Source)) != sha256.Sum256([]byte(r.Reduced.Source)) {
		t.Fatal("source edited")
	}
	if !reflect.DeepEqual(original, reductionCase()) {
		t.Fatal("caller inputs mutated")
	}
}

func TestReduce_RejectsIdentityDrift(t *testing.T) {
	original := reductionCase()
	for _, verdict := range []string{"element", "crash"} {
		t.Run(verdict, func(t *testing.T) {
			r, err := Reduce(context.Background(), original, mismatchRecord(0), ReduceOptions{Repeats: 2, MaxCandidates: 100}, func(_ context.Context, c Case) (cudaschedule.RunRecord, error) {
				run := mismatchRecord(0)
				if !reflect.DeepEqual(c, original) {
					if verdict == "element" {
						run.Mismatch.Index = 1
					} else {
						run.ExitCode = 1
					}
				}
				return run, nil
			})
			if err != nil || r.IdentityDrift == 0 || r.NoReduction == "" || !reflect.DeepEqual(r.Reduced, original) {
				t.Fatalf("drift accepted: %+v, %v", r, err)
			}
		})
	}
}

func TestReduce_RejectsZeroReplayCount(t *testing.T) {
	if _, err := Reduce(context.Background(), reductionCase(), mismatchRecord(0), ReduceOptions{MaxCandidates: 4}, hostReplay); err == nil {
		t.Fatal("accepted zero replay count")
	}
}

func TestReduce_OneOffReportedAsOneOff(t *testing.T) {
	calls := 0
	r, err := Reduce(context.Background(), reductionCase(), mismatchRecord(0), ReduceOptions{Repeats: 4, MaxCandidates: 4}, func(context.Context, Case) (cudaschedule.RunRecord, error) {
		calls++
		run := mismatchRecord(0)
		if calls > 1 {
			run.Stdout = "PASS"
		}
		return run, nil
	})
	if err != nil || calls != 4 || !r.Baseline.OneOff() || len(r.Attempts) != 0 || r.NoReduction == "" {
		t.Fatalf("one-off reduced: %+v, calls=%d, err=%v", r, calls, err)
	}
}

func TestReduce_NoSourceEdits(t *testing.T) {
	original := reductionCase()
	r, err := Reduce(context.Background(), original, mismatchRecord(0), ReduceOptions{Repeats: 2, MaxCandidates: 5}, func(ctx context.Context, c Case) (cudaschedule.RunRecord, error) {
		if c.Source != original.Source {
			t.Fatal("source changed")
		}
		// The runner may reuse or modify its input buffer without changing the
		// reducer's original or another attempt's retained candidate.
		run, err := hostReplay(ctx, c)
		if len(c.Inputs) > 0 {
			c.Inputs[0] = 999
		}
		return run, err
	})
	if err != nil || r.Reduced.Source != original.Source || !r.BudgetExhausted || len(r.Attempts) != 5 {
		t.Fatalf("bounded reduction: %+v, %v", r, err)
	}
	for _, a := range r.Attempts {
		if len(a.Case.Inputs) > 0 && a.Case.Inputs[0] == 999 {
			t.Fatal("runner mutated retained case")
		}
	}
}

func TestReduceRejectsRateDrift(t *testing.T) {
	original := reductionCase()
	calls := 0
	r, err := Reduce(context.Background(), original, mismatchRecord(0), ReduceOptions{Repeats: 4, MaxCandidates: 100}, func(_ context.Context, c Case) (cudaschedule.RunRecord, error) {
		calls++
		run := mismatchRecord(0)
		if !reflect.DeepEqual(c, original) && calls%4 != 1 {
			run.Stdout = "PASS"
		}
		return run, nil
	})
	if err != nil || !reflect.DeepEqual(r.Reduced, original) || r.NoReduction == "" || r.IdentityDrift != 0 || r.CandidateRuns != 4*len(r.Attempts) {
		t.Fatalf("rate drift accepted: %+v, %v", r, err)
	}
}

func TestReduceRejectsOverflowingBudget(t *testing.T) {
	if _, err := Reduce(context.Background(), reductionCase(), mismatchRecord(0), ReduceOptions{Repeats: 2, MaxCandidates: int(^uint(0) >> 1)}, hostReplay); err == nil {
		t.Fatal("accepted overflowing budget")
	}
}

func FuzzReduceShapeAndBudget(f *testing.F) {
	f.Add(uint64(64), uint32(64))
	f.Add(^uint64(0), ^uint32(0))
	f.Fuzz(func(t *testing.T, shape uint64, block uint32) {
		if shape == 0 || block == 0 {
			return
		}
		c := reductionCase()
		c.Shape[0] = shape
		c.Block[0] = block
		r, err := Reduce(context.Background(), c, mismatchRecord(0), ReduceOptions{Repeats: 2, MaxCandidates: 8}, hostReplay)
		if shape < 2 || block < 2 {
			if err == nil {
				t.Fatal("healthy baseline accepted")
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(r.Attempts) > 8 || r.CandidateRuns > 16 || r.Reduced.Source != c.Source {
			t.Fatalf("invalid bounded reduction: %+v", r)
		}
		for _, a := range r.Attempts {
			if a.Case.Shape[0] == 0 || a.Case.Block[0] == 0 {
				t.Fatal("zero extent produced")
			}
		}
	})
}
