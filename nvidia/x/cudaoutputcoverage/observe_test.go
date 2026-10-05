package cudaoutputcoverage

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func TestOracle_PendingWithoutDevice(t *testing.T) {
	_, report, err := Observe(context.Background(), recorded(3), nil)
	if !errors.Is(err, cudaerrors.ErrHardwarePending) || report.Arm != "" {
		t.Fatal(report, err)
	}
}

func TestObserveRetainsReadbacksAndRefusesBeforeLaunch(t *testing.T) {
	spec := recorded(4)
	shared := make([]uint32, spec.Contract.Elements)
	calls := 0
	completed, report, err := Observe(context.Background(), spec, func(_ context.Context, fill uint32) ([]uint32, error) {
		calls++
		for i := range shared {
			shared[i] = fill
		}
		copy(shared[:2], spec.Reference[:2])
		return shared, nil
	})
	if err != nil || calls != 2 || len(report.Unwritten) != 1 || report.Unwritten[0] != (Range{2, 4}) {
		t.Fatal(report, err, calls)
	}
	if completed.ReadbackA[2] != spec.FillA || completed.ReadbackB[2] != spec.FillB {
		t.Fatal("retained reused-buffer alias")
	}
	spec.FillB = spec.FillA
	_, _, err = Observe(context.Background(), spec, func(context.Context, uint32) ([]uint32, error) { t.Fatal("invalid patterns launched"); return nil, nil })
	if err == nil {
		t.Fatal("invalid fill admission")
	}
}

func TestObserveStopsOnFailedOrCancelledIteration(t *testing.T) {
	for _, kind := range []string{"failure", "extent", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			_, report, err := Observe(ctx, recorded(3), func(context.Context, uint32) ([]uint32, error) {
				calls++
				if kind == "failure" {
					return nil, errors.New("owned launch failed")
				}
				if kind == "extent" {
					return []uint32{1}, nil
				}
				cancel()
				return make([]uint32, 6), nil
			})
			if err == nil || calls != 1 || report.Arm != "" {
				t.Fatal("failed execution scored", calls, report, err)
			}
		})
	}
}

func TestObserveCancellationAfterSecondReadbackCannotScore(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	spec := recorded(3)
	completed, report, err := Observe(ctx, spec, func(context.Context, uint32) ([]uint32, error) {
		calls++
		if calls == 2 {
			cancel()
		}
		return spec.Reference, nil
	})
	if !errors.Is(err, context.Canceled) || calls != 2 || report.Arm != "" || len(completed.ReadbackB) != 6 {
		t.Fatal("cancelled iteration scored or completed bytes lost", completed, report, err, calls)
	}
}

func FuzzTwoFillRecorded(f *testing.F) {
	f.Add([]byte{0, 1, 0, 2, 3})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 || len(data) > 256 {
			return
		}
		r := recorded(len(data))
		want := make([]int, 0)
		for i, b := range data {
			if b%3 == 0 {
				r.ReadbackA[i] = r.FillA
				r.ReadbackB[i] = r.FillB
				want = append(want, i)
			}
		}
		out, err := ClassifyTwoFill(r)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]int, 0)
		for _, v := range out.Unwritten {
			for i := v.Start; i < v.End; i++ {
				got = append(got, i)
			}
		}
		if len(got) != len(want) {
			t.Fatal(out, want)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatal(out, want)
			}
		}
	})
}
