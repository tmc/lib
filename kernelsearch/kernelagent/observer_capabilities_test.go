package kernelagent

import (
	"context"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/egraph"
	"testing"
)

type reviewPreparedExecutable struct{ cudasearch.Executable }

func (x reviewPreparedExecutable) Preparation() []cudasearch.PreparationPhase {
	return []cudasearch.PreparationPhase{{Name: "review_prepare", WallNs: 7, InputBytes: 8, OutputBytes: 8, Complete: true}}
}
func TestReviewAdapterPreservesPreparation(t *testing.T) {
	f, l := setup(t)
	compile := l.cfg.Compile
	l.cfg.Compile = func(ctx context.Context, x *egraph.Expr, b cudasearch.WorkloadBindings) (cudasearch.Executable, error) {
		e, err := compile(ctx, x, b)
		if e == nil {
			return nil, err
		}
		return reviewPreparedExecutable{e}, err
	}
	r, err := l.RunEpoch(t.Context(), "prepared", f.state, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if r.Search == nil || r.Search.Winner == nil {
		t.Fatal("no winner")
	}
	if len(r.Search.Winner.Preparation) != 1 || r.Search.Winner.Preparation[0].Name != "review_prepare" {
		t.Fatalf("actual adapter lost declared preparation: %+v", r.Search.Winner.Preparation)
	}
}
