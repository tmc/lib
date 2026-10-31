package qualify

import (
	"crypto/sha256"
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

func measuredFixture() (MeasuredIdentity, MeasuredPair) {
	id := MeasuredIdentity{Context: sha256.Sum256([]byte("SYNTHETIC_FULL_CONTEXT_NOT_NATIVE_AUTHORITY")), Admission: sha256.Sum256([]byte("SYNTHETIC_HOST_ADMISSION"))}
	b := pairedtiming.Block{A1: 10, B1: 20, B2: 20, A2: 10}
	return id, MeasuredPair{Identity: id, Reference: "reference", Candidate: "candidate", Blocks: []pairedtiming.Block{b, b, b}}
}
func TestSelectMeasuredSynthetic(t *testing.T) {
	id, p := measuredFixture()
	r, err := SelectMeasured(id, []MeasuredPair{p}, func(MeasuredPair) error { return nil })
	if err != nil || r.Status != "EXACT_MEASURED_COMPARISON" || r.Selected == nil || r.Selected.Name != "reference" || r.CandidateState != "MEASURED_LOSS" {
		t.Fatalf("%+v %v", r, err)
	}
	if r.Observed == nil || r.Observed.Ratio != .5 || r.Observed.Baseline != 10 || r.Observed.Candidate != 20 {
		t.Fatal("synthetic reference-zero fallback leaked into observations")
	}
	if r.Selected.Entry != "" || r.Selected.Source != "" {
		t.Fatal("whole variant handle became an emitter")
	}
}
func TestSelectMeasuredRejectsBeforeSelector(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*MeasuredIdentity, *[]MeasuredPair, *func(MeasuredPair) error)
	}{
		{"missing_context", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { q.Context = [32]byte{} }},
		{"missing_admission", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { q.Admission = [32]byte{} }},
		{"different_context", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { q.Context[0] ^= 1 }},
		{"different_admission", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { q.Admission[0] ^= 1 }},
		{"no_pair", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { *p = nil }},
		{"ambiguous", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { *p = append(*p, (*p)[0]) }},
		{"no_verifier", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { *v = nil }},
		{"unadmitted", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) {
			*v = func(MeasuredPair) error { return errors.New("raw PASS is not admission") }
		}},
		{"same_name", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) {
			(*p)[0].Candidate = (*p)[0].Reference
		}},
		{"one_block", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) {
			(*p)[0].Blocks = (*p)[0].Blocks[:1]
		}},
		{"zero", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) { (*p)[0].Blocks[0].A1 = 0 }},
		{"NaN", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) {
			(*p)[0].Blocks[0].A1 = math.NaN()
		}},
		{"overflow", func(q *MeasuredIdentity, p *[]MeasuredPair, v *func(MeasuredPair) error) {
			(*p)[0].Blocks[0] = pairedtiming.Block{A1: math.MaxFloat64, A2: math.MaxFloat64, B1: math.SmallestNonzeroFloat64, B2: math.SmallestNonzeroFloat64}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q, p := measuredFixture()
			pairs := []MeasuredPair{p}
			verify := func(MeasuredPair) error { return nil }
			tc.mutate(&q, &pairs, &verify)
			calls := 0
			r, err := selectMeasured(q, pairs, verify, func(v []Variant, b map[string][]pairedtiming.Block, h float64) (Variant, pairedtiming.Summary, error) {
				calls++
				return SelectByPairedTiming(v, b, h)
			})
			if err != nil || calls != 0 || r.Status != "NOT_EVALUATED" || r.Selected != nil || r.Observed != nil {
				t.Fatalf("calls%d result%+v err%v", calls, r, err)
			}
		})
	}
}
func TestSelectMeasuredCopiedAdmissionForgedBlocks(t *testing.T) {
	q, p := measuredFixture()
	actual := p.Blocks[0]
	verify := func(v MeasuredPair) error {
		for _, b := range v.Blocks {
			if b != actual {
				return errors.New("exact admitted block join differs")
			}
		}
		return nil
	}
	p.Blocks[0].B1 = 1
	calls := 0
	r, err := selectMeasured(q, []MeasuredPair{p}, verify, func(v []Variant, b map[string][]pairedtiming.Block, h float64) (Variant, pairedtiming.Summary, error) {
		calls++
		return SelectByPairedTiming(v, b, h)
	})
	if err != nil || calls != 0 || r.Selected != nil || r.Observed != nil || r.Status != "NOT_EVALUATED" {
		t.Fatalf("%+v %v calls%d", r, err, calls)
	}
}

func TestSelectMeasuredConservativePolicy(t *testing.T) {
	id := MeasuredIdentity{Context: [32]byte{1}, Admission: [32]byte{2}}
	cases := []struct {
		name            string
		ratios          []float64
		state, selected string
	}{
		{"threshold", []float64{1.01, 1.01, 1.01}, "MEASURED_NO_PROMOTION", "reference"},
		{"equality", []float64{1, 1, 1}, "MEASURED_NO_PROMOTION", "reference"},
		{"one-regression", []float64{0.99, 1.2, 1.3}, "MEASURED_NO_PROMOTION", "reference"},
		{"promoted", []float64{1.2, 1.3, 1.4}, "PROMOTED_LOCAL_EXACT_CONTEXT", "candidate"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			p := MeasuredPair{Identity: id, Reference: "reference", Candidate: "candidate"}
			for _, r := range tt.ratios {
				p.Blocks = append(p.Blocks, pairedtiming.Block{A1: 10 * r, A2: 10 * r, B1: 10, B2: 10})
			}
			got, err := SelectMeasured(id, []MeasuredPair{p}, func(MeasuredPair) error { return nil })
			if err != nil || got.Selected == nil || got.Selected.Name != tt.selected || got.CandidateState != tt.state {
				t.Fatalf("%+v %v", got, err)
			}
		})
	}
}
func TestSelectMeasuredVerifierCannotRewriteSamples(t *testing.T) {
	id := MeasuredIdentity{Context: [32]byte{1}, Admission: [32]byte{2}}
	p := MeasuredPair{Identity: id, Reference: "reference", Candidate: "candidate", Blocks: []pairedtiming.Block{{A1: 10, A2: 10, B1: 20, B2: 20}, {A1: 10, A2: 10, B1: 20, B2: 20}, {A1: 10, A2: 10, B1: 20, B2: 20}}}
	got, err := SelectMeasured(id, []MeasuredPair{p}, func(v MeasuredPair) error {
		for i := range v.Blocks {
			v.Blocks[i].B1 = 1
			v.Blocks[i].B2 = 1
		}
		return nil
	})
	if err != nil || got.Observed == nil || got.Observed.Ratio != .5 || got.CandidateState != "MEASURED_LOSS" {
		t.Fatalf("callback rewrote measured data: %+v %v", got, err)
	}
}
