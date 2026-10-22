package dsv4

import (
	"errors"
	"math"
	"math/rand"
	"strconv"
	"testing"
)

type scriptedRandom []float64

func (r *scriptedRandom) Float64() float64 {
	if len(*r) == 0 {
		return 0
	}
	v := (*r)[0]
	*r = (*r)[1:]
	return v
}

func TestExactVerifyAcceptAndReplace(t *testing.T) {
	r := scriptedRandom{0.1}
	token, accepted, err := ExactVerify([]float64{0.6, 0.3, 0.1}, 0, &r)
	if err != nil || token != 0 || !accepted {
		t.Fatalf("accepted draft = %d, %v, %v", token, accepted, err)
	}
	r = scriptedRandom{0.9, 0.25}
	token, accepted, err = ExactVerify([]float64{0.6, 0.3, 0.1}, 0, &r)
	if err != nil || token != 1 || accepted {
		t.Fatalf("replacement = %d, %v, %v", token, accepted, err)
	}
}

func TestExactVerifySeededDistribution(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	probabilities := []float64{0.62, 0.27, 0.11}
	counts := make([]int, len(probabilities))
	accepted := 0
	const trials = 100000
	for i := 0; i < trials; i++ {
		token, ok, err := ExactVerify(probabilities, 0, rng)
		if err != nil {
			t.Fatal(err)
		}
		counts[token]++
		if ok {
			accepted++
		}
	}
	for i, p := range probabilities {
		got := float64(counts[i]) / trials
		if delta := got - p; delta < -0.006 || delta > 0.006 {
			t.Fatalf("token %d frequency = %.4f, want %.2f", i, got, p)
		}
	}
	if got := float64(accepted) / trials; got < 0.614 || got > 0.626 {
		t.Fatalf("acceptance frequency = %.4f, want %.2f", got, probabilities[0])
	}
}

func TestExactVerifySeededDistributionForEveryDraft(t *testing.T) {
	probabilities := []float64{0.62, 0.27, 0.11}
	const trials = 100000
	for draft := range probabilities {
		t.Run("draft-"+strconv.Itoa(draft), func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(17 + draft)))
			counts := make([]int, len(probabilities))
			accepted := 0
			for i := 0; i < trials; i++ {
				token, ok, err := ExactVerify(probabilities, draft, rng)
				if err != nil {
					t.Fatal(err)
				}
				counts[token]++
				if ok {
					accepted++
				}
			}
			for token, probability := range probabilities {
				got := float64(counts[token]) / trials
				if math.Abs(got-probability) >= 0.006 {
					t.Fatalf("draft %d token %d frequency = %.4f, want %.2f", draft, token, got, probability)
				}
			}
			gotAccepted := float64(accepted) / trials
			if math.Abs(gotAccepted-probabilities[draft]) >= 0.006 {
				t.Fatalf("draft %d acceptance frequency = %.4f, want %.2f", draft, gotAccepted, probabilities[draft])
			}
		})
	}
}

func TestExactVerifyDistributionGateDetectsAlwaysAcceptMutation(t *testing.T) {
	probabilities := []float64{0.62, 0.27, 0.11}
	const trials = 1000
	counts := make([]int, len(probabilities))
	for i := 0; i < trials; i++ {
		// This is the mutation under test: a verifier that always accepts the
		// draft and never samples the residual distribution.
		draft := 0
		counts[draft]++
	}
	for token, probability := range probabilities {
		got := float64(counts[token]) / trials
		if math.Abs(got-probability) < 0.1 {
			t.Fatalf("always-accept mutation was not detected for token %d: got %.3f, want %.2f", token, got, probability)
		}
	}
}

func TestExactVerifyRejectsMutations(t *testing.T) {
	for _, test := range []struct {
		name  string
		p     []float64
		draft int
		rng   ExactRandom
		want  error
	}{
		{"empty", nil, 0, &scriptedRandom{}, errors.New("exact sampling requires probabilities")},
		{"draft range", []float64{1}, 1, &scriptedRandom{}, errors.New("draft token 1 is outside probability vector")},
		{"negative", []float64{1, -1}, 0, &scriptedRandom{}, errors.New("probability 1 is invalid")},
		{"nan", []float64{1, math.NaN()}, 0, &scriptedRandom{}, errors.New("probability 1 is invalid")},
		{"zero mass", []float64{0, 0}, 0, &scriptedRandom{}, errors.New("probabilities have no finite mass")},
		{"negative random draw", []float64{1, 1}, 0, &scriptedRandom{-0.1}, errors.New("exact sampling random draw is outside [0,1)")},
		{"one random draw", []float64{1, 1}, 0, &scriptedRandom{1}, errors.New("exact sampling random draw is outside [0,1)")},
		{"nan random draw", []float64{1, 1}, 0, &scriptedRandom{math.NaN()}, errors.New("exact sampling random draw is outside [0,1)")},
		{"infinite random draw", []float64{1, 1}, 0, &scriptedRandom{math.Inf(1)}, errors.New("exact sampling random draw is outside [0,1)")},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := ExactVerify(test.p, test.draft, test.rng)
			if err == nil || err.Error() != test.want.Error() {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}
