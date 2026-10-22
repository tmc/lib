package dsv4

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

func TestSampledPointMassRNGProtocol(t *testing.T) {
	for _, tt := range []struct {
		name            string
		rows            [][]float32
		drafts, tokens  []int
		accepted, draws int
	}{
		{"probability one", [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {0, 9, -9}}, []int{1, 1}, []int{1, 1, 1}, 2, 1},
		{"zero first", [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}}, []int{0, 1}, []int{1, 1}, 0, 2},
		{"zero later", [][]float32{{0, 9, -9}, {0, 9, -9}, {0, 9, -9}, {9, 0, -9}, {0, 9, -9}}, []int{1, 1}, []int{1, 1, 0}, 1, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			const seed = 7
			s, _, _ := sampledFixture(t, ExactSampling, sampler.Params{Temperature: 1, TopK: 1, TopP: 1}, seed, tt.rows, tt.drafts)
			want := sampler.RNG{State: seed}
			for i := 0; i < tt.draws; i++ {
				want.Float32()
			}
			got, err := s.Next(12)
			if err != nil || !reflect.DeepEqual(got.Tokens, tt.tokens) || got.Accepted != tt.accepted || s.Random() != want {
				t.Fatalf("cycle=%+v err=%v RNG=%x want tokens=%v accepted=%d RNG=%x (%d draws)", got, err, s.Random().State, tt.tokens, tt.accepted, want.State, tt.draws)
			}
		})
	}
}
