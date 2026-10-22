package dsv4

import "testing"

// This exhaustive rational-grid oracle uses independent conditional
// probabilities, not ExactVerify's output as its expected values. With
// denominator 100, each probability interval contains an integral number of
// midpoints, so there is no Monte Carlo tolerance.
func TestExactVerifyRationalPartitionOracle(t *testing.T) {
	p := []float64{0.62, 0.27, 0.11}
	for draft := range p {
		counts := [3]int{}
		for a := 0; a < 100; a++ {
			for b := 0; b < 100; b++ {
				u, v := (float64(a)+0.5)/100, (float64(b)+0.5)/100
				want, accepted := draft, u < p[draft]
				if !accepted {
					// Direct conditional CDF, independent of the production
					// implementation's subtractive residual walk.
					cdf := 0.0
					for token := range p {
						if token == draft {
							continue
						}
						cdf += p[token] / (1 - p[draft])
						if v < cdf {
							want = token
							break
						}
					}
				}
				rng := scriptedRandom{u, v}
				got, ok, err := ExactVerify(p, draft, &rng)
				if err != nil || got != want || ok != accepted {
					t.Fatalf("draft=%d draws=%g,%g: got %d/%v/%v, want %d/%v", draft, u, v, got, ok, err, want, accepted)
				}
				counts[got]++
			}
		}
		if counts[draft] != int(p[draft]*10000) {
			t.Fatalf("draft %d count %d", draft, counts[draft])
		}
	}
}
