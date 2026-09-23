package main

import (
	"fmt"
	"math"
)

// Tolerance bounds an elementwise comparison. An element passes when
// |got-want| <= Abs + Rel*|want|. Both bounds are stated in every report, so a
// reader can see what "correct" meant for a given run.
type Tolerance struct {
	Abs float64 `json:"abs"`
	Rel float64 `json:"rel"`
}

// Mismatch is one element that failed the comparison.
type Mismatch struct {
	Index  int     `json:"index"`
	Got    float64 `json:"got"`
	Want   float64 `json:"want"`
	AbsErr float64 `json:"abs_err"`
	RelErr float64 `json:"rel_err"`
}

// Comparison is the result of checking one candidate output against the
// reference. NonFinite is counted separately from Mismatches because a NaN and
// a drifted value call for different fixes: a NaN is usually a division or an
// uninitialized read, a drift is usually an accumulation order.
type Comparison struct {
	Elements   int       `json:"elements"`
	Mismatches int       `json:"mismatches"`
	NonFinite  int       `json:"non_finite"`
	Worst      *Mismatch `json:"worst,omitempty"`
}

// OK reports whether every element was within tolerance and finite.
func (c Comparison) OK() bool { return c.Mismatches == 0 && c.NonFinite == 0 }

// Compare checks got against want under tol. A NaN or infinity in got always
// counts as a failure, whatever the tolerance: no tolerance admits a NaN.
//
// The worst element is the one with the largest ratio of error to its own
// allowance, so it is the element that would need the largest tolerance
// increase to pass -- not merely the largest absolute error, which a single
// large-magnitude element would always win.
func Compare(got, want []float32, tol Tolerance) (Comparison, error) {
	if len(got) != len(want) {
		return Comparison{}, fmt.Errorf("output length %d, reference length %d", len(got), len(want))
	}
	if len(want) == 0 {
		// An empty comparison passes every tolerance and proves nothing.
		return Comparison{}, fmt.Errorf("reference is empty: a comparison over zero elements cannot fail")
	}
	c := Comparison{Elements: len(want)}
	worstRatio := math.Inf(-1)
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			c.NonFinite++
		}
		abs := math.Abs(g - w)
		allow := tol.Abs + tol.Rel*math.Abs(w)
		bad := math.IsNaN(g) || math.IsInf(g, 0) || abs > allow
		if !bad {
			continue
		}
		c.Mismatches++
		ratio := math.Inf(1)
		if allow > 0 && !math.IsNaN(abs) && !math.IsInf(abs, 0) {
			ratio = abs / allow
		}
		if ratio > worstRatio || c.Worst == nil {
			worstRatio = ratio
			rel := math.Inf(1)
			if w != 0 {
				rel = abs / math.Abs(w)
			}
			c.Worst = &Mismatch{Index: i, Got: g, Want: w, AbsErr: abs, RelErr: rel}
		}
	}
	return c, nil
}
