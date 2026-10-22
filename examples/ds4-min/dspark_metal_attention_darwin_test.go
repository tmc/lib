//go:build darwin && arm64

package main

import (
	"math"
	"strings"
	"testing"
)

// This reference deliberately does not call the production attention helpers.
// Values are half-exact so the explicit query/storage conversions add no
// uncertainty to the primitive oracle.
func metalDSparkAttentionFixture() (query, raw, sinks, want []float32) {
	query = make([]float32, 64*512)
	raw = make([]float32, 3*512)
	sinks = make([]float32, 64)
	want = make([]float32, len(query))
	for i := range query {
		query[i] = float32(i%7-3) / 16
	}
	for i := range raw {
		raw[i] = float32(i%11-5) / 8
	}
	for head := 0; head < 64; head++ {
		sinks[head] = float32(head%5-2) / 4
		scores := make([]float64, 3)
		maximum := float64(sinks[head])
		for row := range scores {
			for dim := 0; dim < 512; dim++ {
				scores[row] += float64(query[head*512+dim]) * float64(raw[row*512+dim])
			}
			scores[row] /= math.Sqrt(512)
			maximum = math.Max(maximum, scores[row])
		}
		denominator := math.Exp(float64(sinks[head]) - maximum)
		for row := range scores {
			scores[row] = math.Exp(scores[row] - maximum)
			denominator += scores[row]
		}
		for dim := 0; dim < 512; dim++ {
			var value float64
			for row := range scores {
				value += scores[row] * float64(raw[row*512+dim])
			}
			want[head*512+dim] = float32(value / denominator)
		}
	}
	return
}

func TestMetalDSparkAttentionFixture(t *testing.T) {
	q, raw, sinks, want := metalDSparkAttentionFixture()
	if len(q) != 32768 || len(raw) != 1536 || len(sinks) != 64 || len(want) != len(q) || !allFinite(q, raw, sinks, want) {
		t.Fatal("invalid finite fixture extent")
	}
	// Omitting the sink is a semantic mutation, not another reduction order.
	var score, weighted float64
	for row := 0; row < 3; row++ {
		var dot float64
		for dim := 0; dim < 512; dim++ {
			dot += float64(q[dim]) * float64(raw[row*512+dim])
		}
		p := math.Exp(dot / math.Sqrt(512))
		score += p
		weighted += p * float64(raw[row*512])
	}
	if math.Abs(weighted/score-float64(want[0])) < 1e-5 {
		t.Fatal("sink-omission mutation is not discriminated")
	}
	const site = "output[hrow*512+d]=value/denominator;"
	if strings.Count(metalDSparkAttentionMSL, site) != 1 {
		t.Fatal("same-site output mutation must identify exactly one consumer")
	}
}
