//go:build darwin && arm64

package main

import (
	"math"
	"testing"
)

func TestMetalDSparkProposalResult(t *testing.T) {
	for _, test := range []struct {
		name          string
		ids           []int32
		probabilities []float32
		result        []uint32
		want          int
		fail          bool
	}{
		{"all", []int32{1, 2}, []float32{.8, .9}, []uint32{2, 0}, 2, false},
		{"prefix", []int32{1, -1}, []float32{.8, .2}, []uint32{1, 0}, 1, false},
		{"empty", []int32{-1, -1}, []float32{.2, 0}, []uint32{0, 0}, 0, false},
		{"missing-status", []int32{1, 2}, []float32{.8, .9}, nil, 0, true},
		{"device-finite-flag", []int32{1, 2}, []float32{.8, .9}, []uint32{2, 1}, 0, true},
		{"overflow", []int32{1, 2}, []float32{.8, .9}, []uint32{3, 0}, 0, true},
		{"accepted-nan", []int32{1, 2}, []float32{float32(math.NaN()), .9}, []uint32{2, 0}, 0, true},
		{"declined-nan", []int32{1, -1}, []float32{.8, float32(math.NaN())}, []uint32{1, 0}, 0, true},
		{"invalid-id", []int32{1, int32(Flash.Vocab)}, []float32{.8, .9}, []uint32{2, 0}, 0, true},
		{"false-decline", []int32{1, -1}, []float32{.8, .9}, []uint32{1, 0}, 0, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := metalDSparkProposalResult(test.ids, test.probabilities, test.result, 2, .5)
			if (err != nil) != test.fail || err == nil && len(got) != test.want {
				t.Fatalf("result=%v err=%v", got, err)
			}
		})
	}
}
