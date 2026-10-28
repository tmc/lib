package cudagraph_test

import (
	"errors"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func TestBuildBatchBuckets(t *testing.T) {
	tests := []struct {
		name    string
		policy  cudagraph.BucketPolicy
		want    []int
		wantErr bool
	}{
		{
			name:   "max_1",
			policy: cudagraph.BucketPolicy{MaxBsz: 1},
			want:   []int{1},
		},
		{
			name:   "max_3_non_aligned",
			policy: cudagraph.BucketPolicy{MaxBsz: 3},
			want:   []int{1, 2, 3},
		},
		{
			name:   "max_16_defaults",
			policy: cudagraph.BucketPolicy{MaxBsz: 16},
			want:   []int{1, 2, 4, 8, 12, 16},
		},
		{
			name:   "max_64_defaults",
			policy: cudagraph.BucketPolicy{MaxBsz: 64},
			want:   []int{1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64},
		},
		{
			name:   "max_100_piecewise_includes_max",
			policy: cudagraph.BucketPolicy{MaxBsz: 100},
			want:   []int{1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96, 100},
		},
		{
			name: "explicit_dedup_and_max",
			policy: cudagraph.BucketPolicy{
				MaxBsz:          48,
				ExplicitBuckets: []int{32, 8, 1, 8, 16, 64},
			},
			want: []int{1, 8, 16, 32, 48},
		},
		{
			name:    "zero_max_without_explicit",
			policy:  cudagraph.BucketPolicy{MaxBsz: 0},
			wantErr: true,
		},
		{
			name:    "negative_max",
			policy:  cudagraph.BucketPolicy{MaxBsz: -4},
			wantErr: true,
		},
		{
			name:    "negative_step",
			policy:  cudagraph.BucketPolicy{MaxBsz: 32, SmallStep: -4},
			wantErr: true,
		},
		{
			name:    "invalid_explicit_bucket",
			policy:  cudagraph.BucketPolicy{ExplicitBuckets: []int{1, 0, 4}},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := cudagraph.BuildBatchBuckets(tc.policy)
			if tc.wantErr {
				if !errors.Is(err, cudaerrors.ErrUnsupported) {
					t.Fatalf("BuildBatchBuckets(%+v) error = %v, want wrapping ErrUnsupported", tc.policy, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildBatchBuckets(%+v) unexpected error: %v", tc.policy, err)
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("BuildBatchBuckets(%+v) = %v, want %v", tc.policy, got, tc.want)
			}
		})
	}
}

func TestResolveBucketAndRunner(t *testing.T) {
	runner, err := cudagraph.NewBucketRunnerFromPolicy[string](cudagraph.BucketPolicy{MaxBsz: 40})
	if err != nil {
		t.Fatalf("NewBucketRunnerFromPolicy: %v", err)
	}

	wantBuckets := []int{1, 2, 4, 8, 12, 16, 24, 32, 40}
	if got := runner.Buckets(); !slices.Equal(got, wantBuckets) {
		t.Fatalf("runner.Buckets() = %v, want %v", got, wantBuckets)
	}

	wantWarmup := []int{40, 32, 24, 16, 12, 8, 4, 2, 1}
	if got := runner.WarmupOrder(); !slices.Equal(got, wantWarmup) {
		t.Fatalf("runner.WarmupOrder() = %v, want %v", got, wantWarmup)
	}

	// Registering an unknown bucket must fail with ErrUnsupported.
	if err := runner.Register(7, "graph-7"); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("Register(7) error = %v, want wrapping ErrUnsupported", err)
	}

	// Lookup before registration must fail with ErrUnsupported.
	if _, _, err := runner.Lookup(3); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("Lookup(3) before Register error = %v, want wrapping ErrUnsupported", err)
	}

	for _, b := range runner.WarmupOrder() {
		if err := runner.Register(b, "graph"); err != nil {
			t.Fatalf("Register(%d): %v", b, err)
		}
	}
	if err := runner.Register(12, "graph-12"); err != nil {
		t.Fatalf("Register(12): %v", err)
	}

	h, pad, err := runner.Lookup(9)
	if err != nil {
		t.Fatalf("Lookup(9): %v", err)
	}
	if h != "graph-12" || pad.ActualBsz != 9 || pad.BucketBsz != 12 || pad.PadCount != 3 {
		t.Fatalf("Lookup(9) = (%q, %+v), want (\"graph-12\", {ActualBsz:9 BucketBsz:12 PadCount:3})", h, pad)
	}

	// Out-of-range lookups must fail wrapping ErrUnsupported.
	for _, invalid := range []int{0, -3, 41, 128} {
		if _, _, err := runner.Lookup(invalid); !errors.Is(err, cudaerrors.ErrUnsupported) {
			t.Fatalf("Lookup(%d) error = %v, want wrapping ErrUnsupported", invalid, err)
		}
		if _, err := cudagraph.ResolveBucket(wantBuckets, invalid); !errors.Is(err, cudaerrors.ErrUnsupported) {
			t.Fatalf("ResolveBucket(%d) error = %v, want wrapping ErrUnsupported", invalid, err)
		}
	}
}
