package main

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// The native output buffer is split in half, each half holding an 80 byte
// header and one record per PC, so the admitted PC capacity falls as the
// selected reason count rises. These cases pin that relationship rather than
// restating the constants: a budget that silently sized itself to zero would
// start a session that can never return a record.
func TestPCSamplingBudget(t *testing.T) {
	tests := []struct {
		name    string
		reasons int
		period  uint32
		wantPCs int
		wantErr bool
	}{
		{name: "one reason", reasons: 1, period: 10, wantPCs: 2048},
		{name: "capacity clamped by name budget", reasons: 8, period: 10, wantPCs: 2048},
		{name: "wide inventory shrinks capacity", reasons: 512, period: 10, wantPCs: 126},
		{name: "no reasons", reasons: 0, period: 10, wantErr: true},
		{name: "period below the native floor", reasons: 8, period: 4, wantErr: true},
		{name: "period above the native ceiling", reasons: 8, period: 32, wantErr: true},
		{name: "inventory exceeds the output budget", reasons: 1 << 17, period: 10, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := pcSamplingBudget(tt.reasons, tt.period)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("pcSamplingBudget(%d, %d) = %+v, want error", tt.reasons, tt.period, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("pcSamplingBudget(%d, %d): %v", tt.reasons, tt.period, err)
			}
			if got.PCsPerBuffer != tt.wantPCs {
				t.Errorf("PCsPerBuffer = %d, want %d", got.PCsPerBuffer, tt.wantPCs)
			}
			// Every budget must satisfy the package's own preconditions, or
			// Start rejects a configuration this function called valid.
			if got.MaxNames < 2*got.PCsPerBuffer {
				t.Errorf("MaxNames = %d, want at least 2*PCsPerBuffer = %d", got.MaxNames, 2*got.PCsPerBuffer)
			}
			if got.ScratchBytes+got.HardwareBytes > got.MaxNativeBytes {
				t.Errorf("scratch+hardware = %d, exceeds MaxNativeBytes = %d", got.ScratchBytes+got.HardwareBytes, got.MaxNativeBytes)
			}
			if got.MaxReasons != tt.reasons {
				t.Errorf("MaxReasons = %d, want %d", got.MaxReasons, tt.reasons)
			}
		})
	}
}

func TestPCSamplingResolution(t *testing.T) {
	tests := []struct {
		name string
		res  *pcSamplingResult
		want float64
	}{
		{name: "nil", res: nil, want: 0},
		{name: "no pcs", res: &pcSamplingResult{}, want: 0},
		{name: "none resolved", res: &pcSamplingResult{PCs: 8}, want: 0},
		{name: "half resolved", res: &pcSamplingResult{PCs: 8, PCsResolvedToPTX: 4}, want: 0.5},
		{name: "all resolved", res: &pcSamplingResult{PCs: 8, PCsResolvedToPTX: 8}, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.res.Resolution(); got != tt.want {
				t.Errorf("Resolution() = %v, want %v", got, tt.want)
			}
		})
	}
}

// A batch that reaches the profiler and fails to record leaves its samples in
// total_samples and in no sample type, so the reconciliation identity falls
// short by exactly that much. record keeps going -- a reporting failure must not
// discard a workload that ran -- and counts the loss so a reader does not have
// to infer it from arithmetic that may not be performed.
func TestRecordCountsFailedBatches(t *testing.T) {
	// A bridge with no session rejects every batch, which is the failure mode
	// under test: collected, counted, not recorded.
	bridge := &cuptipcsampling.ProfileBridge{}
	batch := cuptipcsampling.Batch{Valid: true, TotalSamples: 7, DroppedSamples: 2}
	res := &pcSamplingResult{}
	record(context.Background(), res, bridge, cuptipcsampling.CodeResolver(nil), batch)
	if res.RecordErrors != 1 {
		t.Errorf("RecordErrors = %d, want 1", res.RecordErrors)
	}
	// The collection counters still hold what the batch contained, which is what
	// makes the shortfall computable rather than merely suspected.
	if res.TotalSamples != 7 {
		t.Errorf("TotalSamples = %d, want 7", res.TotalSamples)
	}
	if res.DroppedSamples != 2 {
		t.Errorf("DroppedSamples = %d, want 2", res.DroppedSamples)
	}
	// An invalid batch never reaches RecordBatch, so it is not a record failure.
	res = &pcSamplingResult{}
	record(context.Background(), res, bridge, cuptipcsampling.CodeResolver(nil), cuptipcsampling.Batch{DroppedSamples: 3})
	if res.RecordErrors != 0 {
		t.Errorf("RecordErrors after an invalid batch = %d, want 0", res.RecordErrors)
	}
	if res.DroppedSamples != 3 {
		t.Errorf("DroppedSamples = %d, want 3", res.DroppedSamples)
	}
}
