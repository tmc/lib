package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

func TestLaunchManifestEvidenceIdentity(t *testing.T) {
	source := cudaevidence.Source{Identity: cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), Entry: "kernel", Target: "sm_70", CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64)}, Producer: "CUPTI", Locator: "activity/4"}
	r := launchManifestRecord{Module: source.Identity.ArtifactDigest, Function: "kernel", Correlation: 4, ActivityCorrelation: 4, Block: [3]uint32{128, 1, 1}}
	v, err := classifyLaunchBlockX(r, &source)
	if err != nil {
		t.Fatal(err)
	}
	if err := v.CheckClaim(cudaevidence.Measured, source.Identity); err != nil {
		t.Fatal(err)
	}
	if *v.Record().Number != 128 {
		t.Fatal("wrong block dimension")
	}
	for _, tc := range []struct {
		name   string
		change func(*launchManifestRecord)
	}{
		{"stale image", func(r *launchManifestRecord) { r.Module = strings.Repeat("4", 64) }},
		{"wrong entry", func(r *launchManifestRecord) { r.Function = "other" }},
		{"wrong correlation", func(r *launchManifestRecord) { r.ActivityCorrelation = 9 }},
		{"absent activity", func(r *launchManifestRecord) { r.ActivityCorrelation = 0 }},
		{"absent dimension", func(r *launchManifestRecord) { r.Block[0] = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := r
			tc.change(&bad)
			v, err := classifyLaunchBlockX(bad, &source)
			if err != nil {
				t.Fatal(err)
			}
			if v.Record().Class != cudaevidence.Unread || v.Record().Number != nil {
				t.Fatal("overclaimed launch")
			}
		})
	}
	r.BlockXEvidence, err = classifyLaunchBlockX(r, nil)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	var got launchManifestRecord
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.BlockXEvidence.Record().Number != nil || got.BlockXEvidence.Record().Reason == "" {
		t.Fatal("metadata-only field became zero")
	}
}

func TestLaunchManifestMetadataEvidence(t *testing.T) {
	records := manifestRecords(captureResult{APIRecords: []captureAPI{{Correlation: 4}}})
	if len(records) != 1 || records[0].BlockXEvidence.Record().Number != nil || records[0].BlockXEvidence.Record().Reason == "" {
		t.Fatalf("records=%+v", records)
	}
}
