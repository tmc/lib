package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestFaultInfoAndNegativeConstraintVersionedRoundTrip(t *testing.T) {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   time.Second,
		Capacity: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Close()

	digest := strings.Repeat("ab", 32)
	fault := cudaflightrecorder.FaultInfo{
		TimestampNs:      123456789,
		ErrorCode:        700,
		ErrorName:        "CUDA_ERROR_ILLEGAL_ADDRESS",
		ErrorMessage:     "illegal memory access",
		OffendingKernel:  "fused_attention_fwd",
		CorrelationID:    99,
		ModuleID:         "mod-01",
		ModuleDigest:     digest,
		ModuleGeneration: 4,
		FailedRuleNames:  []string{"vectorize_ld_128", "fuse_epilogue"},
		Labels:           map[string]string{"stream": "0"},
	}

	var buf bytes.Buffer
	if _, err := rec.FaultSnapshot(context.Background(), &buf, fault); err != nil {
		t.Fatalf("FaultSnapshot failed: %v", err)
	}

	bundle, err := cudaflightrecorder.ReadBundle(&buf)
	if err != nil {
		t.Fatalf("ReadBundle failed: %v", err)
	}

	// Verify raw JSON payloads explicitly carry schema_version = 1.
	for _, name := range []string{"fault.json", "negative_rules.json"} {
		var raw map[string]any
		if err := json.Unmarshal(bundle.Files[name], &raw); err != nil {
			t.Fatalf("unmarshal raw %s: %v", name, err)
		}
		v, ok := raw["schema_version"].(float64)
		if !ok || int(v) != cudaflightrecorder.FaultSchemaVersion {
			t.Fatalf("%s schema_version = %v (%T), want %d", name, raw["schema_version"], raw["schema_version"], cudaflightrecorder.FaultSchemaVersion)
		}
	}

	wantFault := fault
	wantFault.SchemaVersion = cudaflightrecorder.FaultInfoSchemaVersion

	t.Run("zero_destination_unmarshal", func(t *testing.T) {
		var zeroFault cudaflightrecorder.FaultInfo
		if err := json.Unmarshal(bundle.Files["fault.json"], &zeroFault); err != nil {
			t.Fatalf("json.Unmarshal into zero FaultInfo: %v", err)
		}
		if zeroFault.SchemaVersion != cudaflightrecorder.FaultInfoSchemaVersion {
			t.Errorf("zeroFault.SchemaVersion = %d, want %d", zeroFault.SchemaVersion, cudaflightrecorder.FaultInfoSchemaVersion)
		}
		if err := zeroFault.Validate(); err != nil {
			t.Errorf("zeroFault.Validate() failed: %v", err)
		}
		if !reflect.DeepEqual(zeroFault, wantFault) {
			t.Errorf("zeroFault = %+v, want %+v", zeroFault, wantFault)
		}
	})

	t.Run("preseed_99_destination_unmarshal", func(t *testing.T) {
		preseedFault := cudaflightrecorder.FaultInfo{SchemaVersion: 99}
		if err := json.Unmarshal(bundle.Files["fault.json"], &preseedFault); err != nil {
			t.Fatalf("json.Unmarshal into preseed FaultInfo: %v", err)
		}
		if preseedFault.SchemaVersion != cudaflightrecorder.FaultInfoSchemaVersion {
			t.Errorf("preseedFault.SchemaVersion = %d, want %d", preseedFault.SchemaVersion, cudaflightrecorder.FaultInfoSchemaVersion)
		}
		if err := preseedFault.Validate(); err != nil {
			t.Errorf("preseedFault.Validate() failed: %v", err)
		}
		if !reflect.DeepEqual(preseedFault, wantFault) {
			t.Errorf("preseedFault = %+v, want %+v", preseedFault, wantFault)
		}

		preseedConstraint := cudaflightrecorder.NegativeRewriteConstraint{SchemaVersion: 99}
		if err := json.Unmarshal(bundle.Files["negative_rules.json"], &preseedConstraint); err != nil {
			t.Fatalf("json.Unmarshal into preseed NegativeRewriteConstraint: %v", err)
		}
		if preseedConstraint.SchemaVersion != cudaflightrecorder.NegativeRewriteConstraintSchemaVersion {
			t.Errorf("preseedConstraint.SchemaVersion = %d, want %d", preseedConstraint.SchemaVersion, cudaflightrecorder.NegativeRewriteConstraintSchemaVersion)
		}
		if err := preseedConstraint.Validate(); err != nil {
			t.Errorf("preseedConstraint.Validate() failed: %v", err)
		}
	})

	gotFault, gotConstraint, err := bundle.DecodeFaultEvidence()
	if err != nil {
		t.Fatalf("DecodeFaultEvidence failed: %v", err)
	}
	if !reflect.DeepEqual(gotFault, wantFault) {
		t.Fatalf("DecodeFaultInfo = %+v, want %+v", gotFault, wantFault)
	}

	if gotConstraint.SchemaVersion != cudaflightrecorder.NegativeRewriteConstraintSchemaVersion {
		t.Fatalf("constraint.SchemaVersion = %d, want %d", gotConstraint.SchemaVersion, cudaflightrecorder.NegativeRewriteConstraintSchemaVersion)
	}
	if gotConstraint.CandidateDigest != digest {
		t.Fatalf("constraint.CandidateDigest = %q, want %q", gotConstraint.CandidateDigest, digest)
	}
	if gotConstraint.OffendingKernel != fault.OffendingKernel {
		t.Fatalf("constraint.OffendingKernel = %q, want %q", gotConstraint.OffendingKernel, fault.OffendingKernel)
	}
	if gotConstraint.PruneSubtree || len(gotConstraint.QuarantinedRules) != 0 {
		t.Fatalf("unexpected broad ban in constraint: %+v", gotConstraint)
	}
	if !strings.Contains(gotConstraint.Reason, "candidate-only quarantine") {
		t.Fatalf("unexpected constraint reason: %q", gotConstraint.Reason)
	}

	// Explicit versioned FaultInfo round-trip preserves SchemaVersion when set.
	marshaledFault, err := json.Marshal(wantFault)
	if err != nil {
		t.Fatalf("marshal versioned FaultInfo: %v", err)
	}
	decodedFault, err := cudaflightrecorder.DecodeFaultInfo(marshaledFault)
	if err != nil {
		t.Fatalf("DecodeFaultInfo on marshaled bytes: %v", err)
	}
	if !reflect.DeepEqual(decodedFault, wantFault) {
		t.Fatalf("round-trip FaultInfo = %+v, want %+v", decodedFault, wantFault)
	}
}

func TestFaultSchemaVersionRefusal(t *testing.T) {
	digest := strings.Repeat("cd", 32)

	for _, tc := range []struct {
		name string
		json string
	}{
		{
			name: "missing_version",
			json: `{"error_code":700,"error_name":"CUDA_ERROR_ILLEGAL_ADDRESS","error_message":"fault","module_digest":"` + digest + `"}`,
		},
		{
			name: "zero_version",
			json: `{"schema_version":0,"error_code":700,"error_name":"CUDA_ERROR_ILLEGAL_ADDRESS","error_message":"fault","module_digest":"` + digest + `"}`,
		},
		{
			name: "negative_version",
			json: `{"schema_version":-1,"error_code":700,"error_name":"CUDA_ERROR_ILLEGAL_ADDRESS","error_message":"fault","module_digest":"` + digest + `"}`,
		},
		{
			name: "unknown_future_version",
			json: `{"schema_version":2,"error_code":700,"error_name":"CUDA_ERROR_ILLEGAL_ADDRESS","error_message":"fault","module_digest":"` + digest + `"}`,
		},
		{
			name: "string_version",
			json: `{"schema_version":"1","error_code":700,"error_name":"CUDA_ERROR_ILLEGAL_ADDRESS","error_message":"fault","module_digest":"` + digest + `"}`,
		},
		{
			name: "unknown_field",
			json: `{"schema_version":1,"error_code":700,"error_name":"CUDA_ERROR_ILLEGAL_ADDRESS","error_message":"fault","unexpected":true}`,
		},
		{
			name: "trailing_tokens",
			json: `{"schema_version":1,"error_code":700,"error_name":"CUDA_ERROR_ILLEGAL_ADDRESS","error_message":"fault"}{"extra":1}`,
		},
		{
			name: "null_payload",
			json: `null`,
		},
	} {
		t.Run("FaultInfo_"+tc.name, func(t *testing.T) {
			var got cudaflightrecorder.FaultInfo
			if err := got.UnmarshalJSON([]byte(tc.json)); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
				t.Fatalf("UnmarshalJSON FaultInfo(%s) err = %v, want ErrArtifactInvalid", tc.name, err)
			}
			if _, err := cudaflightrecorder.DecodeFaultInfo([]byte(tc.json)); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
				t.Fatalf("DecodeFaultInfo(%s) err = %v, want ErrArtifactInvalid", tc.name, err)
			}
		})
	}

	for _, tc := range []struct {
		name string
		json string
	}{
		{
			name: "missing_version",
			json: `{"candidate_digest":"` + digest + `","offending_kernel":"k","reason":"candidate-only"}`,
		},
		{
			name: "zero_version",
			json: `{"schema_version":0,"candidate_digest":"` + digest + `","offending_kernel":"k","reason":"candidate-only"}`,
		},
		{
			name: "negative_version",
			json: `{"schema_version":-1,"candidate_digest":"` + digest + `","offending_kernel":"k","reason":"candidate-only"}`,
		},
		{
			name: "unknown_future_version",
			json: `{"schema_version":99,"candidate_digest":"` + digest + `","offending_kernel":"k","reason":"candidate-only"}`,
		},
		{
			name: "string_version",
			json: `{"schema_version":"v1","candidate_digest":"` + digest + `","offending_kernel":"k","reason":"candidate-only"}`,
		},
		{
			name: "unknown_field",
			json: `{"schema_version":1,"candidate_digest":"` + digest + `","offending_kernel":"k","reason":"candidate-only","unknown":1}`,
		},
		{
			name: "trailing_tokens",
			json: `{"schema_version":1,"candidate_digest":"` + digest + `","offending_kernel":"k","reason":"candidate-only"}[]`,
		},
		{
			name: "null_payload",
			json: `null`,
		},
	} {
		t.Run("NegativeRewriteConstraint_"+tc.name, func(t *testing.T) {
			var got cudaflightrecorder.NegativeRewriteConstraint
			if err := got.UnmarshalJSON([]byte(tc.json)); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
				t.Fatalf("UnmarshalJSON NegativeRewriteConstraint(%s) err = %v, want ErrArtifactInvalid", tc.name, err)
			}
			if _, err := cudaflightrecorder.DecodeNegativeRewriteConstraint([]byte(tc.json)); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
				t.Fatalf("DecodeNegativeRewriteConstraint(%s) err = %v, want ErrArtifactInvalid", tc.name, err)
			}
		})
	}

	// Marshal must also refuse unknown non-zero schema versions.
	for _, badVer := range []int{-1, 2, 99} {
		if _, err := json.Marshal(cudaflightrecorder.FaultInfo{
			SchemaVersion: badVer,
			ErrorName:     "FAULT",
		}); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
			t.Fatalf("Marshal FaultInfo(SchemaVersion=%d) err = %v, want ErrArtifactInvalid", badVer, err)
		}
		if _, err := json.Marshal(cudaflightrecorder.NegativeRewriteConstraint{
			SchemaVersion:   badVer,
			CandidateDigest: digest,
			Reason:          "candidate-only",
		}); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
			t.Fatalf("Marshal NegativeRewriteConstraint(SchemaVersion=%d) err = %v, want ErrArtifactInvalid", badVer, err)
		}
	}
}

func TestExactCandidateQuarantinePreservationAndAdmission(t *testing.T) {
	validDigest := strings.Repeat("01", 32)
	otherDigest := strings.Repeat("fe", 32)

	// 1. Valid candidate-only constraint is admitted and preserved.
	validConstraint := cudaflightrecorder.NegativeRewriteConstraint{
		CandidateDigest: validDigest,
		OffendingKernel: "gemm_kernel",
		Reason:          "candidate-only quarantine; no reproduced rule policy",
		Timestamp:       time.Unix(1700000000, 0).UTC(),
	}
	rawValid, err := json.Marshal(validConstraint)
	if err != nil {
		t.Fatalf("Marshal validConstraint: %v", err)
	}
	admitted, err := cudaflightrecorder.DecodeNegativeRewriteConstraint(rawValid)
	if err != nil {
		t.Fatalf("DecodeNegativeRewriteConstraint valid: %v", err)
	}
	if admitted.SchemaVersion != cudaflightrecorder.NegativeRewriteConstraintSchemaVersion || admitted.CandidateDigest != validDigest {
		t.Fatalf("admitted = %+v, want version %d and digest %q", admitted, cudaflightrecorder.NegativeRewriteConstraintSchemaVersion, validDigest)
	}

	// 2. Withheld or malformed candidate identities and broad bans are refused on quarantine admission.
	for _, tc := range []struct {
		name       string
		constraint cudaflightrecorder.NegativeRewriteConstraint
		marshalErr bool
	}{
		{
			name: "empty_candidate_digest_withheld",
			constraint: cudaflightrecorder.NegativeRewriteConstraint{
				SchemaVersion: cudaflightrecorder.NegativeRewriteConstraintSchemaVersion,
				Reason:        "quarantine withheld: exact SHA256 candidate identity unavailable",
			},
			marshalErr: false,
		},
		{
			name: "short_candidate_digest",
			constraint: cudaflightrecorder.NegativeRewriteConstraint{
				SchemaVersion:   cudaflightrecorder.NegativeRewriteConstraintSchemaVersion,
				CandidateDigest: strings.Repeat("a", 32),
				Reason:          "candidate-only",
			},
			marshalErr: true,
		},
		{
			name: "non_hex_candidate_digest",
			constraint: cudaflightrecorder.NegativeRewriteConstraint{
				SchemaVersion:   cudaflightrecorder.NegativeRewriteConstraintSchemaVersion,
				CandidateDigest: strings.Repeat("z", 64),
				Reason:          "candidate-only",
			},
			marshalErr: true,
		},
		{
			name: "empty_reason",
			constraint: cudaflightrecorder.NegativeRewriteConstraint{
				SchemaVersion:   cudaflightrecorder.NegativeRewriteConstraintSchemaVersion,
				CandidateDigest: validDigest,
				Reason:          "   ",
			},
			marshalErr: false,
		},
		{
			name: "unreproduced_prune_subtree",
			constraint: cudaflightrecorder.NegativeRewriteConstraint{
				SchemaVersion:   cudaflightrecorder.NegativeRewriteConstraintSchemaVersion,
				CandidateDigest: validDigest,
				Reason:          "candidate-only",
				PruneSubtree:    true,
			},
			marshalErr: false,
		},
		{
			name: "unreproduced_quarantined_rules",
			constraint: cudaflightrecorder.NegativeRewriteConstraint{
				SchemaVersion:    cudaflightrecorder.NegativeRewriteConstraintSchemaVersion,
				CandidateDigest:  validDigest,
				Reason:           "candidate-only",
				QuarantinedRules: []string{"rule_a"},
			},
			marshalErr: false,
		},
		{
			name: "empty_digest_with_prune_subtree",
			constraint: cudaflightrecorder.NegativeRewriteConstraint{
				SchemaVersion: cudaflightrecorder.NegativeRewriteConstraintSchemaVersion,
				Reason:        "invalid broad ban",
				PruneSubtree:  true,
			},
			marshalErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.constraint.Validate(); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
				t.Fatalf("Validate(%s) err = %v, want ErrArtifactInvalid", tc.name, err)
			}
			data, err := json.Marshal(tc.constraint)
			if tc.marshalErr {
				if !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
					t.Fatalf("Marshal(%s) err = %v, want ErrArtifactInvalid", tc.name, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Marshal(%s) unexpected error: %v", tc.name, err)
			}
			if _, err := cudaflightrecorder.DecodeNegativeRewriteConstraint(data); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
				t.Fatalf("DecodeNegativeRewriteConstraint(%s) err = %v, want ErrArtifactInvalid", tc.name, err)
			}
		})
	}

	// 3. Bundle.DecodeFaultEvidence refuses withheld quarantine bundles and mismatched digests/kernels.
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Close()

	var withheldBuf bytes.Buffer
	if _, err := rec.FaultSnapshot(context.Background(), &withheldBuf, cudaflightrecorder.FaultInfo{
		ErrorName:       "CUDA_ERROR_ILLEGAL_ADDRESS",
		ErrorMessage:    "fault without module digest",
		OffendingKernel: "gemm_kernel",
	}); err != nil {
		t.Fatal(err)
	}
	withheldBundle, err := cudaflightrecorder.ReadBundle(&withheldBuf)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := withheldBundle.DecodeFaultEvidence(); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
		t.Fatalf("DecodeFaultEvidence on withheld bundle err = %v, want ErrArtifactInvalid", err)
	}

	// Mismatched candidate digest between fault.json and negative_rules.json must be refused.
	mismatchBundle := cudaflightrecorder.NewBundle(cudaflightrecorder.Manifest{})
	faultBytes, err := json.Marshal(cudaflightrecorder.FaultInfo{
		ErrorName:       "CUDA_ERROR_ILLEGAL_ADDRESS",
		ErrorMessage:    "fault",
		OffendingKernel: "gemm_kernel",
		ModuleDigest:    validDigest,
	})
	if err != nil {
		t.Fatal(err)
	}
	negBytes, err := json.Marshal(cudaflightrecorder.NegativeRewriteConstraint{
		CandidateDigest: otherDigest,
		OffendingKernel: "gemm_kernel",
		Reason:          "candidate-only quarantine",
	})
	if err != nil {
		t.Fatal(err)
	}
	mismatchBundle.Files["fault.json"] = faultBytes
	mismatchBundle.Files["negative_rules.json"] = negBytes
	if _, _, err := mismatchBundle.DecodeFaultEvidence(); !errors.Is(err, cudaflightrecorder.ErrArtifactInvalid) {
		t.Fatalf("DecodeFaultEvidence on mismatched digest err = %v, want ErrArtifactInvalid", err)
	}
}
