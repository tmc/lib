package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func TestFaultPolicyCandidateOnly(t *testing.T) {
	for _, digest := range []string{strings.Repeat("a", 64), "", "candidate-only", strings.Repeat("z", 64)} {
		r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1})
		if err != nil {
			t.Fatal(err)
		}
		fault := cudaflightrecorder.FaultInfo{ModuleDigest: digest, ModuleID: "module", ModuleGeneration: 7, OffendingKernel: "kernel", FailedRuleNames: []string{"vectorize"}, ErrorName: "CUDA_ERROR_ILLEGAL_ADDRESS", ErrorMessage: "original fault"}
		var b bytes.Buffer
		if _, err := r.FaultSnapshot(context.Background(), &b, fault); err != nil {
			t.Fatal(err)
		}
		_ = r.Close()
		bundle, err := cudaflightrecorder.ReadBundle(&b)
		if err != nil {
			t.Fatal(err)
		}
		var got cudaflightrecorder.FaultInfo
		var policy cudaflightrecorder.NegativeRewriteConstraint
		if err := json.Unmarshal(bundle.Files["fault.json"], &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, fault) {
			t.Fatalf("fault facts changed: %+v", got)
		}
		if err := json.Unmarshal(bundle.Files["negative_rules.json"], &policy); err != nil {
			t.Fatal(err)
		}
		if policy.PruneSubtree || len(policy.QuarantinedRules) != 0 || strings.Contains(string(bundle.Files["negative_rules.json"]), "ban_kernel:") {
			t.Fatalf("broad policy invented: %+v", policy)
		}
		if digest == strings.Repeat("a", 64) {
			if policy.CandidateDigest != digest || !strings.Contains(policy.Reason, "candidate-only") {
				t.Fatalf("exact identity lost: %+v", policy)
			}
		} else if policy.CandidateDigest != "" || !strings.Contains(policy.Reason, "withheld") {
			t.Fatalf("unknown identity admitted: %+v", policy)
		}
	}
}

type faultFailWriter struct {
	err   error
	calls int
}

func (w *faultFailWriter) Write(p []byte) (int, error) { w.calls++; return 0, w.err }

func TestFaultSnapshotConfiguredPayloadBudget(t *testing.T) {
	fault := cudaflightrecorder.FaultInfo{ErrorName: "FAULT", ErrorMessage: "original", ModuleDigest: strings.Repeat("b", 64)}
	r, _ := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1})
	var reference bytes.Buffer
	if _, err := r.FaultSnapshot(context.Background(), &reference, fault); err != nil {
		t.Fatal(err)
	}
	_ = r.Close()
	bundle, err := cudaflightrecorder.ReadBundle(&reference)
	if err != nil {
		t.Fatal(err)
	}
	var size int64
	for _, data := range bundle.Files {
		size += int64(len(data))
	}
	for _, tc := range []struct {
		name       string
		limit      int64
		attachment int
		message    string
		pass       bool
	}{
		{"within", size + 32, 0, fault.ErrorMessage, true},
		{"below", size - 32, 0, fault.ErrorMessage, false},
		{"attachment-over", size, 64, fault.ErrorMessage, false},
		{"large-fault", 1024, 0, strings.Repeat("x", 4096), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, _ := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1, MaxSnapshotBytes: tc.limit})
			defer r.Close()
			if tc.attachment > 0 {
				r.AttachFile("attachment.bin", make([]byte, tc.attachment))
			}
			f := fault
			f.ErrorMessage = tc.message
			var output bytes.Buffer
			m, err := r.FaultSnapshot(context.Background(), &output, f)
			if tc.pass {
				if err != nil || m == nil || output.Len() == 0 {
					t.Fatalf("exact boundary refused: %v", err)
				}
			} else if !errors.Is(err, cudaflightrecorder.ErrSnapshotBudgetExceeded) || m != nil || output.Len() != 0 {
				t.Fatalf("preflight writes/incorrect error: %d %v", output.Len(), err)
			}
		})
	}
	r, _ = cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1})
	defer r.Close()
	sentinel := errors.New("original output failure")
	w := &faultFailWriter{err: sentinel}
	if _, err := r.FaultSnapshot(context.Background(), w, fault); !errors.Is(err, sentinel) || w.calls == 0 {
		t.Fatalf("original write error lost: %v", err)
	}
	r.AttachFile("negative_rules.json", []byte(`{"prune_subtree":true}`))
	var output bytes.Buffer
	if _, err := r.FaultSnapshot(context.Background(), &output, fault); err == nil || output.Len() != 0 {
		t.Fatal("attachment replaced fault constraint or wrote partial output")
	}
}
