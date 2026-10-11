//go:build linux && native

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// These controls call only the writer and process-map receipt code.
// Reports and states are explicitly synthetic; there is no CUDA launch.
func TestSidecarNativeWriterRestoredBeforeEdit(t *testing.T) {
	for _, refuse := range []bool{false, true} {
		t.Run(map[bool]string{false: "positive", true: "refusal"}[refuse], func(t *testing.T) {
			rows, reports, _, codes := sidecarNativeContractFixture()
			dir := t.TempDir()
			path := filepath.Join(dir, "candidate.cu")
			if err := os.WriteFile(path, []byte(codes[1]), 0600); err != nil {
				t.Fatal(err)
			}
			proof, err := nativereceipt.New()
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := proof.Close(); err != nil {
					t.Error(err)
				}
			}()
			checked := false
			sentinel := errors.New("restored control refusal")
			w := &sidecarNativeWatch{path: path, dir: dir, codes: codes, index: 1, proof: proof, reports: []*cudahotswap.PushResult{reports[0]}, cancel: func() {}}
			w.validate = func(index int, r *cudahotswap.PushResult, text string) error {
				return sidecarValidateNativeReport(index, r, text, codes[index], rows[3])
			}
			w.recheck = func(state cudahotswap.State) error {
				if state != reports[1].Selected {
					t.Fatal("restored state differs")
				}
				body, err := os.ReadFile(path)
				if err != nil {
					return err
				}
				if string(body) != codes[1] {
					t.Fatal("edit happened before restored launch control")
				}
				checked = true
				if refuse {
					return sentinel
				}
				return nil
			}
			data, err := json.Marshal(struct {
				Kernel string
				Result *cudahotswap.PushResult
				Error  string
			}{"kernel", reports[1], "wrong"})
			if err != nil {
				t.Fatal(err)
			}
			_, err = w.Write(data)
			if !checked {
				t.Fatal("restored launch control omitted")
			}
			if refuse && !errors.Is(err, sentinel) || !refuse && err != nil {
				t.Fatal(err)
			}
			body, e := os.ReadFile(path)
			if e != nil {
				t.Fatal(e)
			}
			expected := codes[2]
			if refuse {
				expected = codes[1]
			}
			if string(body) != expected {
				t.Fatalf("selected edit %q, want %q", body, expected)
			}
			raw, e := os.ReadFile(filepath.Join(dir, "raw-report-0.json"))
			if e != nil || string(raw) != string(data) {
				t.Fatalf("raw report lost: %v", e)
			}
			if err := w.releasePin(context.Background()); err != nil {
				t.Fatal(err)
			}
		})
	}
}
