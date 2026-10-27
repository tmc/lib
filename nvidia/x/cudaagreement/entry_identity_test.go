package cudaagreement_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func TestVerifyEntryPointIdentity(t *testing.T) {
	const target = "sm_121a"
	const header = ".version 9.0\n.target sm_121a\n.address_size 64\n"
	// One owned image contains two different integer kernels with the same ABI.
	// This test qualifies record admission, not compilation or device execution.
	add := strings.Replace(strings.TrimPrefix(binaryPTX(target, "add.s32"), header), ".entry apply(", ".entry add_entry(", 1)
	mul := strings.Replace(strings.TrimPrefix(binaryPTX(target, "mul.lo.s32"), header), ".entry apply(", ".entry mul_entry(", 1)
	image := []byte(header + add + mul)
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "b", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	artifacts := make(map[string]*cudakernel.Artifact)
	for _, entry := range []string{"add_entry", "mul_entry"} {
		a, err := cudakernel.NewArtifact(cudakernel.KindPTX, image, entry, schema, target)
		if err != nil {
			t.Fatal(err)
		}
		artifacts[entry] = a
	}
	if artifacts["add_entry"].Digest() != artifacts["mul_entry"].Digest() || artifacts["add_entry"].EntryPoint() == artifacts["mul_entry"].EntryPoint() {
		t.Fatal("fixture needs one image digest with different selected entries")
	}
	arm := func(name string, backend cudaplan.TargetBackend, entry string) cudaagreement.Arm {
		a := record(name, backend, backend, []byte{1, 2, 3, 4}, "kernel")
		a.Record.Steps[0].Digest = artifacts[entry].Digest()
		a.Record.Steps[0].EntryPoint = artifacts[entry].EntryPoint()
		return a
	}
	for _, tt := range []struct {
		name, first, second string
		wantError           bool
	}{
		{"same entry", "add_entry", "add_entry", false},
		{"different entry", "add_entry", "mul_entry", true},
		{"reversed entries", "mul_entry", "add_entry", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := cudaagreement.Verify([]cudaagreement.Arm{
				arm("launch", cudaplan.BackendLaunch, tt.first),
				arm("graph", cudaplan.BackendGraph, tt.second),
			})
			if tt.wantError {
				if err == nil || !strings.Contains(err.Error(), "used entry") {
					t.Errorf("Verify different entries = %v, want entry mismatch", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, order := range [][3]int{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}} {
		for _, mismatch := range []string{"none", "entry", "geometry"} {
			t.Run(fmt.Sprintf("three_arms/%d%d%d/%s", order[0], order[1], order[2], mismatch), func(t *testing.T) {
				mega := arm("mega", cudaplan.BackendMega, "add_entry")
				mega.Record.Steps[0].Program = "worker-digest"
				mega.Record.Steps[0].EntryPoint = "dag_worker"
				mega.Record.Steps[0].Grid, mega.Record.Steps[0].Block = 48, 128
				launch := arm("launch", cudaplan.BackendLaunch, "add_entry")
				graph := arm("graph", cudaplan.BackendGraph, "add_entry")
				want := ""
				switch mismatch {
				case "entry":
					graph.Record.Steps[0].EntryPoint = artifacts["mul_entry"].EntryPoint()
					want = "used entry"
				case "geometry":
					graph.Record.Steps[0].Grid = 8
					want = "ran "
				}
				arms := []cudaagreement.Arm{mega, launch, graph}
				err := cudaagreement.Verify([]cudaagreement.Arm{arms[order[0]], arms[order[1]], arms[order[2]]})
				if want == "" {
					if err != nil {
						t.Fatal(err)
					}
				} else if err == nil || !strings.Contains(err.Error(), want) {
					t.Errorf("Verify three arms = %v, want %s mismatch", err, mismatch)
				}
			})
		}
	}
}
