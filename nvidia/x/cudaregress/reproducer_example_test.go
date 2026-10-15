package cudaregress_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

func exampleReconstructionPlan(dir string) (*cudaregress.ExperimentPlan, error) {
	path := filepath.Join(dir, "supplied-source")
	if err := os.WriteFile(path, []byte("supplied source bytes, never compiled"), 0600); err != nil {
		return nil, err
	}
	digest, size, err := cudaregress.ComputeFileDigest(path)
	if err != nil {
		return nil, err
	}
	plan, _, err := cudaregress.NewPlanFromIncident(cudaregress.Incident{
		ID: "supplied-incident", Admitted: true,
		ObservedEnvironment: cudaregress.Environment{Model: "supplied-model"},
	}, cudaregress.PlanOptions{
		Sources: []cudaregress.ArtifactRef{{Name: "source", Path: path, Digest: digest, SizeBytes: size}},
	})
	return plan, err
}

func ExampleCheckReconstruction() {
	dir, err := os.MkdirTemp("", "reconstruction-example-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	plan, err := exampleReconstructionPlan(dir)
	if err != nil {
		fmt.Println(err)
		return
	}
	// This environment is supplied by the caller; the check does not observe it.
	ok, blockers := cudaregress.CheckReconstruction(plan, cudaregress.Environment{Model: "supplied-model"})
	fmt.Println(ok, len(blockers))
	// Output: true 0
}

func ExampleExportMinimalReproducer() {
	dir, err := os.MkdirTemp("", "reproducer-example-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	plan, err := exampleReconstructionPlan(dir)
	if err != nil {
		fmt.Println(err)
		return
	}
	target := filepath.Join(dir, "export")
	if err := cudaregress.ExportMinimalReproducer(plan, target); err != nil {
		fmt.Println(err)
		return
	}
	data, err := os.ReadFile(filepath.Join(target, "manifest.json"))
	if err != nil {
		fmt.Println(err)
		return
	}
	var manifest cudaregress.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("artifact closure:", manifest.ArtifactClosureValid)
	fmt.Println("host reconstruction:", manifest.Reconstructable)
	// Output:
	// artifact closure: true
	// host reconstruction: false
}
