package anerun

import (
	"os"
	"testing"

	"github.com/tmc/lib/apple/x/ane/anemil"
)

func TestPackageMLModel(t *testing.T) {
	prog := anemil.NewProgram()
	prog.AddFunction(anemil.Function{
		Name:       "main",
		InputName:  "x",
		InputType:  anemil.TensorType{DType: anemil.FP32, Shape: []int{1, 32, 1, 1}},
		OutputName: "y",
		Statements: []anemil.Statement{
			{
				OutputVar:  "y",
				OutputType: anemil.TensorType{DType: anemil.FP16, Shape: []int{1, 32, 1, 1}},
				Op:         "cast",
				Attributes: map[string]string{
					"dtype": `string("fp16")`,
					"x":     "x",
				},
			},
		},
	})

	milText := prog.Serialize()
	tmpDir, err := os.MkdirTemp("", "ane-test-*")
	if err != nil {
		t.Fatalf("MkdirTemp: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	pkgPath, err := PackageMLModel(milText, tmpDir)
	if err != nil {
		t.Fatalf("PackageMLModel failed: %v", err)
	}

	if _, err := os.Stat(pkgPath); os.IsNotExist(err) {
		t.Errorf("packaged mlpackage does not exist at %s", pkgPath)
	}
	t.Logf("Successfully created ANE model package at: %s", pkgPath)
}
