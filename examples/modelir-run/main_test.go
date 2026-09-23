package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunSExpr(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "llama.sexpr")
	data := `#!/usr/bin/env modelir-run
(modelir 1
 (module-set
  (entry "LlamaForCausalLM")
  (weights "meta-llama/Llama-3.1-8B-Instruct")
  (arch
    (attention (qkv-kind "split"))
    (mlp (kind "swiglu") (act "silu"))
  )
  (module "LlamaDecoderLayer"
    (roles "decoder_block")
  )
  (module "LlamaForCausalLM")
))
`
	if err := os.WriteFile(model, []byte(data), 0o755); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if got := run([]string{model}, &stdout, &stderr); got != 0 {
		t.Fatalf("run returned %d: %s", got, stderr.String())
	}
	for _, want := range []string{
		"ModelIR Graph Runner (LlamaForCausalLM",
		"Modules: 2",
		"Architecture: attention=split, mlp=swiglu",
		"no executable graph declared",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("stdout missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestRunProgram(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "double.modelir")
	inputs := filepath.Join(dir, "inputs.json")
	if err := os.WriteFile(model, []byte(`modelir/v1
version: 1
entry: main
functions: main

-- main.graph --
input x : fp32 [2]

y : fp32 [2] = mul_scalar(x) {scale=2}
return y
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(inputs, []byte(`{"x":{"Shape":[2],"Data":[1.5,3]}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if got := run([]string{"-inputs", inputs, model}, &stdout, &stderr); got != 0 {
		t.Fatalf("run returned %d: %s", got, stderr.String())
	}
	if want := "y: shape=[2] data=[3 6]"; !strings.Contains(stdout.String(), want) {
		t.Errorf("stdout missing %q:\n%s", want, stdout.String())
	}
	if strings.Contains(stdout.String(), "Modules:") {
		t.Errorf("stdout reports modules for a v1 graph:\n%s", stdout.String())
	}
}

func TestRunUnifiedProgram(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "double.modelir")
	inputs := filepath.Join(dir, "inputs.json")
	if err := os.WriteFile(model, []byte(`modelir/v2
entry: Double

-- module Double --
roles: decoder_block

-- Double.forward --
input x : fp32 [2]

y : fp32 [2] = mul_scalar(x) {scale=2}
return y
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(inputs, []byte(`{"x":{"Shape":[2],"Data":[1.5,3]}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if got := run([]string{"-inputs", inputs, model}, &stdout, &stderr); got != 0 {
		t.Fatalf("run returned %d: %s", got, stderr.String())
	}
	for _, want := range []string{
		"ModelIR Graph Runner (Double",
		"Entry function: Double.forward",
		"Modules: 1",
		"Executed Double.forward.",
		"y: shape=[2] data=[3 6]",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Errorf("stdout missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestRunProgramWithWeights(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "add.modelir")
	inputs := filepath.Join(dir, "inputs.json")
	weights := filepath.Join(dir, "weights.json")
	if err := os.WriteFile(model, []byte(`modelir/v1
version: 1
entry: main
functions: main

-- main.graph --
input x : fp32 [2]
const b : fp32 [2] @weights/b

y : fp32 [2] = add(x, b)
return y
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(inputs, []byte(`{"x":{"Shape":[2],"Data":[1.5,3]}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(weights, []byte(`{"weights/b":{"Shape":[2],"Data":[1,2]}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if got := run([]string{"-inputs", inputs, "-weights", weights, model}, &stdout, &stderr); got != 0 {
		t.Fatalf("run returned %d: %s", got, stderr.String())
	}
	if want := "y: shape=[2] data=[2.5 5]"; !strings.Contains(stdout.String(), want) {
		t.Errorf("stdout missing %q:\n%s", want, stdout.String())
	}
}

func TestValidTensor(t *testing.T) {
	for _, test := range []struct {
		name string
		data string
		want string
	}{
		{"valid scalar", `{"x":{"Data":[1]}}`, ""},
		{"wrong size", `{"x":{"Shape":[2],"Data":[1]}}`, "shape has 2 values, data has 1"},
		{"negative dimension", `{"x":{"Shape":[-1],"Data":[]}}`, "negative dimension"},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tensors.json")
			if err := os.WriteFile(path, []byte(test.data), 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := readTensors(path)
			if got := errorText(err); got != test.want {
				t.Errorf("readTensors error = %q, want %q", got, test.want)
			}
		})
	}
}

func TestLoadModelRejectsEmptySExpr(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.sexpr")
	if err := os.WriteFile(path, []byte("()"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadModel(path); err == nil {
		t.Fatal("loadModel succeeded for an empty S-expression")
	}
}

func errorText(err error) string {
	if err == nil {
		return ""
	}
	return strings.TrimPrefix(err.Error(), "tensor \"x\": ")
}
