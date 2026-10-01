package main

import (
	"os"
	"os/exec"
	"testing"
)

func TestPythonStackAdapter(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is unavailable")
	}
	cmd := exec.Command(python, "cudaprofstacks_test.py")
	cmd.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("python stack adapter: %v\n%s", err, out)
	}
}
