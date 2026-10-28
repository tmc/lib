package main

import (
	"flag"
	"io"
	"testing"
)

func TestInterpositionFlag(t *testing.T) {
	var declarations interpositionFlags
	f := flag.NewFlagSet("test", flag.ContinueOnError)
	f.SetOutput(io.Discard)
	f.Var(&declarations, "interposition", "")
	if err := f.Parse([]string{"-interposition", "cuLaunchKernel=declared_active=run=manifest", "-interposition", "cuCtxSynchronize=declared_inactive=caller"}); err != nil {
		t.Fatal(err)
	}
	if len(declarations) != 2 || declarations[0].Interposition.Evidence != "run=manifest" || declarations[1].Interposition.Status != "declared_inactive" {
		t.Fatalf("%+v", declarations)
	}
	for _, value := range []string{"bad", "=declared_active=caller", "cuLaunchKernel=declared_active=caller", "other=verified=caller", "other=declared_active="} {
		before := len(declarations)
		if err := declarations.Set(value); err == nil {
			t.Fatalf("accepted %q", value)
		}
		if len(declarations) != before {
			t.Fatal("failed parse mutated declarations")
		}
	}
}
