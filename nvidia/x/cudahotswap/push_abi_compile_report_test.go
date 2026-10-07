package cudahotswap

import (
	"context"
	"crypto/sha256"
	"fmt"
	"testing"
)

func TestPushABIRefusalCompileReport(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, leases := testService(t, l)
	registerService(t, s)
	defer func() {
		if err := s.Close(context.Background()); err != nil {
			t.Fatal(err)
		}
	}()
	checks := 0
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error {
		checks++
		return nil
	}}, "check"); err != nil {
		t.Fatal(err)
	}
	ptx := ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry(.param .u32 incompatible) { ret; }"
	compiled := 0
	l.compile = func(context.Context, Source) (*Compilation, error) {
		compiled++
		return &Compilation{Entry: "entry", PTX: ptx}, nil
	}
	source := Source{Code: "actual CPU compiler fixture input", Entry: "entry"}
	result, err := s.Push(context.Background(), "kernel", source, "ABI candidate")
	if err == nil || result == nil || result.Report.Outcome != OutcomeABIMismatch || result.Published || result.RolledBack {
		t.Fatalf("ABI refusal: result=%+v err=%v", result, err)
	}
	c := result.Report.Compile
	if c == nil || c.SourceSHA256 != fmt.Sprintf("%x", sha256.Sum256([]byte(source.Code))) || c.SourceBytes != len(source.Code) || c.PTXSHA256 != "" || c.PTXBytes != 0 {
		t.Fatalf("preparation refused after compilation: %+v", c)
	}
	if compiled != 1 || l.loads != 1 || len(*leases) != 1 || checks != 0 || result.Report.Replay != nil {
		t.Fatalf("ABI refusal reached lease/load/check: compiles=%d loads=%d leases=%d checks=%d", compiled, l.loads, len(*leases), checks)
	}
}
