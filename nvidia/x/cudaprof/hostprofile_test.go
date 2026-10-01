package cudaprof

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"runtime/pprof"
	"runtime/trace"
	"testing"
)

func TestHostProfile(t *testing.T) {
	var tr, cpu bytes.Buffer
	p, err := StartHostProfile(&tr, &cpu)
	if err != nil {
		t.Fatal(err)
	}
	p.Stop()
	p.Stop()
	if tr.Len() == 0 || cpu.Len() == 0 {
		t.Fatalf("empty outputs: trace=%d cpu=%d", tr.Len(), cpu.Len())
	}
}

func TestHostProfilePreservesOwners(t *testing.T) {
	t.Run("trace", func(t *testing.T) {
		var existing, output bytes.Buffer
		if err := trace.Start(&existing); err != nil {
			t.Fatal(err)
		}
		defer trace.Stop()
		if _, err := StartHostProfile(&output, nil); err == nil {
			t.Fatal("accepted active trace")
		}
		if !trace.IsEnabled() {
			t.Fatal("stopped existing trace")
		}
	})
	t.Run("cpu", func(t *testing.T) {
		var existing, output, tr bytes.Buffer
		if err := pprof.StartCPUProfile(&existing); err != nil {
			t.Fatal(err)
		}
		defer pprof.StopCPUProfile()
		if _, err := StartHostProfile(&tr, &output); err == nil {
			t.Fatal("accepted active CPU profile")
		}
		if trace.IsEnabled() {
			t.Fatal("failed startup left trace running")
		}
		if err := pprof.StartCPUProfile(&output); err == nil {
			pprof.StopCPUProfile()
			t.Fatal("stopped existing CPU profile")
		}
	})
}

func ExampleStartHostProfile() {
	var runtimeTrace, cpuProfile bytes.Buffer
	p, err := StartHostProfile(&runtimeTrace, &cpuProfile)
	if err != nil {
		fmt.Println(err)
		return
	}
	p.Stop()
	fmt.Println(runtimeTrace.Len() > 0, cpuProfile.Len() > 0)
	// Output: true true
}

func ExampleHostProfile_Stop() {
	var p HostProfile
	p.Stop()
	p.Stop()
	fmt.Println("stopped")
	// Output: stopped
}

type failingHostWriter struct{ err error }

func (w failingHostWriter) Write([]byte) (int, error) { return 0, w.err }

func TestHostProfileWriterErrors(t *testing.T) {
	for _, name := range []string{"trace", "cpu"} {
		t.Run(name, func(t *testing.T) {
			want := fmt.Errorf("controlled write failure")
			var tr, cpu bytes.Buffer
			var traceOutput, cpuOutput io.Writer = &tr, &cpu
			if name == "trace" {
				traceOutput = failingHostWriter{want}
			} else {
				cpuOutput = failingHostWriter{want}
			}
			p, err := StartHostProfile(traceOutput, cpuOutput)
			if err != nil {
				t.Fatal(err)
			}
			if err := p.Stop(); !errors.Is(err, want) {
				t.Fatalf("Stop = %v, want %v", err, want)
			}
			if err := p.Stop(); !errors.Is(err, want) {
				t.Fatalf("repeated Stop = %v", err)
			}
		})
	}
}
