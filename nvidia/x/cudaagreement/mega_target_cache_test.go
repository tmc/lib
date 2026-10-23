package cudaagreement

import (
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func cacheProgram(t *testing.T, target cudamega.Target, scalar uint32) *cudamega.Program {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "add", Bytes: 16, Scope: cudataskgraph.ScopeCTA},
	})
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudamega.NewProgram(g, map[string]cudamega.Operation{
		"add": {Kind: cudamega.Add, Scalar: scalar},
	}, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32, MaxElements: 4}, target)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// cacheImage is a host-only compiler stand-in. It is never loaded or executed.
func cacheImage(target cudamega.Target) []byte {
	return []byte(fmt.Sprintf(".version 8.0\n.target %s\n.address_size 64\n", target))
}

func TestMegaArtifactCacheIdentity(t *testing.T) {
	p80 := cacheProgram(t, cudamega.SM80, 1)
	p90 := cacheProgram(t, cudamega.SM90, 1)
	s80, err := p80.Source()
	if err != nil {
		t.Fatal(err)
	}
	s90, err := p90.Source()
	if err != nil {
		t.Fatal(err)
	}
	if s80 != s90 {
		t.Fatal("target control requires identical emitted source")
	}
	var calls int
	e := &MegaExecutor{Compile: func(src string, target cudamega.Target) (cudakernel.Kind, []byte, error) {
		calls++
		return cudakernel.KindPTX, cacheImage(target), nil
	}}
	var first *cudakernel.Artifact
	for _, tt := range []struct {
		name   string
		target cudamega.Target
		p      *cudamega.Program
		calls  int
	}{
		{"sm80", cudamega.SM80, p80, 1},
		{"sm80_hit", cudamega.SM80, p80, 1},
		{"sm90", cudamega.SM90, p90, 2},
		{"sm90_hit", cudamega.SM90, p90, 2},
		{"sm80_return", cudamega.SM80, p80, 2},
		{"changed_source", cudamega.SM80, cacheProgram(t, cudamega.SM80, 2), 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e.Target = tt.target
			a, err := e.artifact(tt.p)
			if err != nil {
				t.Fatal(err)
			}
			if calls != tt.calls || a.Target() != string(tt.target) || string(a.Bytes()) != string(cacheImage(tt.target)) {
				t.Fatalf("calls=%d target=%s bytes=%q, want calls=%d target=%s", calls, a.Target(), a.Bytes(), tt.calls, tt.target)
			}
			if tt.name == "sm80" {
				first = a
			}
			if (tt.name == "sm80_hit" || tt.name == "sm80_return") && a != first {
				t.Fatal("unchanged source and target did not reuse the artifact")
			}
		})
	}
}

func TestMegaArtifactTargetSnapshot(t *testing.T) {
	e := &MegaExecutor{Target: cudamega.SM80}
	var calls int
	e.Compile = func(src string, target cudamega.Target) (cudakernel.Kind, []byte, error) {
		calls++
		if target != cudamega.SM80 {
			t.Fatalf("compiler target=%s, want sm_80", target)
		}
		// A synchronous callback change distinguishes the captured target from
		// a later field read. This does not exercise concurrent field access.
		e.Target = cudamega.SM90
		return cudakernel.KindPTX, cacheImage(target), nil
	}
	p := cacheProgram(t, cudamega.SM80, 1)
	a, err := e.artifact(p)
	if err != nil {
		t.Fatal(err)
	}
	if a.Target() != "sm_80" {
		t.Fatalf("artifact target=%s, want captured sm_80", a.Target())
	}
	e.Target = cudamega.SM80
	b, err := e.artifact(p)
	if err != nil || b != a || calls != 1 {
		t.Fatalf("captured key lookup: same=%v calls=%d err=%v", b == a, calls, err)
	}
}

func TestMegaArtifactCompilerFailure(t *testing.T) {
	want := errors.New("compiler failed")
	e := &MegaExecutor{Target: cudamega.SM80}
	var calls int
	e.Compile = func(src string, target cudamega.Target) (cudakernel.Kind, []byte, error) {
		calls++
		if calls == 1 {
			return cudakernel.KindPTX, cacheImage(target), want
		}
		return cudakernel.KindPTX, cacheImage(target), nil
	}
	p := cacheProgram(t, cudamega.SM80, 1)
	if a, err := e.artifact(p); a != nil || !errors.Is(err, want) || len(e.compiled) != 0 {
		t.Fatalf("failed compile: artifact=%v err=%v cache size=%d", a, err, len(e.compiled))
	}
	if a, err := e.artifact(p); err != nil || a == nil || calls != 2 || len(e.compiled) != 1 {
		t.Fatalf("successful compile after failure: artifact=%v err=%v calls=%d cache size=%d", a, err, calls, len(e.compiled))
	}
}
