package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func TestEngineStepBoundaries(t *testing.T) {
	for _, n := range []int{1, 128} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			var clock int64
			now := func() time.Time { return time.Unix(0, clock) }
			drains, prefills, decodes := 0, 0, 0
			var emitted []int32
			drain := func() error { drains++; clock += 100; return nil }
			prefill := func(tok int32, pos int) (int32, error) {
				if pos != prefills || tok != int32(pos+1) {
					t.Fatalf("prefill token=%d pos=%d", tok, pos)
				}
				prefills++
				clock += 3
				return tok + 10, nil
			}
			plain := func(step int, tok int32, pos int) (int32, error) {
				if step != decodes || pos != 2+step || tok != int32(12+step) {
					t.Fatalf("decode step=%d tok=%d pos=%d", step, tok, pos)
				}
				decodes++
				clock += 3
				return tok + 1, nil
			}
			emit := func(tok int32) error { emitted = append(emitted, tok); clock += 10000; return nil }
			packets, err := generateEngineSteps([]int32{1, 2}, n, n+1, drain, prefill, plain, emit, now)
			if err != nil {
				t.Fatal(err)
			}
			if drains != n || prefills != 2 || decodes != n-1 || len(emitted) != n || len(packets) != n {
				t.Fatalf("calls drain=%d prefill=%d decode=%d emitted=%d packets=%d", drains, prefills, decodes, len(emitted), len(packets))
			}
			for i, p := range packets {
				wantNS := int64(3)
				phase := "decode"
				if i == 0 {
					wantNS = 6
					phase = "prefill"
				}
				if p.OutputIndex != i || p.ContextLen != 2+i || p.TokenID != int32(12+i) || p.WallNS != wantNS || p.Phase != phase || p.Boundary != engineStepBoundary {
					t.Fatalf("packet %d: %+v", i, p)
				}
			}
			var buf bytes.Buffer
			if err := writeEngineSteps(&buf, packets); err != nil {
				t.Fatal(err)
			}
			lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
			if len(lines) != n {
				t.Fatal(len(lines))
			}
			for i, line := range lines {
				var p engineStep
				if !strings.HasPrefix(line, "ENGINE_STEP ") {
					t.Fatal(line)
				}
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "ENGINE_STEP ")), &p); err != nil || p != packets[i] {
					t.Fatalf("receipt: %s %v", line, err)
				}
			}
		})
	}
}

func TestEngineStepFailure(t *testing.T) {
	sentinel := errors.New("test failure")
	for _, where := range []string{"drain", "prefill", "decode", "emit", "clock"} {
		t.Run(where, func(t *testing.T) {
			calls := 0
			clock := int64(0)
			drain := func() error {
				if where == "drain" {
					return sentinel
				}
				return nil
			}
			prefill := func(int32, int) (int32, error) {
				clock++
				if where == "prefill" {
					return 0, sentinel
				}
				return 1, nil
			}
			plain := func(int, int32, int) (int32, error) {
				clock++
				if where == "decode" {
					return 0, sentinel
				}
				return 2, nil
			}
			emit := func(int32) error {
				calls++
				if where == "emit" {
					return sentinel
				}
				return nil
			}
			now := func() time.Time {
				if where == "clock" {
					return time.Unix(0, 0)
				}
				return time.Unix(0, clock)
			}
			packets, err := generateEngineSteps([]int32{0}, 2, 2, drain, prefill, plain, emit, now)
			if err == nil || packets != nil {
				t.Fatalf("failure produced success packets=%v err=%v", packets, err)
			}
			if where != "clock" && !errors.Is(err, sentinel) {
				t.Fatal(err)
			}
			want := 0
			if where == "decode" || where == "emit" {
				want = 1
			}
			if calls != want {
				t.Fatalf("emit calls=%d want=%d", calls, want)
			}
		})
	}
}

func TestEngineStepValidation(t *testing.T) {
	good := engineStepOptions{Steps: 128, Greedy: true, ExplicitTokens: true}
	if err := validateEngineStepOptions(good); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"steps", "tokens", "draft", "tp", "bench", "step", "trace", "profile", "disasm", "emit", "greedy"} {
		t.Run(name, func(t *testing.T) {
			o := good
			switch name {
			case "steps":
				o.Steps = 127
			case "tokens":
				o.ExplicitTokens = false
			case "draft":
				o.Draft = 1
			case "tp":
				o.TP = 1
			case "bench":
				o.Bench = true
			case "step":
				o.StepTimes = true
			case "trace":
				o.Trace = true
			case "profile":
				o.Profile = true
			case "disasm":
				o.Disasm = true
			case "emit":
				o.Emit = true
			case "greedy":
				o.Greedy = false
			}
			if validateEngineStepOptions(o) == nil {
				t.Fatal("conflict accepted")
			}
		})
	}
	for _, tc := range []struct {
		p      []int32
		n, max int
	}{{nil, 128, 256}, {[]int32{1}, 0, 256}, {[]int32{1}, 128, 127}} {
		calls := 0
		fail := func() error { calls++; return nil }
		_, err := generateEngineSteps(tc.p, tc.n, tc.max, fail, nil, nil, nil, time.Now)
		if err == nil || calls != 0 {
			t.Fatalf("invalid inputs accepted or drained: %v %d", err, calls)
		}
	}
}

// The CLI must reject incompatible requests before opening a model or CUDA.
func TestEngineStepConflictBeforeCUDA(t *testing.T) {
	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	stdout := os.Stdout
	os.Stdout = devNull
	oldMode, oldSteps, oldTokens := *engineStepTimesFlag, *stepsFlag, *tokenIDsFlag
	t.Cleanup(func() {
		*engineStepTimesFlag = oldMode
		*stepsFlag = oldSteps
		*tokenIDsFlag = oldTokens
		os.Stdout = stdout
		if err := devNull.Close(); err != nil {
			t.Error(err)
		}
	})
	*engineStepTimesFlag = true
	*stepsFlag = 127
	*tokenIDsFlag = "1"
	if err := run(); err == nil || !strings.Contains(err.Error(), "requires -steps=128") {
		t.Fatalf("early flag rejection: %v", err)
	}
}

func TestEngineStepWarmup(t *testing.T) {
	resets, drains, prefills, decodes := 0, 0, 0, 0
	var clock int64
	reset := func() error { resets++; return nil }
	drain := func() error { drains++; return nil }
	prefill := func(tok int32, pos int) (int32, error) { prefills++; clock++; return tok + 1, nil }
	plain := func(step int, tok int32, pos int) (int32, error) { decodes++; clock++; return tok + 1, nil }
	ids, err := warmupEngineSteps([]int32{1, 2}, 128, 256, reset, drain, prefill, plain, func() time.Time { return time.Unix(0, clock) })
	if err != nil {
		t.Fatal(err)
	}
	if resets != 2 || drains != 130 || prefills != 2 || decodes != 127 || len(ids) != 128 || ids[0] != 3 || ids[127] != 130 {
		t.Fatalf("warmup reset=%d drain=%d prefill=%d decode=%d ids=%v", resets, drains, prefills, decodes, ids)
	}
}
