package cudahotswap

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

type fixtureTestKernel struct {
	name string
	run  func(cudalaunch.Config, []any) error
	wait func(context.Context) error
}

func (*fixtureTestKernel) Raw() driver.Function                         { return 0 }
func (k *fixtureTestKernel) Launch(c cudalaunch.Config, a ...any) error { return k.run(c, a) }
func (k *fixtureTestKernel) Wait(ctx context.Context, _ *cuda.Stream) error {
	if k.wait != nil {
		return k.wait(ctx)
	}
	return ctx.Err()
}
func (*fixtureTestKernel) Close() error { return nil }

// This deliberately labels timings as fake. It implements the engine's actual
// preparation, submission, completion and cleanup contract without CUDA.
type fixtureTestBackend struct {
	data                                          map[int][]byte
	events                                        []string
	durations                                     map[string]time.Duration
	lease, pending, closed                        bool
	readErr, writeErr, beginErr, endErr, closeErr error
	prepareErr                                    error
}

func (b *fixtureTestBackend) Begin(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.events = append(b.events, "begin")
	b.lease = true
	return b.beginErr
}
func (b *fixtureTestBackend) Prepare(context.Context, replaySpec) error {
	b.events = append(b.events, "prepare")
	return b.prepareErr
}
func (b *fixtureTestBackend) Drain(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	b.events = append(b.events, "drain")
	b.pending = false
	return nil
}
func (b *fixtureTestBackend) End(ctx context.Context) error {
	if err := b.Drain(ctx); err != nil {
		return err
	}
	if b.endErr != nil {
		return b.endErr
	}
	b.lease = false
	b.events = append(b.events, "end")
	return nil
}
func (b *fixtureTestBackend) Read(arg int) ([]byte, error) {
	if !b.lease || b.pending || b.closed {
		return nil, fmt.Errorf("read without lease or before completion")
	}
	if b.readErr != nil {
		return nil, b.readErr
	}
	return append([]byte(nil), b.data[arg]...), nil
}
func (b *fixtureTestBackend) Write(arg int, data []byte) error {
	if !b.lease || b.pending || b.closed {
		return fmt.Errorf("write without lease or before completion")
	}
	if b.writeErr != nil {
		return b.writeErr
	}
	b.events = append(b.events, fmt.Sprintf("restore:%d", arg))
	b.data[arg] = append([]byte(nil), data...)
	return nil
}
func (b *fixtureTestBackend) Run(ctx context.Context, arm replayArm, spec replaySpec, timed bool) (time.Duration, error) {
	if !b.lease {
		return 0, fmt.Errorf("run without lease")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	k := arm.Kernel.(*fixtureTestKernel)
	args := append([]any(nil), spec.Args...)
	for _, buf := range spec.Buffers {
		args[buf.Arg] = b.data[buf.Arg]
	}
	if timed {
		b.events = append(b.events, "evict")
	}
	b.events = append(b.events, "launch:"+k.name)
	b.pending = true
	if err := k.Launch(spec.Config, args...); err != nil {
		return 0, err
	}
	if err := k.Wait(ctx, nil); err != nil {
		return 0, err
	}
	b.pending = false
	return b.durations[k.name], nil
}
func (b *fixtureTestBackend) Close(ctx context.Context) error {
	if !b.lease || b.pending {
		return fmt.Errorf("close before lease or completion")
	}
	if b.closeErr != nil {
		return b.closeErr
	}
	b.closed = true
	return b.End(ctx)
}
func (*fixtureTestBackend) Timing() (cudatrial.TimingKind, bool) { return "fake_event_interval", false }

func fixtureBytes(values ...float32) []byte {
	b := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}
func fixtureSpec() replaySpec {
	return replaySpec{
		Signature: cudalaunch.Signature{KernelName: "replay", Params: []cudalaunch.Param{{Type: "b64", Size: 8}, {Type: "b64", Size: 8}, {Type: "b64", Size: 8}, {Type: "u32", Size: 4}}},
		Args:      []any{nil, nil, nil, uint32(2)},
		Buffers:   []replayBuffer{{Arg: 0, Bytes: 8}, {Arg: 1, Bytes: 8}, {Arg: 2, Bytes: 8, Stable: true}},
		Outputs:   []replayOutput{{Arg: 1, Elem: "f32"}},
		Config:    cudalaunch.Config{Grid: cudashape.Grid3D{X: 1, Y: 1, Z: 1}, Block: driver.Dims(2, 1, 1), SharedMem: 16},
	}
}
func fixturePolicy() replayPolicy { return replayPolicy{Rounds: 4, Exact: true} }
func fixtureBackend() *fixtureTestBackend {
	return &fixtureTestBackend{data: map[int][]byte{0: fixtureBytes(2, 3), 1: fixtureBytes(-8, -9), 2: fixtureBytes(5, 7)}, durations: map[string]time.Duration{"baseline": 20 * time.Nanosecond, "candidate": 10 * time.Nanosecond}}
}
func fixtureKernel(name string, inplace bool) *fixtureTestKernel {
	return &fixtureTestKernel{name: name, run: func(cfg cudalaunch.Config, args []any) error {
		if cfg.Block.X != 2 || cfg.SharedMem != 16 || args[3] != uint32(2) {
			return fmt.Errorf("scalar or geometry lost")
		}
		x, _ := replayFloats(args[0].([]byte))
		w, _ := replayFloats(args[2].([]byte))
		out := args[1].([]byte)
		if inplace {
			x, _ = replayFloats(out)
		}
		copy(out, fixtureBytes(x[0]+w[0], x[1]+w[1]))
		// Both ordinary inputs and in-place outputs must be restored.
		copy(args[0].([]byte), fixtureBytes(80, 90))
		return nil
	}}
}
func fixtureArm(k *fixtureTestKernel) replayArm {
	return replayArm{Kernel: k, Release: func() error { return nil }}
}
func closeFixture(t *testing.T, f *replayFixture) {
	t.Helper()
	if err := f.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestReplayFixtureRestoresEveryArm(t *testing.T) {
	for _, inplace := range []bool{false, true} {
		t.Run(fmt.Sprint(inplace), func(t *testing.T) {
			b := fixtureBackend()
			original := append([]byte(nil), b.data[1]...)
			f, err := recordReplay(context.Background(), b, fixtureArm(fixtureKernel("baseline", inplace)), fixtureSpec(), fixturePolicy())
			if err != nil {
				t.Fatal(err)
			}
			defer closeFixture(t, f)
			if f.writeOnly[1] == inplace {
				t.Fatalf("write-only discovery=%v for inplace=%v", f.writeOnly[1], inplace)
			}
			if !bytes.Equal(b.data[1], original) || !bytes.Equal(b.data[0], fixtureBytes(2, 3)) || b.lease {
				t.Fatal("record did not restore/release")
			}
			b.events = nil
			r, err := f.Check(context.Background(), fixtureArm(fixtureKernel("candidate", inplace)))
			if err != nil {
				t.Fatal(err)
			}
			if !r.Pass || r.Native || r.Timing != "fake_event_interval" || r.Baseline != 20*time.Nanosecond || r.Candidate != 10*time.Nanosecond || r.CandidateWins != 4 || r.SignP != 0.125 {
				t.Fatalf("result=%+v", r)
			}
			var order []string
			for i, e := range b.events {
				if strings.HasPrefix(e, "launch:") {
					order = append(order, e)
					if i < 3 || b.events[i-1] != "evict" || b.events[i-2] != "restore:1" || b.events[i-3] != "restore:0" {
						t.Fatalf("preparation order=%v", b.events)
					}
				}
			}
			want := []string{"launch:baseline", "launch:candidate", "launch:candidate", "launch:baseline", "launch:baseline", "launch:candidate", "launch:candidate", "launch:baseline"}
			if !reflect.DeepEqual(order, want) {
				t.Fatalf("ABBA=%v", order)
			}
			if !bytes.Equal(b.data[1], original) || !bytes.Equal(b.data[0], fixtureBytes(2, 3)) || b.lease {
				t.Fatal("check did not restore/release")
			}
			for _, receipt := range r.Trials {
				if !receipt.Complete || receipt.DeviceNs != cudatrial.Unmeasured || receipt.HasTiming(cudatrial.TimingSummedKernel) {
					t.Fatalf("fake receipt claims device counter: %+v", receipt)
				}
			}
		})
	}
}

func TestReplayFixtureRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*replaySpec, *replayPolicy)
	}{
		{"zero extent", func(s *replaySpec, _ *replayPolicy) { s.Buffers[0].Bytes = 0 }},
		{"oversize", func(s *replaySpec, _ *replayPolicy) { s.Buffers[0].Bytes = replayMaxBytes }},
		{"duplicate buffer", func(s *replaySpec, _ *replayPolicy) { s.Buffers = append(s.Buffers, s.Buffers[0]) }},
		{"raw pointer", func(s *replaySpec, _ *replayPolicy) { s.Args[3] = uintptr(3) }},
		{"architecture int", func(s *replaySpec, _ *replayPolicy) { s.Args[3] = int(2) }},
		{"missing pointer", func(s *replaySpec, _ *replayPolicy) { s.Signature.Params[3].IsPointer = true }},
		{"output stable", func(s *replaySpec, _ *replayPolicy) { s.Outputs[0].Arg = 2 }},
		{"output element", func(s *replaySpec, _ *replayPolicy) { s.Outputs[0].Elem = "f16" }},
		{"output duplicate", func(s *replaySpec, _ *replayPolicy) { s.Outputs = append(s.Outputs, s.Outputs[0]) }},
		{"output partial element", func(s *replaySpec, _ *replayPolicy) { s.Buffers[1].Bytes = 7 }},
		{"zero geometry", func(s *replaySpec, _ *replayPolicy) { s.Config.Block.X = 0 }},
		{"overflow block", func(s *replaySpec, _ *replayPolicy) {
			s.Config.Block = driver.Dims(math.MaxUint32, math.MaxUint32, math.MaxUint32)
		}},
		{"negative tolerance", func(_ *replaySpec, p *replayPolicy) { p.RMS = -1 }},
		{"nan tolerance", func(_ *replaySpec, p *replayPolicy) { p.MaxAbs = math.NaN() }},
		{"unbounded rounds", func(_ *replaySpec, p *replayPolicy) { p.Rounds = 65 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, p := fixtureSpec(), fixturePolicy()
			tc.change(&s, &p)
			b := fixtureBackend()
			f, err := recordReplay(context.Background(), b, fixtureArm(fixtureKernel("baseline", false)), s, p)
			if err == nil {
				t.Fatal("accepted invalid schema")
			}
			if len(b.events) != 0 {
				t.Fatalf("invalid schema touched backend: %v", b.events)
			}
			closeFixture(t, f)
		})
	}
}

func TestReplayFixtureNoOpAndNoise(t *testing.T) {
	for _, name := range []string{"no-op", "nonfinite", "noise"} {
		t.Run(name, func(t *testing.T) {
			b := fixtureBackend()
			k := fixtureKernel("baseline", false)
			calls := 0
			switch name {
			case "no-op":
				k.run = func(cudalaunch.Config, []any) error { return nil }
			case "nonfinite":
				k.run = func(_ cudalaunch.Config, a []any) error {
					copy(a[1].([]byte), fixtureBytes(float32(math.NaN()), 1))
					return nil
				}
			case "noise":
				k.run = func(_ cudalaunch.Config, a []any) error {
					calls++
					copy(a[1].([]byte), fixtureBytes(float32(calls), 1))
					return nil
				}
			}
			f, err := recordReplay(context.Background(), b, fixtureArm(k), fixtureSpec(), fixturePolicy())
			if err == nil {
				t.Fatal("accepted invalid baseline")
			}
			closeFixture(t, f)
		})
	}
}

func TestReplayFixtureCandidateOutcomes(t *testing.T) {
	for _, name := range []string{"wrong", "nonfinite", "stable-write", "unresolved-timing"} {
		t.Run(name, func(t *testing.T) {
			b := fixtureBackend()
			f, err := recordReplay(context.Background(), b, fixtureArm(fixtureKernel("baseline", false)), fixtureSpec(), fixturePolicy())
			if err != nil {
				t.Fatal(err)
			}
			k := fixtureKernel("candidate", false)
			orig := k.run
			k.run = func(c cudalaunch.Config, a []any) error {
				if err := orig(c, a); err != nil {
					return err
				}
				switch name {
				case "wrong":
					copy(a[1].([]byte), fixtureBytes(7, 11))
				case "nonfinite":
					copy(a[1].([]byte), fixtureBytes(float32(math.Inf(1)), 10))
				case "stable-write":
					copy(a[2].([]byte), fixtureBytes(100, 101))
				}
				return nil
			}
			if name == "unresolved-timing" {
				b.durations["candidate"] = 0
			}
			r, err := f.Check(context.Background(), fixtureArm(k))
			if name == "stable-write" || name == "unresolved-timing" {
				if err == nil {
					t.Fatal("accepted harness failure")
				}
			} else {
				if err != nil || r.Pass || r.Outputs[0].Pass {
					t.Fatalf("wrong candidate=%+v %v", r, err)
				}
				if name == "nonfinite" && r.Outputs[0].Candidate.MaxAbs != nil {
					t.Fatal("nonfinite metric presented as finite")
				}
			}
			if !bytes.Equal(b.data[2], fixtureBytes(5, 7)) {
				t.Fatal("stable weights not recovered")
			}
			closeFixture(t, f)
		})
	}
}

func TestReplayFixtureCancellationRetainsOwners(t *testing.T) {
	b := fixtureBackend()
	baseReleased, candReleased := 0, 0
	base := fixtureArm(fixtureKernel("baseline", false))
	base.Release = func() error { baseReleased++; return nil }
	f, err := recordReplay(context.Background(), b, base, fixtureSpec(), fixturePolicy())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	k := fixtureKernel("candidate", false)
	orig := k.run
	k.run = func(c cudalaunch.Config, a []any) error {
		err := orig(c, a)
		copy(a[2].([]byte), fixtureBytes(99, 99))
		cancel()
		return err
	}
	candidate := fixtureArm(k)
	candidate.Release = func() error {
		if b.pending {
			t.Fatal("release before completion")
		}
		candReleased++
		return nil
	}
	_, err = f.Check(ctx, candidate)
	if !errors.Is(err, context.Canceled) || !b.pending || !b.lease || baseReleased != 0 || candReleased != 0 {
		t.Fatalf("canceled state err=%v pending=%v lease=%v pins=%d/%d", err, b.pending, b.lease, baseReleased, candReleased)
	}
	if err = f.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled close=%v", err)
	}
	if baseReleased != 0 || candReleased != 0 {
		t.Fatal("canceled close dropped pins")
	}
	closeFixture(t, f)
	if baseReleased != 1 || candReleased != 1 || !b.closed || b.lease || !bytes.Equal(b.data[2], fixtureBytes(5, 7)) {
		t.Fatal("valid retry did not drain, recover and release exactly once")
	}
}

func TestReplayFixtureExternalStableChangeIsNotOverwritten(t *testing.T) {
	b := fixtureBackend()
	f, err := recordReplay(context.Background(), b, fixtureArm(fixtureKernel("baseline", false)), fixtureSpec(), fixturePolicy())
	if err != nil {
		t.Fatal(err)
	}
	changed := fixtureBytes(80, 90)
	b.data[2] = changed
	b.events = nil
	if _, err := f.Check(context.Background(), fixtureArm(fixtureKernel("candidate", false))); err == nil {
		t.Fatal("external stable change accepted")
	}
	for _, event := range b.events {
		if strings.HasPrefix(event, "launch:") {
			t.Fatal("launched after stable input changed")
		}
	}
	closeFixture(t, f)
	if !bytes.Equal(b.data[2], changed) {
		t.Fatal("overwrote caller's external weight change")
	}
}

func TestReplayFixtureCloseRetries(t *testing.T) {
	for _, failure := range []string{"pin", "backend", "lease"} {
		t.Run(failure, func(t *testing.T) {
			b := fixtureBackend()
			releaseAttempts := 0
			fail := true
			arm := fixtureArm(fixtureKernel("baseline", false))
			arm.Release = func() error {
				releaseAttempts++
				if failure == "pin" && fail {
					return fmt.Errorf("pin close")
				}
				return nil
			}
			f, err := recordReplay(context.Background(), b, arm, fixtureSpec(), fixturePolicy())
			if err != nil {
				t.Fatal(err)
			}
			if failure == "backend" {
				b.closeErr = fmt.Errorf("backend close")
			}
			if failure == "lease" {
				b.endErr = fmt.Errorf("lease close")
			}
			if err = f.Close(context.Background()); err == nil || f.closed || !b.lease {
				t.Fatal("failed cleanup did not retain owner/lease")
			}
			fail = false
			b.closeErr = nil
			b.endErr = nil
			closeFixture(t, f)
			closeFixture(t, f)
			want := 1
			if failure == "pin" {
				want = 2
			}
			if releaseAttempts != want {
				t.Fatalf("release attempts=%d want=%d", releaseAttempts, want)
			}
		})
	}
}

func TestReplayFixturePartialFailuresRemainReachable(t *testing.T) {
	for _, failure := range []string{"prepare", "snapshot", "restore", "partial-acquire", "launch-after-write"} {
		t.Run(failure, func(t *testing.T) {
			b := fixtureBackend()
			k := fixtureKernel("baseline", false)
			sentinel := errors.New(failure)
			switch failure {
			case "prepare":
				b.prepareErr = sentinel
			case "snapshot":
				b.readErr = sentinel
			case "restore":
				b.writeErr = sentinel
			case "partial-acquire":
				b.beginErr = sentinel
			case "launch-after-write":
				orig := k.run
				k.run = func(c cudalaunch.Config, a []any) error {
					_ = orig(c, a)
					copy(a[2].([]byte), fixtureBytes(90, 91))
					return sentinel
				}
			}
			releases := 0
			arm := fixtureArm(k)
			arm.Release = func() error {
				if b.pending {
					t.Fatal("pin released before drain")
				}
				releases++
				return nil
			}
			f, err := recordReplay(context.Background(), b, arm, fixtureSpec(), fixturePolicy())
			if f == nil || !errors.Is(err, sentinel) || releases != 0 {
				t.Fatalf("lost partial owner: %p %v releases=%d", f, err, releases)
			}
			b.prepareErr, b.readErr, b.writeErr, b.beginErr = nil, nil, nil, nil
			closeFixture(t, f)
			if releases != 1 || b.lease || b.pending || !b.closed {
				t.Fatal("partial failure cleanup incomplete")
			}
			if failure == "launch-after-write" && !bytes.Equal(b.data[2], fixtureBytes(5, 7)) {
				t.Fatal("failed launch left stable mutation live")
			}
		})
	}
}

func TestReplayFixtureCandidatePinsBoundedAndRetryable(t *testing.T) {
	b := fixtureBackend()
	f, err := recordReplay(context.Background(), b, fixtureArm(fixtureKernel("baseline", false)), fixtureSpec(), fixturePolicy())
	if err != nil {
		t.Fatal(err)
	}
	releases := 0
	for i := 0; i < 8; i++ {
		arm := fixtureArm(fixtureKernel("candidate", false))
		arm.Release = func() error {
			if b.pending || !b.lease {
				t.Fatal("pin release outside drained lease")
			}
			releases++
			return nil
		}
		if _, err := f.Check(context.Background(), arm); err != nil {
			t.Fatal(err)
		}
		if len(f.pins) != 1 || releases != i+1 || b.lease {
			t.Fatal("candidate pin accumulated or lease retained while idle")
		}
	}
	fail := true
	arm := fixtureArm(fixtureKernel("candidate", false))
	arm.Release = func() error {
		releases++
		if fail {
			return fmt.Errorf("release failed")
		}
		return nil
	}
	if _, err := f.Check(context.Background(), arm); err == nil || !b.lease || len(f.pins) != 2 {
		t.Fatal("failed pin not retained")
	}
	fail = false
	closeFixture(t, f)
	if releases != 10 {
		t.Fatalf("release attempts=%d", releases)
	}
}

func TestReplayFixtureImmutableSchemaAndIndependentMetrics(t *testing.T) {
	b := fixtureBackend()
	s := fixtureSpec()
	f, err := recordReplay(context.Background(), b, fixtureArm(fixtureKernel("baseline", false)), s, fixturePolicy())
	if err != nil {
		t.Fatal(err)
	}
	defer closeFixture(t, f)
	s.Args[3] = uint32(99)
	s.Buffers[0].Bytes = 1
	s.Outputs[0].Elem = "bad"
	s.Signature.Params[0].Size = 1
	if _, err := f.Check(context.Background(), fixtureArm(fixtureKernel("candidate", false))); err != nil {
		t.Fatal(err)
	}
	a, _ := MeasureReplay([]float32{3, 2}, []float32{1, 2})
	bmetric, _ := MeasureReplay([]float32{1, 3}, []float32{1, 2})
	m := replayWorst(a, bmetric)
	if *m.MaxAbs != 2 || *m.RMS != math.Sqrt(2) || *m.MaxULP != *a.MaxULP || m.Exact {
		t.Fatalf("worst=%+v", m)
	}
	if replayMedian([]time.Duration{math.MaxInt64 - 2, math.MaxInt64}) != math.MaxInt64-1 {
		t.Fatal("median overflow")
	}
	if replaySignP(0, 0) != 1 || replaySignP(1, 1) != 1 || replaySignP(0, 4) != 0.125 {
		t.Fatal("sign test mismatch")
	}
}

func FuzzReplayFixtureSchema(f *testing.F) {
	f.Add(uint32(8), uint32(4), uint8(1), uint8(2))
	f.Fuzz(func(t *testing.T, extent, rounds uint32, arg, elem uint8) {
		s, p := fixtureSpec(), fixturePolicy()
		s.Buffers[0].Bytes = int(extent)
		s.Buffers[0].Arg = int(arg)
		p.Rounds = int(rounds)
		if elem%2 != 0 {
			s.Outputs[0].Elem = "f16"
		}
		_ = validateReplay(s, p)
	})
}
