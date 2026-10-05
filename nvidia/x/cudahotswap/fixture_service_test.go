package cudahotswap

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
)

type serviceFixtureKernel struct {
	*fixtureTestKernel
	serviceCloser
}

func (*serviceFixtureKernel) Raw() driver.Function { return 1 }
func (k *serviceFixtureKernel) Close() error       { return k.serviceCloser.Close() }

// This backend exercises actual recording through the public adapter while
// declaring its fake event clock. It acquires the actual test Service lease.
type serviceFixtureBackend struct {
	*fixtureTestBackend
	service *Service
	lease   io.Closer
}

func (b *serviceFixtureBackend) Begin(ctx context.Context) error {
	if b.lease == nil {
		lease, err := b.service.replayLease(ctx, "kernel")
		b.lease = lease
		if err != nil {
			return err
		}
	}
	return b.fixtureTestBackend.Begin(ctx)
}

func (b *serviceFixtureBackend) End(ctx context.Context) error {
	if err := b.fixtureTestBackend.End(ctx); err != nil {
		return err
	}
	if b.lease != nil {
		if err := b.lease.Close(); err != nil {
			return err
		}
		b.lease = nil
	}
	return nil
}

func (b *serviceFixtureBackend) Close(ctx context.Context) error {
	if err := b.fixtureTestBackend.Close(ctx); err != nil {
		return err
	}
	return b.End(ctx)
}

func (b *serviceFixtureBackend) Run(ctx context.Context, arm replayArm, spec replaySpec, timed bool) (d time.Duration, err error) {
	if k, ok := arm.Kernel.(*serviceFixtureKernel); ok {
		arm.Kernel = k.fixtureTestKernel
	}
	return b.fixtureTestBackend.Run(ctx, arm, spec, timed)
}

func serviceFixture(t *testing.T) (*Service, *serviceFixtureKernel, *serviceFixtureBackend, FixtureConfig, *[]*serviceCloser) {
	t.Helper()
	k := &serviceFixtureKernel{fixtureTestKernel: fixtureKernel("baseline", false)}
	l := &serviceLoader{kernel: k, compile: func(context.Context, Source) (*Compilation, error) {
		return &Compilation{Entry: "entry", Key: "fixture", PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry(.param .b64 x, .param .b64 y, .param .b64 w, .param .u32 n) { ret; }"}, nil
	}}
	s, leases := testService(t, l)
	registerService(t, s)
	b := &serviceFixtureBackend{fixtureTestBackend: fixtureBackend(), service: s}
	s.newReplay = func(name string, spec replaySpec, policy replayPolicy, bindings []replayDeviceBinding) (replayBackend, error) {
		if name != "kernel" || len(bindings) != 3 || len(spec.Signature.Params) != 4 {
			t.Fatal("factory lost kernel, ABI or pointer declarations")
		}
		return b, nil
	}
	spec := fixtureSpec()
	c := FixtureConfig{Args: spec.Args, Launch: spec.Config, Policy: DefaultReplayPolicy(), Outputs: []FixtureOutput{{Arg: 1, Elem: "f32"}}}
	for _, d := range spec.Buffers {
		c.Buffers = append(c.Buffers, FixtureBuffer{Arg: d.Arg, Bytes: d.Bytes, Stable: d.Stable})
	}
	return s, k, b, c, leases
}

func TestServiceRecordFixtureOwnsBaselineAndLease(t *testing.T) {
	s, k, b, config, leases := serviceFixture(t)
	g := LaunchGeometry{Grid: config.Launch.Grid, Block: config.Launch.Block, SharedMem: config.Launch.SharedMem}
	g.Grid.X = 2
	if _, err := s.SetGeometry("kernel", &g, "record override"); err != nil {
		t.Fatal(err)
	}
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if err != nil || f == nil || !f.replay.ready || f.State().Version != 1 || f.State().ConfigRevision != 1 || len(s.fixtures) != 1 {
		t.Fatalf("record: %v, %v", f, err)
	}
	if f.replay.spec.Config.Grid.X != 2 || config.Launch.Grid.X != 1 {
		t.Fatal("record ignored geometry override or changed caller launch")
	}
	if b.lease != nil || b.pending || len(*leases) != 2 || (*leases)[1].calls != 1 {
		t.Fatal("record retains idle lease or releases incomplete work")
	}
	if s.table.entries["kernel"].current.Load().pins.Load() != 1 || !reflect.DeepEqual(b.data[0], fixtureBytes(2, 3)) {
		t.Fatal("baseline not pinned or original inputs not restored")
	}
	config.Args[3] = uint32(99)
	config.Buffers[0].Bytes = 99
	config.Outputs[0].Elem = "bad"
	if f.replay.spec.Args[3] != uint32(2) || f.replay.spec.Buffers[0].Bytes != 8 || f.replay.spec.Outputs[0].Elem != "f32" {
		t.Fatal("fixture aliases caller declarations")
	}
	if err := f.Close(context.Background()); err != nil || len(s.fixtures) != 0 || k.calls != 0 {
		t.Fatalf("close fixture: %v", err)
	}
	if err := f.Close(context.Background()); err != nil || len(*leases) != 3 {
		t.Fatal("closed fixture acquires a second cleanup lease")
	}
	if err := s.Close(context.Background()); err != nil || k.calls != 1 {
		t.Fatalf("close service: %v", err)
	}
}

func TestServiceRetainedFixtureLimit(t *testing.T) {
	s, _, _, config, leases := serviceFixture(t)
	s.newReplay = func(string, replaySpec, replayPolicy, []replayDeviceBinding) (replayBackend, error) {
		return &serviceFixtureBackend{fixtureTestBackend: fixtureBackend(), service: s}, nil
	}
	var first *Fixture
	for i := 0; i < 32; i++ {
		f, err := s.RecordFixture(context.Background(), "kernel", config)
		if err != nil || f == nil {
			t.Fatalf("record %d: %v", i, err)
		}
		if first == nil {
			first = f
		}
	}
	if f, err := s.RecordFixture(context.Background(), "kernel", config); f != nil || err == nil || len(*leases) != 33 {
		t.Fatal("retained fixture limit does not refuse before lease")
	}
	if err := first.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f, err := s.RecordFixture(context.Background(), "kernel", config); err != nil || f == nil || len(s.fixtures) != 32 {
		t.Fatal("successful fixture close did not restore capacity")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceFailedFixtureRetainedThroughShutdown(t *testing.T) {
	s, k, b, config, _ := serviceFixture(t)
	b.prepareErr = errors.New("failed after backend and pin adoption")
	f, err := s.RecordFixture(context.Background(), "kernel", config)
	if f == nil || err == nil || f.replay.ready || len(s.fixtures) != 1 {
		t.Fatal("record error discarded its adopted fixture")
	}
	b.closeErr = errors.New("allocation cleanup failed")
	if err := s.Close(context.Background()); !errors.Is(err, b.closeErr) || k.calls != 0 || len(s.fixtures) != 1 || b.lease == nil {
		t.Fatal("failed shutdown lost owners, released lease or unloaded module")
	}
	b.closeErr = nil
	if err := s.Close(context.Background()); err != nil || len(s.fixtures) != 0 || k.calls != 1 || b.lease != nil {
		t.Fatalf("retry shutdown: %v", err)
	}
	if err := f.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceRecordFixtureRefusesBeforeLease(t *testing.T) {
	s, _, b, config, leases := serviceFixture(t)
	for _, tc := range []struct {
		name   string
		ctx    context.Context
		change func(*FixtureConfig)
	}{
		{name: "nil context"},
		{name: "unknown kernel", ctx: context.Background()},
		{name: "zero rounds", ctx: context.Background(), change: func(c *FixtureConfig) { c.Policy.Rounds = 0 }},
		{name: "missing pointer", ctx: context.Background(), change: func(c *FixtureConfig) { c.Args = []any{nil} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := config
			if tc.change != nil {
				tc.change(&c)
			}
			name := "kernel"
			if tc.name == "unknown kernel" {
				name = "missing"
			}
			f, err := s.RecordFixture(tc.ctx, name, c)
			if f != nil || err == nil || len(*leases) != 1 || len(s.fixtures) != 0 || b.lease != nil {
				t.Fatal("pre-adoption refusal acquires a lease or retains a fixture")
			}
		})
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, f := range []*Fixture{nil, new(Fixture)} {
		if f.Close(context.Background()) != nil || f.Close(nil) == nil || f.State() != (State{}) {
			t.Fatal("zero fixture contract")
		}
	}
}
