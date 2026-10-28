package cudakv_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

var hostCommandConfig = cudakv.HostCommandConfig{
	Mode:             "host-finite",
	Generation:       7,
	VocabSize:        9,
	MaxEntries:       2,
	MaxRegistryWords: 4,
	MaxBatch:         3,
	MaxBatchWords:    4,
}

func hostCommandFixture(t *testing.T, capacity uint64) (*cudakv.DeviceCommandRing, *cudakv.HostCommandConsumer) {
	t.Helper()
	ring, err := cudakv.NewDeviceCommandRing(capacity, 7)
	if err != nil {
		t.Fatal(err)
	}
	inputs := []cudakv.HostCommandInput{{SeqID: 1, Words: []uint32{10, 20}}, {SeqID: 2, Words: []uint32{30, 40}}}
	c, err := cudakv.NewHostCommandConsumer(ring, hostCommandConfig, inputs)
	if err != nil {
		t.Fatal(err)
	}
	return ring, c
}

func publishHostCommands(t *testing.T, r *cudakv.DeviceCommandRing, descs ...cudakv.Descriptor) {
	t.Helper()
	for _, d := range descs {
		d.Generation = 7
		if err := r.Publish(d); err != nil {
			t.Fatal(err)
		}
	}
}

// hostCommandRecorder records each call as the first input word plus the token.
type hostCommandRecorder struct {
	got []uint32
}

func (r *hostCommandRecorder) run(_ context.Context, d cudakv.Descriptor, w []uint32) error {
	r.got = append(r.got, w[0]+uint32(d.PromptToken))
	w[0] = 999 // must not reach the registry
	return nil
}

func TestHostCommandConsumerFIFO(t *testing.T) {
	ring, c := hostCommandFixture(t, 3)
	publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1, PromptToken: 2}, cudakv.Descriptor{SeqID: 2, PromptToken: 3})
	var rec hostCommandRecorder
	b, err := c.ConsumeBatch(context.Background(), 1, rec.run)
	if err != nil || b != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 2}) || ring.Len() != 1 {
		t.Fatalf("first batch = %+v, %v; ring %d", b, err, ring.Len())
	}
	// Wrap the ring and drain past the end of its storage.
	publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1, PromptToken: 4}, cudakv.Descriptor{SeqID: 2, PromptToken: 0})
	ring.Close()
	if err := ring.Publish(cudakv.Descriptor{SeqID: 1, Generation: 7}); err == nil {
		t.Fatal("Publish after Close accepted")
	}
	b, err = c.ConsumeBatch(context.Background(), 3, rec.run)
	if err != nil || b != (cudakv.HostCommandBatch{Consumed: 2, Completed: 2, Words: 4}) || ring.Len() != 1 {
		t.Fatalf("second batch = %+v, %v; ring %d", b, err, ring.Len())
	}
	b, err = c.ConsumeBatch(context.Background(), 3, rec.run)
	if err != nil || b != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 2}) || ring.Len() != 0 {
		t.Fatalf("third batch = %+v, %v; ring %d", b, err, ring.Len())
	}
	if want := []uint32{12, 33, 14, 30}; !reflect.DeepEqual(rec.got, want) {
		t.Fatalf("calls = %v, want %v", rec.got, want)
	}
	b, err = c.ConsumeBatch(context.Background(), 3, rec.run)
	if err != nil || b != (cudakv.HostCommandBatch{}) {
		t.Fatalf("empty batch = %+v, %v", b, err)
	}
}

func TestHostCommandConsumerWordBudgetLeavesQueued(t *testing.T) {
	ring, c := hostCommandFixture(t, 3)
	publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1}, cudakv.Descriptor{SeqID: 2}, cudakv.Descriptor{SeqID: 1})
	var rec hostCommandRecorder
	b, err := c.ConsumeBatch(context.Background(), 3, rec.run)
	if err != nil || b != (cudakv.HostCommandBatch{Consumed: 2, Completed: 2, Words: 4}) || ring.Len() != 1 {
		t.Fatalf("budgeted batch = %+v, %v; ring %d", b, err, ring.Len())
	}
	b, err = c.ConsumeBatch(context.Background(), 3, rec.run)
	if err != nil || b.Completed != 1 || ring.Len() != 0 {
		t.Fatalf("next batch = %+v, %v; ring %d", b, err, ring.Len())
	}
	if want := []uint32{10, 30, 10}; !reflect.DeepEqual(rec.got, want) {
		t.Fatalf("calls = %v, want %v", rec.got, want)
	}
}

func TestNewHostCommandConsumerRefuses(t *testing.T) {
	one := []cudakv.HostCommandInput{{SeqID: 1, Words: []uint32{1, 2}}}
	tests := []struct {
		name        string
		change      func(*cudakv.HostCommandConfig)
		inputs      []cudakv.HostCommandInput
		unsupported bool
	}{
		{name: "empty mode", change: func(c *cudakv.HostCommandConfig) { c.Mode = "" }, unsupported: true},
		{name: "device poll", change: func(c *cudakv.HostCommandConfig) { c.Mode = "device-poll" }, unsupported: true},
		{name: "GIN", change: func(c *cudakv.HostCommandConfig) { c.Mode = "gin" }, unsupported: true},
		{name: "decode", change: func(c *cudakv.HostCommandConfig) { c.Mode = "decode" }, unsupported: true},
		{name: "generation", change: func(c *cudakv.HostCommandConfig) { c.Generation = 8 }},
		{name: "zero generation", change: func(c *cudakv.HostCommandConfig) { c.Generation = 0 }},
		{name: "zero vocab", change: func(c *cudakv.HostCommandConfig) { c.VocabSize = 0 }},
		{name: "zero entries", change: func(c *cudakv.HostCommandConfig) { c.MaxEntries = 0 }},
		{name: "unbounded entries", change: func(c *cudakv.HostCommandConfig) { c.MaxEntries = 1025 }},
		{name: "zero batch", change: func(c *cudakv.HostCommandConfig) { c.MaxBatch = 0 }},
		{name: "unbounded batch", change: func(c *cudakv.HostCommandConfig) { c.MaxBatch = 1025 }},
		{name: "unbounded registry words", change: func(c *cudakv.HostCommandConfig) { c.MaxRegistryWords = 1<<20 + 1 }},
		{name: "unbounded batch words", change: func(c *cudakv.HostCommandConfig) { c.MaxBatchWords = 1<<20 + 1 }},
		{name: "registry budget", change: func(c *cudakv.HostCommandConfig) { c.MaxRegistryWords = 1 }},
		{name: "entry exceeds batch budget", change: func(c *cudakv.HostCommandConfig) { c.MaxBatchWords = 1 }},
		{name: "too many entries", change: func(c *cudakv.HostCommandConfig) { c.MaxEntries = 1 }, inputs: []cudakv.HostCommandInput{{SeqID: 1, Words: []uint32{1}}, {SeqID: 2, Words: []uint32{2}}}},
		{name: "no entries", inputs: []cudakv.HostCommandInput{}},
		{name: "zero sequence", inputs: []cudakv.HostCommandInput{{SeqID: 0, Words: []uint32{1}}}},
		{name: "empty words", inputs: []cudakv.HostCommandInput{{SeqID: 1}}},
		{name: "duplicate", inputs: []cudakv.HostCommandInput{{SeqID: 1, Words: []uint32{1}}, {SeqID: 1, Words: []uint32{2}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ring, err := cudakv.NewDeviceCommandRing(2, 7)
			if err != nil {
				t.Fatal(err)
			}
			cfg := hostCommandConfig
			if tt.change != nil {
				tt.change(&cfg)
			}
			inputs := tt.inputs
			if inputs == nil {
				inputs = one
			}
			c, err := cudakv.NewHostCommandConsumer(ring, cfg, inputs)
			if err == nil || c != nil {
				t.Fatalf("NewHostCommandConsumer = %v, %v; want refusal", c, err)
			}
			if got := errors.Is(err, cudakv.ErrHostCommandUnsupported); got != tt.unsupported {
				t.Fatalf("errors.Is(%v, ErrHostCommandUnsupported) = %v, want %v", err, got, tt.unsupported)
			}
		})
	}
	if _, err := cudakv.NewHostCommandConsumer(nil, hostCommandConfig, one); err == nil {
		t.Fatal("nil ring admitted")
	}
	if _, err := cudakv.NewHostCommandConsumer(new(cudakv.DeviceCommandRing), hostCommandConfig, one); err == nil {
		t.Fatal("zero-value ring admitted")
	}
}

func TestHostCommandConsumerCopiesRegistry(t *testing.T) {
	ring, err := cudakv.NewDeviceCommandRing(2, 7)
	if err != nil {
		t.Fatal(err)
	}
	w := []uint32{42}
	c, err := cudakv.NewHostCommandConsumer(ring, hostCommandConfig, []cudakv.HostCommandInput{{SeqID: 1, Words: w}})
	if err != nil {
		t.Fatal(err)
	}
	w[0] = 99
	publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1}, cudakv.Descriptor{SeqID: 1})
	var rec hostCommandRecorder // overwrites each copy it receives
	if _, err := c.ConsumeBatch(context.Background(), 2, rec.run); err != nil {
		t.Fatal(err)
	}
	if want := []uint32{42, 42}; !reflect.DeepEqual(rec.got, want) {
		t.Fatalf("calls = %v, want %v", rec.got, want)
	}
}

func TestHostCommandConsumerRefusesDescriptor(t *testing.T) {
	tests := []struct {
		name string
		desc cudakv.Descriptor
	}{
		{"unknown sequence", cudakv.Descriptor{SeqID: 9}},
		{"zero sequence", cudakv.Descriptor{SeqID: 0}},
		{"flags", cudakv.Descriptor{SeqID: 1, Flags: 1}},
		{"negative token", cudakv.Descriptor{SeqID: 1, PromptToken: -1}},
		{"token at vocab size", cudakv.Descriptor{SeqID: 1, PromptToken: 9}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ring, c := hostCommandFixture(t, 2)
			publishHostCommands(t, ring, tt.desc, cudakv.Descriptor{SeqID: 2, PromptToken: 8})
			var rec hostCommandRecorder
			b, err := c.ConsumeBatch(context.Background(), 2, rec.run)
			if !errors.Is(err, cudakv.ErrHostCommandInvalid) || b != (cudakv.HostCommandBatch{Consumed: 1}) || len(rec.got) != 0 || ring.Len() != 1 {
				t.Fatalf("batch = %+v, %v, calls %v, ring %d; want refusal with later command queued", b, err, rec.got, ring.Len())
			}
			if _, err := c.ConsumeBatch(context.Background(), 2, rec.run); err != nil || !reflect.DeepEqual(rec.got, []uint32{38}) {
				t.Fatalf("later command: %v, calls %v", err, rec.got)
			}
		})
	}
}

func TestHostCommandConsumerCallbackError(t *testing.T) {
	ring, c := hostCommandFixture(t, 2)
	publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1}, cudakv.Descriptor{SeqID: 2})
	sentinel := errors.New("caller failure")
	calls := 0
	b, err := c.ConsumeBatch(context.Background(), 2, func(context.Context, cudakv.Descriptor, []uint32) error {
		calls++
		return sentinel
	})
	if !errors.Is(err, sentinel) || calls != 1 || b != (cudakv.HostCommandBatch{Consumed: 1, Words: 2}) || ring.Len() != 1 {
		t.Fatalf("batch = %+v, %v, calls %d, ring %d", b, err, calls, ring.Len())
	}
}

func TestHostCommandConsumerCancellation(t *testing.T) {
	t.Run("before removal", func(t *testing.T) {
		ring, c := hostCommandFixture(t, 2)
		publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1}, cudakv.Descriptor{SeqID: 2})
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		var rec hostCommandRecorder
		b, err := c.ConsumeBatch(ctx, 2, rec.run)
		if !errors.Is(err, context.Canceled) || b != (cudakv.HostCommandBatch{}) || len(rec.got) != 0 || ring.Len() != 2 {
			t.Fatalf("batch = %+v, %v, calls %v, ring %d", b, err, rec.got, ring.Len())
		}
	})
	t.Run("during callback", func(t *testing.T) {
		ring, c := hostCommandFixture(t, 2)
		publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1}, cudakv.Descriptor{SeqID: 2})
		ctx, cancel := context.WithCancel(context.Background())
		b, err := c.ConsumeBatch(ctx, 2, func(context.Context, cudakv.Descriptor, []uint32) error {
			cancel() // the callback's effects stand; the next command waits
			return nil
		})
		if !errors.Is(err, context.Canceled) || b != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 2}) || ring.Len() != 1 {
			t.Fatalf("batch = %+v, %v, ring %d", b, err, ring.Len())
		}
	})
	t.Run("callback honors cancellation", func(t *testing.T) {
		ring, c := hostCommandFixture(t, 2)
		publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1})
		ctx, cancel := context.WithCancel(context.Background())
		b, err := c.ConsumeBatch(ctx, 1, func(ctx context.Context, _ cudakv.Descriptor, _ []uint32) error {
			cancel()
			return ctx.Err()
		})
		if !errors.Is(err, context.Canceled) || b != (cudakv.HostCommandBatch{Consumed: 1, Words: 2}) || ring.Len() != 0 {
			t.Fatalf("batch = %+v, %v, ring %d; want consumed, not completed", b, err, ring.Len())
		}
	})
}

func TestHostCommandConsumerConcurrentBatchRefused(t *testing.T) {
	ring, c := hostCommandFixture(t, 2)
	publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1}, cudakv.Descriptor{SeqID: 2})
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		_, err := c.ConsumeBatch(context.Background(), 1, func(context.Context, cudakv.Descriptor, []uint32) error {
			close(entered)
			<-release
			return nil
		})
		done <- err
	}()
	<-entered
	b, err := c.ConsumeBatch(context.Background(), 1, func(context.Context, cudakv.Descriptor, []uint32) error {
		t.Error("concurrent callback ran")
		return nil
	})
	if !errors.Is(err, cudakv.ErrHostCommandBusy) || b != (cudakv.HostCommandBatch{}) || ring.Len() != 1 {
		t.Errorf("concurrent batch = %+v, %v, ring %d", b, err, ring.Len())
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestHostCommandConsumerBatchRefuses(t *testing.T) {
	ring, c := hostCommandFixture(t, 2)
	publishHostCommands(t, ring, cudakv.Descriptor{SeqID: 1})
	var rec hostCommandRecorder
	tests := []struct {
		name  string
		c     *cudakv.HostCommandConsumer
		ctx   context.Context
		limit int
		run   func(context.Context, cudakv.Descriptor, []uint32) error
	}{
		{"zero limit", c, context.Background(), 0, rec.run},
		{"limit above MaxBatch", c, context.Background(), 4, rec.run},
		{"nil context", c, nil, 1, rec.run},
		{"nil function", c, context.Background(), 1, nil},
		{"nil consumer", nil, context.Background(), 1, rec.run},
		{"zero consumer", new(cudakv.HostCommandConsumer), context.Background(), 1, rec.run},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := tt.c.ConsumeBatch(tt.ctx, tt.limit, tt.run)
			if err == nil || b != (cudakv.HostCommandBatch{}) || ring.Len() != 1 || len(rec.got) != 0 {
				t.Fatalf("batch = %+v, %v, ring %d; want refusal before removal", b, err, ring.Len())
			}
		})
	}
}
