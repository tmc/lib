package cudasearch

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/nvvm"
)

type outputFixture struct {
	prefill                      *prefillExecutable
	sequence                     *sequenceExecutable
	sizes                        []int
	runs, drains, reads, decodes int
	failRead, failDecode         int
	cancel                       context.CancelFunc
}

var outputReadFailure = errors.New("injected output read failure")
var outputDecodeFailure = errors.New("injected output decode failure")

func newOutputFixture(kind string) *outputFixture {
	f := &outputFixture{failRead: -1, failDecode: -1}
	if kind == "prefill" {
		f.prefill = &prefillExecutable{checkLease: func() bool { return true }}
		f.sizes = []int{4}
		return f
	}
	f.sequence = &sequenceExecutable{checkLease: func() bool { return true }, outputs: []sequenceOutput{{name: "float", buffer: "a"}}}
	f.sizes = []int{8}
	if kind == "bytes" {
		f.sequence.outputs[0].bytes = true
		f.sizes[0] = 3
	}
	if kind == "multiple" {
		f.sequence.outputs = append(f.sequence.outputs, sequenceOutput{name: "bytes", buffer: "b", bytes: true})
		f.sizes = append(f.sizes, 3)
	}
	return f
}

func (f *outputFixture) ops() outputReadOps {
	return outputReadOps{
		run: func(ctx context.Context) error {
			poison := func() error { f.runs++; return nil }
			wait := func(ctx context.Context) error { f.drains++; return ctx.Err() }
			if f.prefill != nil {
				_, err := f.prefill.runWith(ctx, false, prefillRunOps{poison: poison, launch: func() error { return nil }, wait: wait})
				return err
			}
			_, err := f.sequence.runWith(ctx, false, sequenceRunOps{poison: poison, wait: wait})
			return err
		},
		length: func(i int) (int, error) { return f.sizes[i], nil },
		readFloats: func(i int, dst []float32) error {
			f.reads++
			if f.drains == 0 {
				return errors.New("read before drain")
			}
			if i == f.failRead {
				return outputReadFailure
			}
			for j := range dst {
				dst[j] = float32(j + 1)
			}
			if f.cancel != nil {
				f.cancel()
			}
			return nil
		},
		readBytes: func(i int, dst []byte) error {
			f.reads++
			if f.drains == 0 {
				return errors.New("read before drain")
			}
			if i == f.failRead {
				return outputReadFailure
			}
			if f.sequence.outputs[i].bytes {
				copy(dst, []byte{7, 8, 9})
			} else {
				binary.LittleEndian.PutUint32(dst, math.Float32bits(1.25))
				binary.LittleEndian.PutUint32(dst[4:], math.Float32bits(-2.5))
			}
			if f.cancel != nil {
				f.cancel()
			}
			return nil
		},
		decode: func(i int, src []byte) ([]float32, error) {
			f.decodes++
			if i == f.failDecode {
				return nil, outputDecodeFailure
			}
			return f.sequence.outputs[i].decode(src)
		},
	}
}

func (f *outputFixture) owned(ctx context.Context) (map[string][]float32, io.Closer, error) {
	if f.prefill != nil {
		return f.prefill.outputsOwnedWith(ctx, f.ops())
	}
	return f.sequence.outputsOwnedWith(ctx, f.ops())
}

func (f *outputFixture) legacy(ctx context.Context) (map[string][]float32, error) {
	if f.prefill != nil {
		return f.prefill.outputsLegacyWith(ctx, f.ops())
	}
	return f.sequence.outputsLegacyWith(ctx, f.ops())
}

func (f *outputFixture) close(closeResources func() error) error {
	if f.prefill != nil {
		return f.prefill.closeWith(closeResources)
	}
	return f.sequence.closeWith(closeResources)
}

func outputMemoryContext(limit int64) (context.Context, *memoryBudget) {
	b := &memoryBudget{limit: limit}
	return context.WithValue(context.Background(), memoryBudgetKey{}, b), b
}

func checkOutputLibraries(t *testing.T) {
	t.Helper()
	check := func() {
		if driver.Loaded() || nvrtc.Loaded() || nvvm.Loaded() {
			t.Errorf("output host control loaded native libraries")
		}
	}
	check()
	t.Cleanup(check)
}

func TestBuiltInOutputQuotaAndOwnership(t *testing.T) {
	checkOutputLibraries(t)
	for _, kind := range []string{"prefill", "float", "bytes", "multiple"} {
		t.Run(kind, func(t *testing.T) {
			ctx, b := outputMemoryContext(1 << 20)
			f := newOutputFixture(kind)
			values, owner, err := f.owned(ctx)
			if err != nil || owner == nil || b.live <= 0 || b.peak <= b.live {
				t.Fatalf("owned outputs = %v, %v; live=%d peak=%d", values, err, b.live, b.peak)
			}
			peak := b.peak
			want := map[string][]float32{"float": {1.25, -2.5}}
			if kind == "prefill" {
				want = map[string][]float32{"out": {1, 2, 3, 4}}
			} else if kind == "bytes" {
				want = map[string][]float32{"float": {7, 8, 9}}
			} else if kind == "multiple" {
				want["bytes"] = []float32{7, 8, 9}
			}
			if !reflect.DeepEqual(values, want) {
				t.Fatalf("readback = %v, want %v", values, want)
			}
			owner.Close()
			owner.Close()
			if b.live != 0 {
				t.Fatalf("per-call owner retained %d bytes", b.live)
			}
			ctx, b = outputMemoryContext(peak - 1)
			f = newOutputFixture(kind)
			if _, owner, err := f.owned(ctx); !errors.Is(err, ErrMemoryBudgetExceeded) || owner != nil {
				t.Fatalf("one-short outputs = %v, owner=%v", err, owner)
			}
			if f.runs != 0 || f.reads != 0 || f.decodes != 0 || b.live != 0 {
				t.Fatalf("one-short ran=%d read=%d decoded=%d retained=%d", f.runs, f.reads, f.decodes, b.live)
			}
			ctx, b = outputMemoryContext(peak)
			f = newOutputFixture(kind)
			for range 3 {
				_, owner, err := f.owned(ctx)
				if err != nil || b.peak != peak {
					t.Fatalf("exact quota = %v, peak=%d want %d", err, b.peak, peak)
				}
				owner.Close()
				if b.live != 0 {
					t.Fatalf("borrowed executable retained per-call bytes: %d", b.live)
				}
			}
			if f.prefill != nil && (f.prefill.closed || f.prefill.outputMemory != nil) || f.sequence != nil && (f.sequence.closed || f.sequence.outputMemory != nil) {
				t.Fatal("per-call cleanup closed executable or retained legacy owner")
			}
		})
	}
}

func TestBuiltInOutputFailureUnwind(t *testing.T) {
	checkOutputLibraries(t)
	for _, tc := range []struct {
		name                     string
		read, decode             int
		cancelBefore, cancelRead bool
		want                     error
	}{
		{"later_read", 1, -1, false, false, outputReadFailure},
		{"later_decode", -1, 1, false, false, outputDecodeFailure},
		{"cancel_before", -1, -1, true, false, context.Canceled},
		{"cancel_read", -1, -1, false, true, context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, b := outputMemoryContext(1 << 20)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			f := newOutputFixture("multiple")
			f.failRead, f.failDecode = tc.read, tc.decode
			if tc.cancelBefore {
				cancel()
			}
			if tc.cancelRead {
				f.cancel = cancel
			}
			values, owner, err := f.owned(ctx)
			if !errors.Is(err, tc.want) || values != nil || owner != nil || b.live != 0 {
				t.Fatalf("failure outputs=%v owner=%v error=%v retained=%d", values, owner, err, b.live)
			}
			if tc.cancelBefore && (f.runs != 0 || f.reads != 0) || tc.cancelRead && (f.reads != 1 || f.decodes != 0) {
				t.Fatalf("cancellation ran=%d read=%d decoded=%d", f.runs, f.reads, f.decodes)
			}
			if tc.read >= 0 || tc.decode >= 0 {
				if !f.sequence.closed {
					t.Fatal("failed read/decode did not retire executable")
				}
			}
		})
	}
}

func TestBuiltInLegacyOutputCloseRetry(t *testing.T) {
	checkOutputLibraries(t)
	for _, kind := range []string{"prefill", "multiple"} {
		t.Run(kind, func(t *testing.T) {
			ctx, b := outputMemoryContext(1 << 20)
			f := newOutputFixture(kind)
			first, err := f.legacy(ctx)
			if err != nil {
				t.Fatal(err)
			}
			firstCharge := b.live
			second, err := f.legacy(ctx)
			if err != nil || b.live != 2*firstCharge || !reflect.DeepEqual(first, second) {
				t.Fatalf("legacy outputs lost live owner: live=%d first=%d err=%v", b.live, firstCharge, err)
			}
			closeErr := errors.New("resource cleanup still pending")
			if err := f.close(func() error { return closeErr }); !errors.Is(err, closeErr) || b.live != 2*firstCharge {
				t.Fatalf("failed resource Close released aliases: live=%d err=%v", b.live, err)
			}
			for range 2 {
				if err := f.close(func() error { return nil }); err != nil || b.live != 0 {
					t.Fatalf("Close retry = %v, retained=%d", err, b.live)
				}
			}
		})
	}
}

func TestBuiltInOutputMetadataAdmission(t *testing.T) {
	checkOutputLibraries(t)
	for _, kind := range []string{"prefill", "multiple"} {
		f := newOutputFixture(kind)
		f.sizes[0] = math.MaxInt
		ctx, b := outputMemoryContext(1 << 20)
		if _, _, err := f.owned(ctx); err == nil || f.runs != 0 || b.live != 0 {
			t.Fatalf("%s overflow reached run or retained memory: %v", kind, err)
		}
	}
	f := newOutputFixture("multiple")
	f.sequence.outputs[1].name = f.sequence.outputs[0].name
	ctx, b := outputMemoryContext(1 << 20)
	if _, _, err := f.owned(ctx); !errors.Is(err, ErrMemoryAccountingUnavailable) || f.runs != 0 || b.live != 0 {
		t.Fatalf("duplicate public output metadata admitted: %v", err)
	}
	// Exercise the actual public adapters without a fake driver or buffer.
	// Missing metadata and closed execution both fail before device operations.
	x := &sequenceExecutable{outputs: []sequenceOutput{{name: "out", buffer: "missing"}}}
	if _, _, err := x.outputsOwned(ctx); !errors.Is(err, ErrMemoryAccountingUnavailable) || b.live != 0 {
		t.Fatalf("missing public buffer admitted: %v", err)
	}
	p := &prefillExecutable{closed: true}
	if _, err := p.Outputs(ctx); err == nil || b.live != 0 {
		t.Fatalf("closed public prefill admitted: %v", err)
	}
}
