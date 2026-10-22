//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalGroupAdmission(t *testing.T) {
	r := &metalRuntime{ctx: 128, pos: 2}
	if err := admitMetalGroup(r, []int{3, 4}, 2); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name     string
		r        *metalRuntime
		tokens   []int
		position int
	}{
		{"nil", nil, []int{3}, 2},
		{"empty", r, nil, 2},
		{"bound", r, make([]int, 9), 2},
		{"negative-token", r, []int{-1}, 2},
		{"vocab-token", r, []int{Flash.Vocab}, 2},
		{"position", r, []int{3}, 1},
		{"context", &metalRuntime{ctx: 256, pos: 2}, []int{3}, 2},
		{"extent", &metalRuntime{ctx: 3, pos: 2}, []int{3, 4}, 2},
		{"profiling", &metalRuntime{ctx: 128, pos: 2, collectors: &profileCollectors{}}, []int{3}, 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := admitMetalGroup(test.r, test.tokens, test.position); err == nil {
				t.Fatal("invalid group admitted")
			}
		})
	}
}

func TestMetalGroupOutputCollapse(t *testing.T) {
	const rows = 8
	hc := make([]float32, rows*ds4HCWidth)
	pre := make([]float32, rows*4)
	base, scale := []float32{0.1, -0.3, 0.7, -1.2}, []float32{0.625}
	for row := range rows {
		for stream := range 4 {
			pre[row*4+stream] = float32(row*4+stream-15) * 0.17
			for col := range ds4Width {
				hc[row*ds4HCWidth+stream*ds4Width+col] = float32((row+1)*(stream+1)*(col%23-11)) * 0.03125
			}
		}
	}
	got, want := make([]float32, rows*ds4Width), make([]float32, rows*ds4Width)
	if err := collapseMetalOutputRows(hc, pre, base, scale, got, rows); err != nil {
		t.Fatal(err)
	}
	// Independent row/column traversal retains the scalar's four f32 adds.
	for row := range rows {
		for col := range ds4Width {
			for stream := range 4 {
				z := pre[row*4+stream]*scale[0] + base[stream]
				weight := float32(1/(1+math.Exp(-float64(z))) + 1e-6)
				want[row*ds4Width+col] += hc[row*ds4HCWidth+stream*ds4Width+col] * weight
			}
		}
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("output %d = %g want %g", i, got[i], want[i])
		}
	}
	mutant := make([]float32, len(got))
	wrongPre := append([]float32(nil), pre...)
	for row := 1; row < rows; row++ {
		copy(wrongPre[row*4:(row+1)*4], pre[:4])
	}
	if err := collapseMetalOutputRows(hc, wrongPre, base, scale, mutant, rows); err != nil {
		t.Fatal(err)
	}
	different := 0
	for i := range want {
		if math.Float32bits(mutant[i]) != math.Float32bits(want[i]) {
			different++
		}
	}
	if different == 0 {
		t.Fatal("row-zero mutation was invisible")
	}
	if err := collapseMetalOutputRows(hc, pre[:4], base, scale, got, rows); err == nil {
		t.Fatal("short pre admitted")
	}
}

// This bounded device fixture checks the real embedding and residual-copy
// sites with distinct rows. It is not the full transformer equivalence gate.
func TestMetalGroupEmbeddingCopies(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	arena := metalmem.NewArena(dev)
	defer arena.Free()
	weights := metalmem.AllocIn[byte](arena, "embedding", 8*ds4Width*2)
	for row := range 8 {
		for col := range ds4Width {
			bits := floatToF16Bits(float32((row+1)*(col%31-15)) * 0.125)
			binary.LittleEndian.PutUint16(weights.HostSlice()[(row*ds4Width+col)*2:], bits)
		}
	}
	tensor := boundMetalTensor{name: "embedding", typeID: 1, dimensions: []uint64{ds4Width, 8}, data: weights}
	scratch := metalmem.AllocIn[float32](arena, "scratch", ds4HCWidth)
	output := metalmem.AllocIn[float32](arena, "all-rows", 8*ds4HCWidth)
	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	for row := range 8 {
		if err := batch.EncodeEmbedding(tensor, row, scratch); err != nil {
			t.Fatal(err)
		}
		dst, _ := output.Slice(row*ds4HCWidth, (row+1)*ds4HCWidth)
		if err := batch.encodeCopy(scratch, dst); err != nil {
			t.Fatal(err)
		}
	}
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	for row := range 8 {
		for stream := range 4 {
			for col := range ds4Width {
				got := output.HostSlice()[row*ds4HCWidth+stream*ds4Width+col]
				want := float32((row+1)*(col%31-15)) * 0.125
				if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("row %d stream %d column %d = %g want %g", row, stream, col, got, want)
				}
			}
		}
	}
	if batch.Stats().Commits != 1 || batch.Stats().Waits != 1 {
		t.Fatal("embedding group did not share one submission")
	}
}

func TestMetalGroupCommandOwnership(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	arena := metalmem.NewArena(dev)
	defer arena.Free()
	weights := metalmem.AllocIn[byte](arena, "embedding", ds4Width*2)
	tensor := boundMetalTensor{name: "embedding", typeID: 1, dimensions: []uint64{ds4Width, 1}, data: weights}
	fault := errors.New("after actual encode")
	for phase := range 2 {
		t.Run(fmt.Sprintf("unsubmitted-%d", phase), func(t *testing.T) {
			x := &metalGroupedExecution{afterEncode: func(i int) error {
				if i == phase {
					return fault
				}
				return nil
			}}
			buffer, err := metalmem.Alloc[float32](dev, ds4HCWidth)
			if err != nil {
				t.Fatal(err)
			}
			x.scratch = append(x.scratch, buffer)
			for i := 0; i <= phase; i++ {
				b, err := newMetalBatch(ops)
				if err != nil {
					t.Fatal(err)
				}
				x.batches = append(x.batches, b)
				if err := b.EncodeEmbedding(tensor, 0, buffer); err != nil {
					t.Fatal(err)
				}
				ops.hc.pending = b
				if err := x.checkpoint(i); errors.Is(err, fault) {
					break
				} else if err != nil {
					t.Fatal(err)
				}
				if err := b.Wait(t.Context()); err != nil {
					t.Fatal(err)
				}
			}
			if err := x.close(); err != nil {
				t.Fatal(err)
			}
			if !x.released || buffer.HostSlice() != nil || ops.hc.pending != nil {
				t.Fatal("unsubmitted owner not cleared after encoding ended")
			}
			if x.batches[phase].done {
				t.Fatal("faulted command submitted")
			}
		})
	}
	t.Run("submitted-error-retained", func(t *testing.T) {
		buffer, err := metalmem.Alloc[float32](dev, ds4HCWidth)
		if err != nil {
			t.Fatal(err)
		}
		b, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.EncodeEmbedding(tensor, 0, buffer); err != nil {
			t.Fatal(err)
		}
		if err := b.Commit(); err != nil {
			t.Fatal(err)
		}
		x := &metalGroupedExecution{batches: []*metalBatch{b}, scratch: []metalmem.Slice[float32]{buffer}, wait: func(batch *metalBatch) error {
			return errors.Join(libmetal.WaitCommandBuffer(t.Context(), batch.buffer), fault)
		}}
		if err := x.close(); !errors.Is(err, fault) {
			t.Fatalf("cleanup error lost: %v", err)
		}
		if !metalGroupedTerminal(b) || !x.released || buffer.HostSlice() != nil {
			t.Fatal("scratch release before terminal completion")
		}
	})
	t.Run("synthetic-nonterminal-close-retry", func(t *testing.T) {
		// The device buffer really completes. Only the status classification is
		// faulted: this tests retention branches, not a hung-device recovery.
		localOps, err := newMetalPinnedOps(dev)
		if err != nil {
			t.Fatal(err)
		}
		ownArena := metalmem.NewArena(dev)
		marker := metalmem.AllocIn[float32](ownArena, "owner-marker", 1)
		marker.HostSlice()[0] = 17
		buffer, err := metalmem.Alloc[float32](dev, ds4HCWidth)
		if err != nil {
			t.Fatal(err)
		}
		b, err := newMetalBatch(localOps)
		if err != nil {
			t.Fatal(err)
		}
		if err := b.EncodeEmbedding(tensor, 0, buffer); err != nil {
			t.Fatal(err)
		}
		if err := b.Wait(t.Context()); err != nil {
			t.Fatal(err)
		}
		terminal := false
		x := &metalGroupedExecution{batches: []*metalBatch{b}, scratch: []metalmem.Slice[float32]{buffer}, terminal: func(*metalBatch) bool { return terminal }}
		r := &metalRuntime{dev: dev, ops: localOps, arena: ownArena, borrowed: true}
		if err := x.finish(r, nil); err == nil {
			t.Fatal("nonterminal admission passed")
		}
		if x.released || r.groupedPending != x || !errors.Is(r.failed, dsv4.ErrDesynchronized) || buffer.HostSlice() == nil || marker.HostSlice()[0] != 17 {
			t.Fatal("nonterminal owner released or not poisoned")
		}
		called := false
		if err := synchronizeMetalState(r, func() error { called = true; return nil }); !errors.Is(err, dsv4.ErrDesynchronized) || called {
			t.Fatal("live grouped owner bypassed synchronization guard")
		}
		a := &metalStateAdapter{runtime: r}
		s := &metalRequestState{owner: a, runtime: r}
		if err := s.Restore(&metalSnapshot{owner: a}); !errors.Is(err, dsv4.ErrDesynchronized) {
			t.Fatalf("live owner restore not refused: %v", err)
		}
		if err := r.Close(); err == nil || r.arena != ownArena || buffer.HostSlice() == nil {
			t.Fatal("close released nonterminal owner")
		}
		terminal = true
		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
		if !x.released || r.groupedPending != nil || r.arena != nil || buffer.HostSlice() != nil || marker.HostSlice() != nil {
			t.Fatal("terminal close did not release all owners")
		}
		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
	})
}
