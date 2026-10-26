//go:build !windows

package nvlshm

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func makeTestFabricHandle(rank int) FabricHandle {
	var fh FabricHandle
	pl, err := PlacementForRank(rank % NVL72TotalGPUs)
	if err == nil {
		fh[0] = byte(pl.TrayID)
		fh[1] = byte(pl.TrayLocalGPU)
		fh[2] = byte(pl.GraceSocket)
		fh[3] = byte(pl.GlobalGraceID)
	}
	fh[4] = byte(rank)
	fh[5] = byte(rank >> 8)
	for i := 6; i < FabricHandleSize; i++ {
		fh[i] = byte((rank*31 + i*17 + 0x5A) & 0xFF)
	}
	return fh
}

func TestFabricHandleEncodeDecode(t *testing.T) {
	for _, rank := range []int{0, 1, 7, 15, 35, 71} {
		orig := makeTestFabricHandle(rank)
		enc := orig.String()
		if len(enc) != FabricHandleSize*2 {
			t.Fatalf("rank %d: encoded length = %d, want %d", rank, len(enc), FabricHandleSize*2)
		}
		dec, err := ParseFabricHandle(enc)
		if err != nil {
			t.Fatalf("rank %d: ParseFabricHandle(%q) unexpected error: %v", rank, enc, err)
		}
		if dec != orig {
			t.Fatalf("rank %d: decoded handle != original", rank)
		}
	}

	badInputs := []struct {
		name  string
		input string
	}{
		{"empty", ""},
		{"short", "deadbeef"},
		{"long", strings.Repeat("ab", FabricHandleSize+1)},
		{"non-hex", strings.Repeat("zz", FabricHandleSize)},
	}
	for _, tc := range badInputs {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ParseFabricHandle(tc.input); err == nil {
				t.Fatalf("ParseFabricHandle(%q) succeeded, want error", tc.input)
			}
		})
	}
}

func TestFabricExchange8Ranks(t *testing.T) {
	const world = 8
	runRanks(t, world, func(rank int, b Bootstrap) error {
		local := makeTestFabricHandle(rank)
		got, err := ExchangeFabricHandles(b, rank, world, local)
		if err != nil {
			return fmt.Errorf("rank %d: ExchangeFabricHandles: %w", rank, err)
		}
		if len(got) != world {
			return fmt.Errorf("rank %d: got %d handles, want %d", rank, len(got), world)
		}
		for r := 0; r < world; r++ {
			want := makeTestFabricHandle(r)
			if got[r] != want {
				return fmt.Errorf("rank %d: handle[%d] mismatch: got %x, want %x", rank, r, got[r], want)
			}
		}
		return nil
	})
}

func TestFabricExchange72RanksNVL72(t *testing.T) {
	const world = NVL72TotalGPUs // 72 GPUs across 18 compute trays
	runRanks(t, world, func(rank int, b Bootstrap) error {
		local := makeTestFabricHandle(rank)
		got, err := ExchangeFabricHandles(b, rank, world, local)
		if err != nil {
			return fmt.Errorf("rank %d: ExchangeFabricHandles: %w", rank, err)
		}
		if len(got) != world {
			return fmt.Errorf("rank %d: got %d handles, want %d", rank, len(got), world)
		}
		seen := make(map[FabricHandle]int, world)
		for r := 0; r < world; r++ {
			want := makeTestFabricHandle(r)
			if got[r] != want {
				return fmt.Errorf("rank %d: handle[%d] mismatch", rank, r)
			}
			if prev, dup := seen[got[r]]; dup {
				return fmt.Errorf("rank %d: duplicate handle at ranks %d and %d", rank, prev, r)
			}
			seen[got[r]] = r
			if bytes.Equal(got[r][:], make([]byte, FabricHandleSize)) {
				return fmt.Errorf("rank %d: handle[%d] is all zeros", rank, r)
			}
		}
		return nil
	})
}

type stubBootstrap struct {
	rank          int
	world         int
	broadcastResp string
	broadcastErr  error
	barrierErr    error
}

func (s *stubBootstrap) Rank() int      { return s.rank }
func (s *stubBootstrap) WorldSize() int { return s.world }
func (s *stubBootstrap) BroadcastString(v string, root int) (string, error) {
	if s.broadcastErr != nil {
		return "", s.broadcastErr
	}
	if s.broadcastResp != "" {
		return s.broadcastResp, nil
	}
	return v, nil
}
func (s *stubBootstrap) Barrier() error {
	return s.barrierErr
}

func TestFabricExchangeValidationAndMalformed(t *testing.T) {
	fh := makeTestFabricHandle(0)

	if _, err := ExchangeFabricHandles(nil, 0, 1, fh); err == nil {
		t.Fatal("nil bootstrap: want error")
	}

	boots := newLocalBoots(2)
	invalidArgs := []struct {
		name  string
		rank  int
		world int
	}{
		{"negative rank", -1, 2},
		{"rank >= world", 2, 2},
		{"zero world", 0, 0},
		{"negative world", 0, -1},
		{"rank mismatch", 1, 2},
		{"world mismatch", 0, 4},
	}
	for _, tc := range invalidArgs {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ExchangeFabricHandles(boots[0], tc.rank, tc.world, fh); err == nil {
				t.Fatalf("ExchangeFabricHandles(%d, %d) succeeded, want error", tc.rank, tc.world)
			}
		})
	}

	t.Run("malformed payload from peer", func(t *testing.T) {
		sb := &stubBootstrap{
			rank:          0,
			world:         1,
			broadcastResp: "not-a-valid-64-byte-hex-fabric-handle",
		}
		if _, err := ExchangeFabricHandles(sb, 0, 1, fh); err == nil {
			t.Fatal("malformed broadcast payload: want error")
		}
	})

	t.Run("broadcast error", func(t *testing.T) {
		sb := &stubBootstrap{
			rank:         0,
			world:        1,
			broadcastErr: errors.New("network timeout"),
		}
		if _, err := ExchangeFabricHandles(sb, 0, 1, fh); err == nil {
			t.Fatal("broadcast error: want error")
		}
	})

	t.Run("barrier error", func(t *testing.T) {
		sb := &stubBootstrap{
			rank:       0,
			world:      1,
			barrierErr: errors.New("barrier broken"),
		}
		if _, err := ExchangeFabricHandles(sb, 0, 1, fh); err == nil {
			t.Fatal("barrier error: want error")
		}
	})

	t.Run("DistAllocFabric validation", func(t *testing.T) {
		if _, err := DistAllocFabric(nil, 0, 1, 0, 4096); err == nil {
			t.Fatal("DistAllocFabric(nil) succeeded, want error")
		}
		if _, err := DistAllocFabric(boots[0], 1, 2, 0, 4096); err == nil {
			t.Fatal("DistAllocFabric with mismatched rank succeeded, want error")
		}
		if _, err := DistAllocFabricMulticast(boots[0], 0, 2, 0, ^uintptr(0)); err == nil {
			t.Fatal("DistAllocFabricMulticast with overflowing size succeeded, want error")
		}
	})
}
