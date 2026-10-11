package cudatma

import (
	"fmt"
	"reflect"
	"testing"
)

func TestNewMulticastRankBounds(t *testing.T) {
	for _, source := range []uint32{0, 15} {
		t.Run(fmt.Sprintf("source_%d", source), func(t *testing.T) {
			m, err := NewMulticast(source, []uint32{15, 0}, 64)
			if err != nil {
				t.Fatal(err)
			}
			if m.SourceRank != source || m.Mask != 1|1<<15 || m.Bytes != 64 ||
				!reflect.DeepEqual(m.Destinations, []uint32{0, 15}) {
				t.Fatalf("unexpected multicast: %+v", m)
			}
			for _, rank := range []uint32{0, 15} {
				if !m.Participating(rank) {
					t.Fatalf("rank %d is not participating", rank)
				}
			}
			if m.Participating(1) || m.CompleteAll() {
				t.Fatal("non-destination participates or transactions are already complete")
			}
			for _, rank := range []uint32{16, 63, 64, ^uint32(0)} {
				if m.Participating(rank) {
					t.Fatalf("invalid rank %d participates", rank)
				}
				if err := m.Complete(rank, 64); err == nil {
					t.Fatalf("invalid rank %d completed", rank)
				}
			}
			if err := m.Complete(0, 64); err != nil {
				t.Fatal(err)
			}
			if m.CompleteAll() {
				t.Fatal("completed with rank 15 still pending")
			}
			if err := m.Complete(15, 64); err != nil {
				t.Fatal(err)
			}
			if !m.CompleteAll() {
				t.Fatal("both destination transactions completed but multicast remains incomplete")
			}
		})
	}
	for _, rank := range []uint32{16, 63, 64, ^uint32(0)} {
		t.Run(fmt.Sprintf("source_%d", rank), func(t *testing.T) {
			m, err := NewMulticast(rank, []uint32{0, 15}, 64)
			if err == nil {
				t.Fatalf("accepted source rank %d", rank)
			}
			if !reflect.DeepEqual(m, Multicast{}) {
				t.Fatalf("invalid source returned a partial multicast: %+v", m)
			}
		})
		for _, destinations := range [][]uint32{{rank, 0}, {0, rank}} {
			t.Run(fmt.Sprintf("destinations_%v", destinations), func(t *testing.T) {
				m, err := NewMulticast(0, destinations, 64)
				if err == nil {
					t.Fatalf("accepted destination rank %d", rank)
				}
				if !reflect.DeepEqual(m, Multicast{}) {
					t.Fatalf("invalid destination returned a partial multicast: %+v", m)
				}
			})
		}
	}
}
