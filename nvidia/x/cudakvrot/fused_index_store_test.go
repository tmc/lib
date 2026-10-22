package cudakvrot_test

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudakvrot"
)

func TestStoreIndexKVPagedAndCompile(t *testing.T) {
	// Test both FP4 (68B/tok: 32B data at 0..32, 4B scale at 64..68) and MXFP8 (132B/tok: 128B data at 0..128, 4B scale at 128..132).
	t.Run("FP4", func(t *testing.T) {
		spec := cudakvrot.FusedIndexKVStoreSpec{
			HeadDim:       64,
			PageBlockSize: 4,
			UseFP4:        true,
		}
		if err := spec.Validate(); err != nil {
			t.Fatalf("Validate() unexpected error: %v", err)
		}

		// 2 tokens: token 0 at pos=2 (block 0, slot 2 -> phys page 1), token 1 at pos=5 (block 1, slot 1 -> phys page 2).
		kQuant := make([]byte, 2*32)
		kScale := make([]byte, 2*4)
		for i := range kQuant {
			kQuant[i] = byte(i + 1)
		}
		for i := range kScale {
			kScale[i] = byte(0xD0 + i)
		}
		pageTable := []int32{
			1, 0, // seq 0: block 0 -> page 1
			0, 2, // seq 1: block 1 -> page 2
		}
		seqPositions := []int32{2, 5}
		pageBytes := 4 * 68
		pagePool := bytes.Repeat([]byte{0xA5}, 3*pageBytes)

		if err := cudakvrot.StoreIndexKVPagedCPU(kQuant, kScale, pageTable, seqPositions, pagePool, spec); err != nil {
			t.Fatalf("StoreIndexKVPagedCPU() unexpected error: %v", err)
		}

		// Check token 0 at page 1, tokenInPage 2.
		slot0Base := 1*pageBytes + 2*68
		if !bytes.Equal(pagePool[slot0Base:slot0Base+32], kQuant[0:32]) {
			t.Fatalf("token 0 data mismatch: got %x, want %x", pagePool[slot0Base:slot0Base+32], kQuant[0:32])
		}
		// Padding region [32..64) must remain canary 0xA5.
		for i := 32; i < 64; i++ {
			if pagePool[slot0Base+i] != 0xA5 {
				t.Fatalf("token 0 padding byte %d modified: got 0x%x, want 0xA5", i, pagePool[slot0Base+i])
			}
		}
		if !bytes.Equal(pagePool[slot0Base+64:slot0Base+68], kScale[0:4]) {
			t.Fatalf("token 0 scale mismatch: got %x, want %x", pagePool[slot0Base+64:slot0Base+68], kScale[0:4])
		}

		// Check token 1 at page 2, tokenInPage 1.
		slot1Base := 2*pageBytes + 1*68
		if !bytes.Equal(pagePool[slot1Base:slot1Base+32], kQuant[32:64]) {
			t.Fatalf("token 1 data mismatch")
		}
		if !bytes.Equal(pagePool[slot1Base+64:slot1Base+68], kScale[4:8]) {
			t.Fatalf("token 1 scale mismatch")
		}

		src, err := cudakvrot.CompileFusedIndexKVStore(spec)
		if err != nil {
			t.Fatalf("CompileFusedIndexKVStore() unexpected error: %v", err)
		}
		if !strings.Contains(src, "set_index_kv_paged_kernel") || !strings.Contains(src, "dst_slot[64 + s]") {
			t.Fatalf("CompileFusedIndexKVStore() missing expected kernel symbols:\n%s", src)
		}
	})

	t.Run("MXFP8", func(t *testing.T) {
		spec := cudakvrot.FusedIndexKVStoreSpec{
			HeadDim:       128,
			PageBlockSize: 4,
			UseFP4:        false,
		}
		kQuant := bytes.Repeat([]byte{0x42}, 128)
		kScale := []byte{0x7F, 0x80, 0x81, 0x82}
		pageTable := []int32{0}
		seqPositions := []int32{3}
		pagePool := make([]byte, 4*132)

		if err := cudakvrot.StoreIndexKVPagedCPU(kQuant, kScale, pageTable, seqPositions, pagePool, spec); err != nil {
			t.Fatalf("StoreIndexKVPagedCPU(MXFP8) unexpected error: %v", err)
		}
		slotBase := 3 * 132
		if !bytes.Equal(pagePool[slotBase:slotBase+128], kQuant) {
			t.Fatalf("MXFP8 data mismatch")
		}
		if !bytes.Equal(pagePool[slotBase+128:slotBase+132], kScale) {
			t.Fatalf("MXFP8 scale mismatch")
		}
	})

	// Negative controls.
	badSpec := cudakvrot.FusedIndexKVStoreSpec{HeadDim: 96, PageBlockSize: 64}
	if _, err := cudakvrot.CompileFusedIndexKVStore(badSpec); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("invalid HeadDim error = %v, want ErrUnsupported", err)
	}
	validSpec := cudakvrot.FusedIndexKVStoreSpec{HeadDim: 64, PageBlockSize: 4, UseFP4: true}
	if err := cudakvrot.StoreIndexKVPagedCPU(make([]byte, 32), make([]byte, 4), []int32{5}, []int32{0}, make([]byte, 4*68), validSpec); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("out-of-bounds physical page error = %v, want ErrUnsupported", err)
	}
}
