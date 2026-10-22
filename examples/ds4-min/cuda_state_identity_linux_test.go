//go:build linux

package main

import (
	"errors"
	"os"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/gguf"
)

func TestCUDAStateIdentityCache(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "model")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.Write([]byte("weights")); err != nil {
		t.Fatal(err)
	}
	owner := cudaFoundation{file: f, stateIdentity: new(modelStateIdentityCache), gguf: &gguf.File{Tensors: []gguf.Tensor{{Name: "weight", TypeID: 1, Dimensions: []uint64{4}}}}}
	m, q, err := owner.nativeStateIdentity()
	if err != nil {
		t.Fatal(err)
	}
	clone := owner
	if clone.stateIdentity != owner.stateIdentity {
		t.Fatal("clone does not share identity cache")
	}
	gotM, gotQ, err := clone.nativeStateIdentity()
	if err != nil || gotM != m || gotQ != q {
		t.Fatalf("clone identity=%x/%x err=%v", gotM, gotQ, err)
	}
	other := owner
	other.stateIdentity = new(modelStateIdentityCache)
	other.gguf = &gguf.File{Tensors: []gguf.Tensor{{Name: "weight", TypeID: 2, Dimensions: []uint64{4}}}}
	otherM, otherQ, err := other.nativeStateIdentity()
	if err != nil || otherM != m || otherQ == q {
		t.Fatalf("quant identity=%x/%x err=%v", otherM, otherQ, err)
	}
}

func TestCUDAStateIdentity(t *testing.T) {
	model, quant := [32]byte{1}, [32]byte{2}
	data := make([]byte, cudaStateHeaderBytes)
	copy(data[40:72], model[:])
	copy(data[72:104], quant[:])
	for _, tt := range []struct {
		name         string
		model, quant [32]byte
		short        bool
		wantErr      bool
	}{
		{"match", model, quant, false, false},
		{"wrong model", [32]byte{3}, quant, false, true},
		{"wrong quant", model, [32]byte{3}, false, true},
		{"missing model", [32]byte{}, quant, false, true},
		{"old header", model, quant, true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := data
			if tt.short {
				b = b[:40]
			}
			if err := validateCUDAStateIdentity(b, tt.model, tt.quant); (err != nil) != tt.wantErr {
				t.Fatalf("error=%v wantErr=%v", err, tt.wantErr)
			}
		})
	}
}

func TestCUDAStateTransaction(t *testing.T) {
	primary, rollbackErr := errors.New("copy failed"), errors.New("rollback failed")
	for _, failRollback := range []bool{false, true} {
		value, poisoned := 7, false
		err := cudaStateTransaction(func() error { value = 9; return primary }, func() error {
			if failRollback {
				return rollbackErr
			}
			value = 7
			return nil
		}, func(error) { poisoned = true })
		if !errors.Is(err, primary) || poisoned != failRollback {
			t.Fatalf("err=%v poisoned=%v", err, poisoned)
		}
		if failRollback {
			if !errors.Is(err, rollbackErr) || !errors.Is(err, dsv4.ErrDesynchronized) {
				t.Fatal(err)
			}
		} else if value != 7 {
			t.Fatal("rollback did not restore value")
		}
	}
}

func TestCUDARestorePrevalidation(t *testing.T) {
	var dst, src cudaSessionState
	if err := validateCUDARestore(dst, src, 3, 2); err == nil {
		t.Fatal("logit mismatch accepted")
	}
	src.layers[0].attn.n = 1
	if err := validateCUDARestore(dst, src, 0, 0); err == nil {
		t.Fatal("overcapacity count accepted")
	}
	for _, position := range []int{-1, 129} {
		if err := validateCUDAStateFrontier(position, 128); err == nil {
			t.Fatalf("position %d accepted", position)
		}
	}
	for _, position := range []int{0, 128} {
		if err := validateCUDAStateFrontier(position, 128); err != nil {
			t.Fatal(err)
		}
	}
}
