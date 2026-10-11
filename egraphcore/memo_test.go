package egraphcore

import (
	"errors"
	"reflect"
	"testing"
)

func TestMemoLookupConfirmsEqualityAcrossCollisions(t *testing.T) {
	type payload struct{ value int }
	want := &payload{7}
	other := &payload{9}
	// Unsupported pointers share a type-only structural bucket. They must
	// still be compared by value, including after class owners merge.
	if StructuralFingerprint("leaf", "", want, nil) != StructuralFingerprint("leaf", "", other, nil) {
		t.Fatal("test values did not collide")
	}
	visits := make(map[uint32]int)
	id, ok, err := FindMemoCandidate([]uint32{0, 1, 2, 3}, want,
		func(id uint32) uint32 {
			if id == 1 {
				return 0
			}
			return id
		},
		func(id uint32) []*payload {
			visits[id]++
			if id == 2 {
				return []*payload{want}
			}
			return []*payload{other}
		},
		func(value *payload) *payload { return value },
		func(a, b *payload) bool { return reflect.DeepEqual(a, b) },
	)
	if err != nil || !ok || id != 2 || visits[0] != 1 || visits[3] != 0 {
		t.Fatalf("id=%d ok=%v visits=%v err=%v", id, ok, visits, err)
	}
	if _, _, err := FindMemoCandidate[uint32, int](nil, 0, nil, nil, nil, nil); !errors.Is(err, ErrInvalidMemoLookup) {
		t.Fatalf("err=%v", err)
	}
}

func TestFingerprintRefusesCyclesAndPreservesTypes(t *testing.T) {
	cyclic := make([]any, 1)
	cyclic[0] = cyclic
	if _, ok := ValueFingerprint(cyclic); ok {
		t.Fatal("accepted cyclic payload")
	}
	a, aok := ValueFingerprint(int32(7))
	b, bok := ValueFingerprint(int64(7))
	if !aok || !bok || a == b {
		t.Fatal("lost dynamic integer type")
	}
}
