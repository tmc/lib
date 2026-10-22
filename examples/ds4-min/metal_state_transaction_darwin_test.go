//go:build darwin && arm64

package main

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalStateFieldTransaction(t *testing.T) {
	for _, rollbackFails := range []bool{false, true} {
		t.Run(map[bool]string{false: "rollback", true: "poison"}[rollbackFails], func(t *testing.T) {
			fields := []any{[]float32{1, 2}, []uint32{3, 4}, []byte{5, 6}}
			incoming := []any{[]float32{7, 8}, []uint32{9, 10}, []byte{11, 12}}
			backup := make([][]byte, len(fields))
			encoded := make([][]byte, len(fields))
			for i := range fields {
				backup[i], _ = encodeMetalStateField(fields[i])
				encoded[i], _ = encodeMetalStateField(incoming[i])
			}
			primary, rollback := errors.New("injected write failure"), errors.New("injected rollback failure")
			var poisoned error
			hooks := &metalStateLoadHooks{afterWrite: func(i int, restoring bool) error {
				if restoring && rollbackFails && i == 0 {
					return rollback
				}
				if !restoring && i == 1 {
					for j := 0; j <= i; j++ {
						got, _ := encodeMetalStateField(fields[j])
						if !bytes.Equal(got, encoded[j]) {
							t.Fatal("hook ran before actual field write")
						}
					}
					return primary
				}
				return nil
			}}
			err := replaceMetalStateFields(fields, encoded, hooks, func(err error) { poisoned = err })
			if !errors.Is(err, primary) {
				t.Fatalf("primary error lost: %v", err)
			}
			if rollbackFails {
				if !errors.Is(err, rollback) || !errors.Is(poisoned, dsv4.ErrDesynchronized) {
					t.Fatalf("rollback failure did not poison: %v", err)
				}
				return
			}
			if poisoned != nil {
				t.Fatalf("successful rollback poisoned state: %v", poisoned)
			}
			for i := range fields {
				got, _ := encodeMetalStateField(fields[i])
				if !bytes.Equal(got, backup[i]) {
					t.Fatalf("rollback field %d changed", i)
				}
			}
		})
	}
}

func TestMetalStateTerminalPoison(t *testing.T) {
	r := &metalRuntime{arena: &metalmem.Arena{}, pos: 1, failed: errors.Join(errors.New("rollback failed"), dsv4.ErrDesynchronized)}
	s := &metalRequestState{runtime: r, owner: &metalStateAdapter{}}
	checks := []struct {
		name string
		call func() error
	}{
		{"reset", r.Reset},
		{"eval", func() error { return r.Eval(0, 1) }},
		{"empty batch", func() error { return r.EvalBatch(nil, 1) }},
		{"save", func() error { return s.Save(&bytes.Buffer{}) }},
		{"load", func() error { return s.Load(bytes.NewReader(nil)) }},
		{"snapshot", func() error { _, err := s.Snapshot(); return err }},
		{"restore", func() error { return s.Restore(nil) }},
	}
	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if err := check.call(); !errors.Is(err, dsv4.ErrDesynchronized) {
				t.Fatalf("poison not retained: %v", err)
			}
		})
	}
	if r.Logits() != nil {
		t.Fatal("poisoned logits published")
	}
}
