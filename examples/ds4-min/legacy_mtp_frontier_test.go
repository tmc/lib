package main

import "testing"

func TestLegacyMTPIdentityGuards(t *testing.T) {
	model, quant := [32]byte{1}, [32]byte{2}
	for _, tc := range []struct {
		name                                       string
		model, quant, expectedModel, expectedQuant [32]byte
		ok                                         bool
	}{
		{"match", model, quant, model, quant, true},
		{"model", [32]byte{3}, quant, model, quant, false},
		{"quant", model, [32]byte{3}, model, quant, false},
		{"missing model", model, quant, [32]byte{}, quant, false},
		{"missing quant", model, quant, model, [32]byte{}, false},
		{"all absent", [32]byte{}, [32]byte{}, [32]byte{}, [32]byte{}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateLegacyMTPIdentity(tc.model, tc.quant, tc.expectedModel, tc.expectedQuant)
			if (err == nil) != tc.ok {
				t.Fatalf("error=%v, want admitted=%v", err, tc.ok)
			}
		})
	}
}

func TestLegacyMTPFrontierGuards(t *testing.T) {
	history := []int{8, 1, 2}
	key := new(legacyMTPRequestKey)
	f, err := newLegacyMTPFrontier(key, 17, 3, history)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.validate(f); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name   string
		change func(*legacyMTPFrontier)
	}{
		{"request", func(c *legacyMTPFrontier) { c.request = new(legacyMTPRequestKey) }},
		{"epoch", func(c *legacyMTPFrontier) { c.epoch++ }},
		{"position", func(c *legacyMTPFrontier) { c.position++ }},
		{"prefix", func(c *legacyMTPFrontier) {
			changed, e := newLegacyMTPFrontier(key, 17, 3, []int{9, 1, 2})
			if e != nil {
				t.Fatal(e)
			}
			*c = changed
		}},
		{"unbound", func(c *legacyMTPFrontier) { c.request = nil }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			changed := f
			tc.change(&changed)
			if err := f.validate(changed); err == nil {
				t.Fatal("stale preparation admitted")
			}
		})
	}
	history[0] = 9
	if err := f.validate(f); err != nil {
		t.Fatalf("caller mutation changed detached frontier: %v", err)
	}
}

func TestLegacyMTPFrontierAdmission(t *testing.T) {
	key := new(legacyMTPRequestKey)
	for _, tc := range []struct {
		name    string
		key     *legacyMTPRequestKey
		pos     int
		history []int
	}{
		{"nil request", nil, 1, []int{1}},
		{"empty", key, 0, nil},
		{"length", key, 2, []int{1}},
		{"negative", key, 1, []int{-1}},
		{"vocabulary", key, 1, []int{Flash.Vocab}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := newLegacyMTPFrontier(tc.key, 1, tc.pos, tc.history); err == nil {
				t.Fatal("invalid frontier admitted")
			}
		})
	}
	for _, pos := range []int{2047, 2056} {
		history := make([]int, pos)
		if _, err := newLegacyMTPFrontier(key, 1, pos, history); err != nil {
			t.Fatal(err)
		}
	}
}
