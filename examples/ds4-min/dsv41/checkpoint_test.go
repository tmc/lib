package dsv41

import (
	"fmt"
	"reflect"
	"testing"
)

func TestCheckpointOwnership(t *testing.T) {
	g := modelFixture(t)
	m, err := Bind(&countReader{}, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.NewState(4)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.NewState(4)
	if err != nil {
		t.Fatal(err)
	}
	for layer := range a.kv.raw {
		a.kv.raw[layer][0] = float32(layer + 3)
		a.kv.raw[layer][len(a.kv.raw[layer])-1] = float32(layer + 7)
	}
	for owner := range a.kv.owners {
		a.kv.owners[owner].kv = []float32{float32(owner + 10), 20}
		a.kv.owners[owner].index = []float32{float32(owner + 30), 40}
		a.pool[owner].kv = []float32{float32(owner + 50), 60}
		a.pool[owner].scores = []float32{float32(owner + 70), 80}
	}
	a.candidates.mask = []bool{true, false}
	a.candidates.selected = []int{0, 1}
	a.position, a.kv.position = 1, 1
	a.tokens = []int32{17}
	a.logits = make([]float32, 129280)
	a.logits[0], a.logits[len(a.logits)-1] = 111, 222
	for site := range a.engram.rows {
		a.engram.rows[site][0] = float32(site + 90)
		a.engram.rows[site][len(a.engram.rows[site])-1] = float32(site + 100)
	}
	checkValues := func(s *State) {
		t.Helper()
		if !reflect.DeepEqual(s.tokens, []int32{17}) || len(s.logits) != 129280 {
			t.Fatal("token or logit storage changed")
		}
		for i, got := range s.logits {
			want := float32(0)
			if i == 0 {
				want = 111
			}
			if i == len(s.logits)-1 {
				want = 222
			}
			if got != want {
				t.Fatalf("logit%d changed: %g != %g", i, got, want)
			}
		}
		for layer, raw := range s.kv.raw {
			for i, got := range raw {
				want := float32(0)
				if i == 0 {
					want = float32(layer + 3)
				}
				if i == len(raw)-1 {
					want = float32(layer + 7)
				}
				if got != want {
					t.Fatalf("raw layer%d index%d changed: %g != %g", layer, i, got, want)
				}
			}
		}
		for owner := range s.kv.owners {
			for _, pair := range []struct{ got, want []float32 }{
				{s.kv.owners[owner].kv, []float32{float32(owner + 10), 20}},
				{s.kv.owners[owner].index, []float32{float32(owner + 30), 40}},
				{s.pool[owner].kv, []float32{float32(owner + 50), 60}},
				{s.pool[owner].scores, []float32{float32(owner + 70), 80}},
			} {
				if !reflect.DeepEqual(pair.got, pair.want) {
					t.Fatalf("owner%d storage changed: %v != %v", owner, pair.got, pair.want)
				}
			}
		}
		for site, rows := range s.engram.rows {
			for i, got := range rows {
				want := float32(0)
				if i == 0 {
					want = float32(site + 90)
				}
				if i == len(rows)-1 {
					want = float32(site + 100)
				}
				if got != want {
					t.Fatalf("engram site%d index%d changed", site, i)
				}
			}
		}
		if !reflect.DeepEqual(s.candidates.mask, []bool{true, false}) || !reflect.DeepEqual(s.candidates.selected, []int{0, 1}) {
			t.Fatal("candidate storage changed")
		}
	}
	c, err := a.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	if err = b.Restore(c); err != nil {
		t.Fatal(err)
	}
	a.Reset()
	checkValues(b)
	mutate := func(x []float32) {
		for i := range x {
			x[i] = 1000
		}
	}
	for _, raw := range b.kv.raw {
		mutate(raw)
	}
	for owner := range b.kv.owners {
		mutate(b.kv.owners[owner].kv)
		mutate(b.kv.owners[owner].index)
		mutate(b.pool[owner].kv)
		mutate(b.pool[owner].scores)
	}
	for _, rows := range b.engram.rows {
		mutate(rows)
	}
	mutate(b.logits)
	for i := range b.tokens {
		b.tokens[i] = 1000
	}
	for i := range b.candidates.mask {
		b.candidates.mask[i] = !b.candidates.mask[i]
	}
	for i := range b.candidates.selected {
		b.candidates.selected[i] = 1000
	}
	if err = b.Restore(c); err != nil {
		t.Fatal(err)
	}
	checkValues(b)
	before := cloneState(b)
	for _, bad := range []*Checkpoint{nil, {}, {state: &State{}}, {state: &State{model: &Model{}}}} {
		if err = b.Restore(bad); err == nil {
			t.Fatal("invalid checkpoint accepted")
		}
		if !reflect.DeepEqual(b, before) {
			t.Fatal("invalid restore mutated destination")
		}
	}
	other, err := Bind(&countReader{}, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	d, err := other.NewState(4)
	if err != nil {
		t.Fatal(err)
	}
	if err = d.Restore(c); err == nil {
		t.Fatal("same-shape different model accepted")
	}
	differentContext, err := m.NewState(3)
	if err != nil {
		t.Fatal(err)
	}
	if err = differentContext.Restore(c); err == nil {
		t.Fatal("different context accepted")
	}
	b.valid = false
	if _, err = b.Checkpoint(); err == nil {
		t.Fatal("failed request checkpointed")
	}
	if err = b.Restore(c); err != nil || !b.valid {
		t.Fatal("failed request not recovered", err)
	}
	b.kv.layer = 1
	if _, err = b.Checkpoint(); err == nil {
		t.Fatal("partial token checkpointed")
	}
}

func ExampleState_Checkpoint() {
	var state State
	_, err := state.Checkpoint()
	fmt.Println(err != nil)
	// Output: true
}

func ExampleState_Restore() {
	var state State
	fmt.Println(state.Restore(nil) != nil)
	// Output: true
}
