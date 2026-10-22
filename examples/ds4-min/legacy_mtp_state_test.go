package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

type legacyMTPValueReader struct {
	tensors  []gguf.Tensor
	negative bool
}

func (r legacyMTPValueReader) ReadAt(p []byte, off int64) (int, error) {
	clear(p)
	for _, t := range r.tensors {
		n, err := t.Bytes()
		if err != nil {
			return 0, err
		}
		if uint64(off) < t.Offset || uint64(off) >= t.Offset+n {
			continue
		}
		if strings.HasSuffix(t.Name, "norm.weight") {
			for i := 0; i < len(p); i += 4 {
				binary.LittleEndian.PutUint32(p[i:], math.Float32bits(1))
			}
		}
		if t.Name == "mtp.0.e_proj.weight" && uint64(off) == t.Offset {
			binary.LittleEndian.PutUint16(p, 0x3c00)
			p[2] = 1
			if r.negative {
				p[2] = 255
			}
		}
		if t.Name == "mtp.0.h_proj.weight" && uint64(off) == t.Offset+uint64(Flash.Embed/32*34) {
			binary.LittleEndian.PutUint16(p, 0x3c00)
			p[3] = 1
		}
		return len(p), nil
	}
	return 0, fmt.Errorf("unknown fixture offset")
}

func legacyMTPValueFixture(t *testing.T) (*legacyMTPModel, *gguf.File) {
	t.Helper()
	g := &gguf.File{Size: 1 << 40}
	m := &dsparkModel{}
	rules := dsparkRules(1, m)
	e, h := uint64(Flash.Embed), uint64(Flash.HC)
	rules = append(rules, dsparkRule{"hc_head_base.weight", 0, []uint64{h}}, dsparkRule{"hc_head_fn.weight", 1, []uint64{e * h, h}}, dsparkRule{"hc_head_scale.weight", 0, []uint64{1}}, dsparkRule{"e_proj.weight", 2, []uint64{e, e}}, dsparkRule{"h_proj.weight", 2, []uint64{e, e}}, dsparkRule{"enorm.weight", 0, []uint64{e}}, dsparkRule{"hnorm.weight", 0, []uint64{e}}, dsparkRule{"norm.weight", 0, []uint64{e}})
	var offset uint64
	for _, r := range rules {
		typ := uint32(0)
		if r.kind == 2 || r.kind == 3 {
			typ = 8
		}
		if r.name == "ffn_gate_inp.weight" {
			typ = 0
		}
		tensor := gguf.Tensor{Name: "mtp.0." + r.name, TypeID: typ, Dimensions: r.dims, Offset: offset}
		n, err := tensor.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		offset += n
		g.Tensors = append(g.Tensors, tensor)
	}
	model, err := admitLegacyMTPModel(g)
	if err != nil {
		t.Fatal(err)
	}
	model.weights.reader = legacyMTPValueReader{tensors: g.Tensors}
	return model, g
}

func TestLegacyMTPFullStepFloat64(t *testing.T) {
	// Actual full-shape Q8 projection tensors have one nonzero coefficient
	// each; attention and MoE projections are zero. This exercises the whole
	// pipeline while yielding an independent, nonconstant analytic head.
	model, g := legacyMTPValueFixture(t)
	embedding := make([]float32, Flash.Embed)
	embedding[0] = 2
	embedding[2] = 1
	hidden := make([]float32, Flash.HC*Flash.Embed)
	var b float64
	for lane := 0; lane < Flash.HC; lane++ {
		x := float32(lane + 1)
		hidden[lane*Flash.Embed+1] = x
		hidden[lane*Flash.Embed+3] = 4 - x
		b += float64(x) / math.Sqrt((float64(x*x)+float64((4-x)*(4-x)))/float64(Flash.Embed)+float64(Flash.RMSEps)) / float64(Flash.HC)
	}
	a := 2 / math.Sqrt(5/float64(Flash.Embed)+float64(Flash.RMSEps))
	s, err := newLegacyMTPState(model)
	if err != nil {
		t.Fatal(err)
	}
	head, _, err := s.step(embedding, hidden, 0)
	if err != nil {
		t.Fatal(err)
	}
	scale := 1 / math.Sqrt((a*a+b*b)/float64(Flash.Embed)+float64(Flash.RMSEps))
	for i, v := range head {
		want := 0.0
		if i == 0 {
			want = a * scale
		}
		if i == 1 {
			want = b * scale
		}
		if math.Abs(float64(v)-want) > 2e-4 {
			t.Fatalf("full step[%d]=%g want%g", i, v, want)
		}
	}
	// A byte-level sign mutation in the embedding projection must change the
	// corresponding head coordinate, not vanish in a normalized comparison.
	model.weights.reader = legacyMTPValueReader{tensors: g.Tensors, negative: true}
	mutant, _ := newLegacyMTPState(model)
	changed, _, err := mutant.step(embedding, hidden, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(changed[0]+head[0])) > 2e-4 || changed[0] >= 0 {
		t.Fatal("embedding sign mutation not observable")
	}
}

type legacyMTPTargetReader struct{}

func (legacyMTPTargetReader) ReadAt(p []byte, off int64) (int, error) {
	clear(p)
	// Two-row lifecycle projection fixture: row0 selects positive coordinate0,
	// row1 selects negative coordinate0. Embedding rows are [2,0,1,0,...].
	if off < 4*int64(Flash.Embed)*4 {
		binary.LittleEndian.PutUint32(p, math.Float32bits(2))
		binary.LittleEndian.PutUint32(p[8:], math.Float32bits(1))
	} else {
		binary.LittleEndian.PutUint16(p, 0x3c00)
		p[2] = 1
		if off > 4*int64(Flash.Embed)*4 {
			p[2] = 255
		}
	}
	return len(p), nil
}

func TestLegacyMTPPrepareFrontierAndEOS(t *testing.T) {
	model, _ := legacyMTPValueFixture(t)
	target := &dsparkModel{reader: legacyMTPTargetReader{}, tensors: map[string]gguf.Tensor{
		"token_embd.weight": {Name: "token_embd.weight", TypeID: 0, Dimensions: []uint64{uint64(Flash.Embed), 4}},
		"output.weight":     {Name: "output.weight", TypeID: 8, Dimensions: []uint64{uint64(Flash.Embed), 2}, Offset: uint64(4 * Flash.Embed * 4)},
	}}
	for _, frontier := range []int{2047, 2056} {
		s, _ := newLegacyMTPState(model)
		d := &legacyMTPDrafter{state: s, target: target, margin: 0, eos: 0}
		hc := make([]float32, Flash.Embed*Flash.HC)
		for i := range hc {
			hc[i] = 1
		}
		history := make([]int, frontier+1)
		history[frontier] = 3
		if err := d.prepareHistory(history, hc); err != nil {
			t.Fatal(err)
		}
		encoded := make([]byte, len(history)*4)
		binary.LittleEndian.PutUint32(encoded[frontier*4:], 3)
		if d.historyHash != fmt.Sprintf("%x", sha256.Sum256(encoded)) {
			t.Fatal("prepared history hash does not bind zero-based committed prefix")
		}
		seed, pos, err := d.PreparedSeed()
		if err != nil || seed != 3 || pos != frontier+1 {
			t.Fatal("producer does not expose prepared committed frontier", seed, pos, err)
		}
		history[0] = 1
		binary.LittleEndian.PutUint32(encoded, 1)
		if d.historyHash == fmt.Sprintf("%x", sha256.Sum256(encoded)) {
			t.Fatal("history hash changed through borrowed prefix")
		}
		if d.state.next != frontier+1 || d.state.count != 1 {
			t.Fatal("nonzero frontier did not install one seed row")
		}
		oldHidden := d.hidden[1]
		drafts, err := d.Draft(3, frontier+1, 4)
		if err != nil {
			t.Fatal(err)
		}
		if len(drafts) != 1 || drafts[0] != 0 {
			t.Fatal("EOS did not truncate proposal")
		}
		if err := d.DiscardDrafts(); err != nil {
			t.Fatal(err)
		}
		if _, _, err := d.PreparedSeed(); err == nil {
			t.Fatal("discard reused a consumed preparation")
		}
		if d.state.count != 1 || d.state.next != frontier+1 {
			t.Fatal("free miss discarded prepared seed row")
		}
		if _, err := d.Draft(3, frontier+1, 1); err == nil {
			t.Fatal("reused stale target HC without preparation")
		}
		// Refresh with independently supplied target HC, not prior support HC.
		for i := range hc {
			hc[i] = 0
		}
		for lane := 0; lane < Flash.HC; lane++ {
			hc[lane*Flash.Embed+1] = 2
		}
		history[0] = 0
		history = append(history, 0)
		if err := d.prepareHistory(history, hc); err != nil {
			t.Fatal(err)
		}
		if d.state.count != 2 || d.state.next != frontier+2 {
			t.Fatal("target HC refresh lost frontier")
		}
		if d.hidden[1] <= 10*oldHidden {
			t.Fatal("preparation reused stale support HC instead of fresh target HC")
		}
		drafts, err = d.Draft(0, frontier+2, 1)
		if err != nil || len(drafts) != 1 {
			t.Fatal("refreshed proposal failed", err)
		}
		if err := d.CommitDrafts(1); err != nil {
			t.Fatal(err)
		}
		if d.state.count != 2 || d.state.next != frontier+2 {
			t.Fatal("accepted first prefix lost prepared row")
		}
	}
}

func TestLegacyMTPPreparedSeedRequiresHistory(t *testing.T) {
	d := &legacyMTPDrafter{state: &legacyMTPState{next: 9}, prepared: true, seed: 3, seedPosition: 8}
	if _, _, err := d.PreparedSeed(); err == nil {
		t.Fatal("accepted unbound preparation")
	}
	d.historyHash = strings.Repeat("a", 64)
	if seed, pos, err := d.PreparedSeed(); err != nil || seed != 3 || pos != 9 {
		t.Fatal("bound seed", seed, pos, err)
	}
	d.state.next = 10
	if _, _, err := d.PreparedSeed(); err == nil {
		t.Fatal("accepted stale support position")
	}
	var missing *legacyMTPDrafter
	if _, _, err := missing.PreparedSeed(); err == nil {
		t.Fatal("accepted missing producer")
	}
}

func TestLegacyMTPStateFailureAndClone(t *testing.T) {
	s, err := newLegacyMTPState(&legacyMTPModel{weights: &dsparkModel{}})
	if err != nil {
		t.Fatal(err)
	}
	s.raw[0] = 7
	c := s.clone()
	c.raw[0] = 9
	if s.raw[0] != 7 {
		t.Fatal("snapshot aliases active ring")
	}
	if _, _, err := s.step(nil, nil, 0); err == nil {
		t.Fatal("accepted malformed input")
	}
	if s.raw[0] != 7 || s.count != 0 || s.next != 0 {
		t.Fatal("failed step mutated ring")
	}
	d := &legacyMTPDrafter{state: s, hidden: []float32{1}, pending: []*legacyMTPState{c}, pendingHidden: [][]float32{{2}}}
	if err := d.commit(2); err == nil {
		t.Fatal("accepted invalid commit")
	}
	if d.state != s || d.hidden[0] != 1 {
		t.Fatal("invalid commit mutated state")
	}
	if err := d.commit(1); err != nil {
		t.Fatal(err)
	}
	c.raw[0] = 11
	if d.state.raw[0] != 9 || d.hidden[0] != 2 {
		t.Fatal("commit failed to detach frontier")
	}
	if len(d.pending) != 0 {
		t.Fatal("commit retained speculative suffix")
	}
}
