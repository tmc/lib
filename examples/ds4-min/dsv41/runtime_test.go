package dsv41

import (
	"math"
	"reflect"
	"testing"
)

func TestSchedule(t *testing.T) {
	owners := []int{2, 8, 14, 20}
	for i := 0; i < 40; i++ {
		l, err := Schedule(i)
		if err != nil {
			t.Fatal(err)
		}
		if i < 2 {
			if l.Owner != -1 || l.Ratio != 0 {
				t.Fatal(i, l)
			}
			continue
		}
		if owners[l.Owner] > i || l.PublishKV != (owners[l.Owner] == i) {
			t.Fatal(i, l)
		}
		if l.Ratio != map[bool]int{true: 2, false: 1}[i < 20] {
			t.Fatal(i, l)
		}
	}
}

func TestPoolChunkCarry(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	scores := make([]float32, 6)
	var whole, split Pool
	a, b := make([]float32, 3), make([]float32, 3)
	if err := whole.Append(a, x, scores, 1, 0); err != nil {
		t.Fatal(err)
	}
	if err := split.Append(nil, x[:1], scores[:1], 1, 0); err != nil {
		t.Fatal(err)
	}
	if err := split.Append(b[:2], x[1:4], scores[1:4], 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := split.Append(b[2:], x[4:], scores[4:], 1, 4); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal(a, b)
	}
	before := append([]float32(nil), b...)
	if err := split.Append(b, x, scores, 1, 7); err == nil || !reflect.DeepEqual(before, b) {
		t.Fatal("invalid position changed output")
	}
}

func TestEngramGateValue(t *testing.T) {
	r := make([]float32, 128)
	kv := make([]float32, 160)
	q, k := make([]float32, 128), make([]float32, 128)
	for i := range r {
		r[i] = 1
		kv[i] = 1
		q[i] = 1
		k[i] = 1
	}
	for i := 128; i < 160; i++ {
		kv[i] = 2
	}
	if err := AddEngram(r, kv, q, k, 32, 1e-20); err != nil {
		t.Fatal(err)
	}
	d := float32(math.Sqrt(32))
	s := float32(math.Sqrt(float64(d)))
	gate := float32(1) / float32(1+float32(math.Exp(float64(-s))))
	want := BF16(1 + gate*2)
	for i, v := range r {
		if v != want {
			t.Fatalf("%d: %g want %g", i, v, want)
		}
	}
}

func TestCandidateMaskAndInheritance(t *testing.T) {
	var c Candidates
	scores := make([]float32, 16400)
	for i := range scores {
		scores[i] = float32(len(scores) - i)
	}
	ids, err := c.Pick(20, len(scores)-1, scores)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 512 || !c.mask[len(c.mask)-1] {
		t.Fatal("latest block not admitted")
	}
	if c.mask[2047] || c.mask[2048] {
		t.Fatal("low-scoring block admitted")
	}
	inherited, err := c.Pick(21, len(scores)-1, nil)
	if err != nil || !reflect.DeepEqual(ids, inherited) {
		t.Fatal(inherited, err)
	}
	scores[2047*8] = float32(math.Inf(1))
	ids, err = c.Pick(24, len(scores)-1, scores)
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range ids {
		if i == 2047*8 {
			t.Fatal("filtered block selected")
		}
	}
	var other Candidates
	if _, err := other.Pick(24, len(scores)-1, scores); err == nil {
		t.Fatal("mask leaked between requests")
	}
}

func TestCandidateSpecialScores(t *testing.T) {
	var c Candidates
	ids, err := c.Pick(20, 3, []float32{float32(math.Inf(-1)), float32(math.Inf(1)), float32(math.Inf(1)), 0})
	if err != nil || !reflect.DeepEqual(ids, []int{1, 2, 3, 0}) {
		t.Fatal(ids, err)
	}
	before := append([]int(nil), c.selected...)
	if _, err := c.Pick(24, 3, []float32{0, float32(math.NaN()), 1, 2}); err == nil {
		t.Fatal("NaN admitted")
	}
	if !reflect.DeepEqual(before, c.selected) {
		t.Fatal("NaN altered selection")
	}
	// Literal source priority is +Inf, not a tie-breaking override. If input
	// blocks are themselves +Inf, upstream topk prefers lower block IDs.
	scores := make([]float32, 2050*8)
	for i := range scores {
		scores[i] = float32(math.Inf(1))
	}
	if _, err := c.Pick(20, len(scores)-1, scores); err != nil {
		t.Fatal(err)
	}
	if c.mask[len(c.mask)-1] {
		t.Fatal("latest block incorrectly overrides +Inf ties")
	}
}
