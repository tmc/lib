package nvl72

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

func seqTokens(start, n int) []int32 {
	t := make([]int32, n)
	for i := range t {
		t[i] = int32(start + i)
	}
	return t
}

// cachePrefix stores tokens in a's radix tree and releases them, as a
// finished prefill would.
func cachePrefix(t *testing.T, a *RadixPageAllocator, id string, tokens []int32) {
	t.Helper()
	if _, err := a.MatchOrAllocatePrefix(id, paginate(tokens, a.PageSize)); err != nil {
		t.Fatal(err)
	}
	if err := a.ReleaseRequest(id); err != nil {
		t.Fatal(err)
	}
}

func newAllocator(t *testing.T, hbmPages, c2cPages int) *RadixPageAllocator {
	t.Helper()
	a, err := NewRadixPageAllocator(16, 16*640, hbmPages, c2cPages)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// newTestCoordinator returns ranks 0..3 active with spares 70 and 71.
// Rank 2 holds the first 3840 tokens of the shared system prompt, rank 1
// the first 1024, rank 3 an unrelated prompt, and rank 0 nothing.
func newTestCoordinator(t *testing.T) (*Coordinator, []int32) {
	t.Helper()
	prompt := append(seqTokens(1000, 3840), seqTokens(90000, 256)...)
	c := &Coordinator{
		ActiveRanks: []int{0, 1, 2, 3},
		SpareRanks:  []int{70, 71},
		RadixTrees:  map[int]*RadixPageAllocator{},
	}
	for r, n := range map[int]int{1: 1024, 2: 3840} {
		a := newAllocator(t, 512, 0)
		cachePrefix(t, a, "sys", prompt[:n])
		c.RadixTrees[r] = a
	}
	a := newAllocator(t, 512, 0)
	cachePrefix(t, a, "other", seqTokens(5000, 2048))
	c.RadixTrees[3] = a
	return c, prompt
}

func TestCoordinatorRoutePrompt(t *testing.T) {
	c, prompt := newTestCoordinator(t)
	tests := []struct {
		name      string
		load      map[int]int
		loadCap   int
		wantRank  int
		wantMatch int
		wantErr   error
	}{
		{"longest_prefix", nil, 0, 2, 3840, nil},
		{"rank2_full", map[int]int{2: 8}, 8, 1, 1024, nil},
		{"rank1_and_2_full", map[int]int{1: 8, 2: 9}, 8, 0, 0, nil},
		{"tie_lower_load", map[int]int{0: 3, 1: 8, 2: 8, 3: 1}, 8, 3, 0, nil},
		{"all_full", map[int]int{0: 8, 1: 8, 2: 8, 3: 8}, 8, -1, 0, ErrNoEligibleRank},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rank, match, err := c.RoutePrompt(prompt, tc.load, tc.loadCap)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if rank != tc.wantRank || match != tc.wantMatch {
				t.Fatalf("RoutePrompt = rank %d, %d tokens; want rank %d, %d tokens", rank, match, tc.wantRank, tc.wantMatch)
			}
		})
	}
	rank, match, _ := c.RoutePrompt(prompt, nil, 0)
	if saved := float64(match) / float64(len(prompt)); rank != 2 || saved != 0.9375 {
		t.Errorf("rank %d saves %.4f of prefill, want rank 2 saving 0.9375", rank, saved)
	}
}

func TestMatchPrefixReadOnly(t *testing.T) {
	a := newAllocator(t, 64, 0)
	prompt := seqTokens(0, 512)
	cachePrefix(t, a, "p", prompt)
	used := a.HBMUsedPages()
	for range 3 {
		h, c2c := a.MatchPrefix(paginate(prompt, 16))
		if h != 32 || c2c != 0 {
			t.Fatalf("MatchPrefix = %d HBM, %d C2C pages; want 32, 0", h, c2c)
		}
	}
	if a.HBMUsedPages() != used {
		t.Errorf("MatchPrefix changed HBM use from %d to %d", used, a.HBMUsedPages())
	}
	// A diverging page ends the match.
	other := slices.Clone(prompt)
	other[100] = -1
	if h, _ := a.MatchPrefix(paginate(other, 16)); h != 6 {
		t.Errorf("diverging at token 100: matched %d pages, want 6", h)
	}
}

func TestMatchPrefixTiers(t *testing.T) {
	// HBM holds 4 pages; with C2C room, evicted pages are demoted and
	// still match; without it they are dropped and stop the match.
	for _, tc := range []struct {
		c2c      int
		wantHBM  int
		wantC2C  int
		wantDesc string
	}{
		{4, 0, 4, "demoted to C2C"},
		{0, 0, 0, "dropped"},
	} {
		a := newAllocator(t, 4, tc.c2c)
		first := seqTokens(0, 64)
		cachePrefix(t, a, "a", first)
		cachePrefix(t, a, "b", seqTokens(1000, 64))
		h, c2c := a.MatchPrefix(paginate(first, 16))
		if h != tc.wantHBM || c2c != tc.wantC2C {
			t.Errorf("%s: MatchPrefix = %d HBM, %d C2C; want %d, %d", tc.wantDesc, h, c2c, tc.wantHBM, tc.wantC2C)
		}
	}
}

func TestCoordinatorFailoverRank(t *testing.T) {
	c, prompt := newTestCoordinator(t)
	tree1, tree3 := c.RadixTrees[1], c.RadixTrees[3]

	if _, _, err := c.FailoverRank(2, 13); !errors.Is(err, ErrXIDNotFatal) {
		t.Fatalf("xid 13: err = %v, want ErrXIDNotFatal", err)
	}
	if !slices.Equal(c.ActiveRanks, []int{0, 1, 2, 3}) || len(c.SpareRanks) != 2 || c.RadixTrees[2] == nil {
		t.Fatal("non-fatal xid changed the coordinator")
	}

	active, repl, err := c.FailoverRank(2, 79)
	if err != nil {
		t.Fatal(err)
	}
	if repl != 70 || !slices.Equal(active, []int{0, 1, 70, 3}) {
		t.Fatalf("FailoverRank = %v, %d; want [0 1 70 3], 70", active, repl)
	}
	if _, ok := c.RadixTrees[2]; ok {
		t.Error("faulted rank's KV cache kept")
	}
	if c.RadixTrees[1] != tree1 || c.RadixTrees[3] != tree3 {
		t.Error("healthy ranks' KV caches replaced")
	}
	active[0] = 99
	if c.ActiveRanks[0] != 0 {
		t.Error("returned slice aliases ActiveRanks")
	}
	if rank, match, err := c.RoutePrompt(prompt, nil, 0); err != nil || rank != 1 || match != 1024 {
		t.Errorf("after failover RoutePrompt = %d, %d, %v; want 1, 1024", rank, match, err)
	}

	for _, tc := range []struct {
		rank, xid int
		want      error
	}{
		{2, 48, ErrRankNotActive},
		{70, 149, nil},
		{99, 119, ErrRankNotActive},
		{0, 48, ErrNoSpareRank},
	} {
		_, _, err := c.FailoverRank(tc.rank, tc.xid)
		if !errors.Is(err, tc.want) {
			t.Errorf("FailoverRank(%d, %d) = %v, want %v", tc.rank, tc.xid, err, tc.want)
		}
	}
	if !slices.Equal(c.ActiveRanks, []int{0, 1, 71, 3}) {
		t.Errorf("ActiveRanks = %v, want [0 1 71 3]", c.ActiveRanks)
	}
}

func ExampleCoordinator_RoutePrompt() {
	prefix := make([]int32, 64)
	for i := range prefix {
		prefix[i] = int32(i)
	}
	a, _ := NewRadixPageAllocator(16, 16*640, 64, 0)
	a.MatchOrAllocatePrefix("warm", paginate(prefix, 16))
	a.ReleaseRequest("warm")

	c := &Coordinator{
		ActiveRanks: []int{0, 1},
		SpareRanks:  []int{7},
		RadixTrees:  map[int]*RadixPageAllocator{1: a},
	}
	prompt := append(prefix, 500, 501, 502)
	rank, matched, _ := c.RoutePrompt(prompt, nil, 0)
	fmt.Println(rank, matched)
	active, spare, _ := c.FailoverRank(1, 79)
	fmt.Println(active, spare)
	// Output:
	// 1 64
	// [0 7] 7
}

func TestCoordinatorPrefersHBM(t *testing.T) {
	first := seqTokens(0, 64)
	demoted := newAllocator(t, 4, 4)
	cachePrefix(t, demoted, "a", first)
	cachePrefix(t, demoted, "b", seqTokens(1000, 64))
	hot := newAllocator(t, 4, 4)
	cachePrefix(t, hot, "a", first)
	c := &Coordinator{ActiveRanks: []int{0, 1}, RadixTrees: map[int]*RadixPageAllocator{0: demoted, 1: hot}}
	rank, match, err := c.RoutePrompt(first, nil, 0)
	if err != nil || rank != 1 || match != 64 {
		t.Fatalf("RoutePrompt = %d, %d, %v; want HBM rank 1 with 64 tokens", rank, match, err)
	}
}
