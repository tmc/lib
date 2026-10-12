package cudaspec_test

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamtp"
	"github.com/tmc/lib/nvidia/x/cudaspec"
)

func TestSpeculationKeyModelIdentity(t *testing.T) {
	type input struct {
		model  string
		prefix []int32
		mode   string
	}
	for _, tt := range []struct {
		name string
		a, b input
	}{
		{"revision beyond 32 bytes",
			input{"meta-llama/Llama-3.1-8B-Instruct@rev1", []int32{1}, "fp32"},
			input{"meta-llama/Llama-3.1-8B-Instruct@rev2", []int32{1}, "fp32"}},
		{"suffix beyond 32 bytes",
			input{"0123456789abcdef0123456789abcdef", []int32{1}, "fp32"},
			input{"0123456789abcdef0123456789abcdef!", []int32{1}, "fp32"}},
		{"model and mode boundary",
			input{"ab", []int32{1}, "c"},
			input{"a", []int32{1}, "bc"}},
		{"model and prefix boundary",
			input{"x\x05\x00\x00\x00", nil, "fp32"},
			input{"x", []int32{5}, "fp32"}},
		{"prefix length",
			input{"m", []int32{0}, "fp32"},
			input{"m", []int32{0, 0}, "fp32"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ka := cudaspec.ComputeSpeculationKey([]byte(tt.a.model), tt.a.prefix, 1, 0, 1, tt.a.mode, 1)
			kb := cudaspec.ComputeSpeculationKey([]byte(tt.b.model), tt.b.prefix, 1, 0, 1, tt.b.mode, 1)
			if ka == kb {
				t.Fatalf("distinct identities share key %x", ka.ModelDigest)
			}
			ka.DraftDigest, kb.DraftDigest = [32]byte{1}, [32]byte{1}
			c := cudaspec.NewSpeculationCache(4)
			if err := c.Put(&cudaspec.SpeculationCacheEntry{Key: ka, BonusToken: 7}); err != nil {
				t.Fatal(err)
			}
			if _, ok := c.Get(ka); !ok {
				t.Fatal("missing entry for its own key")
			}
			if e, ok := c.Get(kb); ok {
				t.Fatalf("reused entry %+v across identities", e)
			}
		})
	}
}

func TestSpeculationKeyModelDigestStable(t *testing.T) {
	// A 32-byte digest is the identity itself, so keys recomputed from a key's
	// ModelDigest must reproduce it.
	k := cudaspec.ComputeSpeculationKey([]byte("meta-llama/Llama-3.1-8B-Instruct@rev1"), []int32{3}, 1, 0, 1, "fp32", 1)
	again := cudaspec.ComputeSpeculationKey(k.ModelDigest[:], []int32{3}, 1, 0, 1, "fp32", 1)
	if again != k {
		t.Fatalf("recomputed key %+v, want %+v", again, k)
	}
}

func TestSpeculationKeyIdentityNotDigest(t *testing.T) {
	// An identity is hashed with a domain tag and length, so it never
	// produces the key of the raw 32-byte digest of the same bytes.
	for _, id := range []string{"", "a", "meta-llama/Llama-3.1-8B-Instruct@rev1"} {
		sum := sha256.Sum256([]byte(id))
		ki := cudaspec.ComputeSpeculationKey([]byte(id), []int32{1}, 1, 0, 1, "fp32", 1)
		kd := cudaspec.ComputeSpeculationKey(sum[:], []int32{1}, 1, 0, 1, "fp32", 1)
		if ki == kd {
			t.Errorf("identity %q has the key of its raw digest", id)
		}
	}
}

func TestSpeculationCacheCopiesEntries(t *testing.T) {
	key := cudaspec.ComputeSpeculationKey([]byte("model"), []int32{1, 2}, 1, 2, 2, "fp32", 1)
	key.DraftDigest = [32]byte{1}
	for _, tt := range []struct {
		name   string
		mutate func(put, got *cudaspec.SpeculationCacheEntry)
	}{
		{"put draft", func(p, _ *cudaspec.SpeculationCacheEntry) { p.DraftPath[0] = 99 }},
		{"put logits", func(p, _ *cudaspec.SpeculationCacheEntry) { p.Logits[0][0] = 99 }},
		{"put logits rows", func(p, _ *cudaspec.SpeculationCacheEntry) { p.Logits[0] = []float32{99, 99} }},
		{"put bonus", func(p, _ *cudaspec.SpeculationCacheEntry) { p.BonusToken = 99 }},
		{"get draft", func(_, g *cudaspec.SpeculationCacheEntry) { g.DraftPath[0] = 99 }},
		{"get logits", func(_, g *cudaspec.SpeculationCacheEntry) { g.Logits[0][0] = 99 }},
		{"get bonus", func(_, g *cudaspec.SpeculationCacheEntry) { g.BonusToken = 99 }},
		{"get key", func(_, g *cudaspec.SpeculationCacheEntry) { g.Key.Generation = 99 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := cudaspec.NewSpeculationCache(4)
			put := &cudaspec.SpeculationCacheEntry{Key: key, DraftPath: []int32{4, 5}, Logits: [][]float32{{1, 2}, {3, 4}}, BonusToken: 6}
			if err := c.Put(put); err != nil {
				t.Fatal(err)
			}
			got, ok := c.Get(key)
			if !ok {
				t.Fatal("missing entry")
			}
			tt.mutate(put, got)
			e, ok := c.Get(key)
			if !ok {
				t.Fatal("missing entry after mutation")
			}
			if e.Key != key || len(e.DraftPath) != 2 || e.DraftPath[0] != 4 || e.DraftPath[1] != 5 || len(e.Logits) != 2 || e.Logits[0][0] != 1 || e.Logits[0][1] != 2 || e.Logits[1][1] != 4 || e.BonusToken != 6 {
				t.Fatalf("cache contents changed: %+v", e)
			}
		})
	}
}

func TestSpeculationKeyDraftDigest(t *testing.T) {
	k1 := cudaspec.ComputeSpeculationKey([]byte("model"), []int32{1, 2}, 1, 2, 2, "fp32", 1)
	k1.DraftDigest = [32]byte{1}
	k2 := k1
	k2.DraftDigest = [32]byte{2}
	c := cudaspec.NewSpeculationCache(4)
	if err := c.Put(&cudaspec.SpeculationCacheEntry{Key: k1, DraftPath: []int32{2, 3}}); err != nil {
		t.Fatal(err)
	}
	if e, ok := c.Get(k2); ok {
		t.Fatalf("reused entry %+v after draft weight change", e)
	}
	if _, ok := c.Get(k1); !ok {
		t.Fatal("missing entry for unchanged draft weights")
	}
}

// cycleRequest returns a host-only batch cycle request using the given MTP
// head weights.
func cycleRequest(t *testing.T, heads []float32) (*cudakv.BlockAllocator, cudaspec.BatchCycleRequest) {
	t.Helper()
	a := cudakv.NewBlockAllocator(2, 8)
	if err := a.AllocSequence(10, nil, 2); err != nil {
		t.Fatal(err)
	}
	req := cudaspec.BatchCycleRequest{Prefix: []int32{0, 1}, PrefixSequence: 10, Sequence: 11, Hidden: []float32{1}, DraftWeights: []float32{1, 1}, HeadWeights: heads, MTP: cudamtp.MTPConfig{NumDraftHeads: 2, Dim: 1, VocabSize: 4}}
	model := [32]byte{1}
	req.Key = cudaspec.ComputeSpeculationKey(model[:], req.Prefix, 1, 2, 2, "fp32", 1)
	req.ExpectedKey = req.Key
	var err error
	req.DraftDigest, err = cudaspec.MTPDraftDigest(req.MTP, req.DraftWeights, req.HeadWeights)
	if err != nil {
		t.Fatal(err)
	}
	return a, req
}

func runCycle(t *testing.T, a *cudakv.BlockAllocator, req cudaspec.BatchCycleRequest) (*cudaspec.BatchCycleResult, error) {
	t.Helper()
	v := cudaspec.BatchVerifier{ReadOnlyPrefix: true, Engine: cudabatch.NewHostDeterministicEngine(0, []int32{2, 3, 1}), Wait: func(*cudabatch.ExecutionStep) error { return nil }}
	result, err := cudaspec.ExecuteBatchCycle(context.Background(), a, req, v)
	if result != nil && result.KV != nil {
		if err := result.KV.Close(); err != nil {
			t.Fatal(err)
		}
	}
	return result, err
}

func TestBatchCycleKeyBindsDraftDigest(t *testing.T) {
	// Both weight sets draft [2 3]; only their digests differ.
	a1, r1 := cycleRequest(t, []float32{0, 0, 2, 0, 0, 0, 0, 3})
	a2, r2 := cycleRequest(t, []float32{0, 0, 2, 0, 0, 0, 0, 4})
	res1, err := runCycle(t, a1, r1)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := runCycle(t, a2, r2)
	if err != nil {
		t.Fatal(err)
	}
	if res1.Key.DraftDigest != r1.DraftDigest || res2.Key.DraftDigest != r2.DraftDigest {
		t.Fatal("result key does not carry the verified draft digest")
	}
	want := r1.Key
	want.DraftDigest = r1.DraftDigest
	if res1.Key != want {
		t.Fatalf("result key = %+v, want %+v", res1.Key, want)
	}
	c := cudaspec.NewSpeculationCache(4)
	if err := c.Put(&cudaspec.SpeculationCacheEntry{Key: res1.Key, DraftPath: res1.Outcome.DraftTokens, BonusToken: res1.Outcome.BonusToken}); err != nil {
		t.Fatal(err)
	}
	if e, ok := c.Get(res2.Key); ok {
		t.Fatalf("reused entry %+v after draft weight change", e)
	}
	if _, ok := c.Get(res1.Key); !ok {
		t.Fatal("missing entry for unchanged draft weights")
	}
}

func TestBatchCycleRejectsKeyDraftDigest(t *testing.T) {
	for _, tt := range []struct {
		name    string
		mutate  func(*cudaspec.BatchCycleRequest)
		wantErr error
	}{
		{"declared", func(r *cudaspec.BatchCycleRequest) {
			r.Key.DraftDigest = r.DraftDigest
			r.ExpectedKey.DraftDigest = r.DraftDigest
		}, nil},
		{"key", func(r *cudaspec.BatchCycleRequest) { r.Key.DraftDigest = [32]byte{9} }, cudaspec.ErrCacheKeyMismatch},
		{"expected", func(r *cudaspec.BatchCycleRequest) { r.ExpectedKey.DraftDigest = [32]byte{9} }, cudaspec.ErrCacheKeyMismatch},
		{"one side", func(r *cudaspec.BatchCycleRequest) { r.Key.DraftDigest = r.DraftDigest }, cudaspec.ErrCacheKeyMismatch},
		{"both wrong", func(r *cudaspec.BatchCycleRequest) {
			r.Key.DraftDigest = [32]byte{9}
			r.ExpectedKey.DraftDigest = [32]byte{9}
		}, cudaspec.ErrCacheKeyMismatch},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a, r := cycleRequest(t, []float32{0, 0, 2, 0, 0, 0, 0, 3})
			tt.mutate(&r)
			_, err := runCycle(t, a, r)
			if !errors.Is(err, tt.wantErr) || (tt.wantErr == nil) != (err == nil) {
				t.Fatalf("err = %v, want %v", err, tt.wantErr)
			}
		})
	}
}

func ExampleSpeculationCache() {
	c := cudaspec.NewSpeculationCache(8)
	key := cudaspec.ComputeSpeculationKey([]byte("org/model@rev1"), []int32{0, 1}, 1, 2, 2, "fp32", 1)
	key.DraftDigest = [32]byte{1}
	entry := &cudaspec.SpeculationCacheEntry{Key: key, DraftPath: []int32{2, 3}}
	if err := c.Put(entry); err != nil {
		fmt.Println(err)
		return
	}
	entry.DraftPath[0] = 9 // does not change the cached copy

	got, ok := c.Get(key)
	fmt.Println(got.DraftPath, ok)

	rev2 := cudaspec.ComputeSpeculationKey([]byte("org/model@rev2"), []int32{0, 1}, 1, 2, 2, "fp32", 1)
	rev2.DraftDigest = key.DraftDigest
	_, ok = c.Get(rev2)
	fmt.Println(ok)

	draft2 := key
	draft2.DraftDigest = [32]byte{2}
	_, ok = c.Get(draft2)
	fmt.Println(ok)
	// Output:
	// [2 3] true
	// false
	// false
}

func TestSpeculationCacheZeroDraftDigestReuse(t *testing.T) {
	// Two cycles whose draft weights differ but whose request keys,
	// lacking DraftDigest, are equal. A drafted result cached under the
	// first request key must not be returned for the second.
	a1, r1 := cycleRequest(t, []float32{0, 0, 2, 0, 0, 0, 0, 3})
	a2, r2 := cycleRequest(t, []float32{0, 0, 2, 0, 0, 0, 0, 4})
	res1, err := runCycle(t, a1, r1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runCycle(t, a2, r2); err != nil {
		t.Fatal(err)
	}
	if r1.Key != r2.Key {
		t.Fatal("fixture request keys differ; scenario needs equal zero-digest keys")
	}
	c := cudaspec.NewSpeculationCache(4)
	if err := c.Put(&cudaspec.SpeculationCacheEntry{Key: r1.Key, DraftPath: res1.Outcome.DraftTokens}); !errors.Is(err, cudaspec.ErrDraftDigestRequired) {
		t.Fatalf("Put without draft digest = %v, want ErrDraftDigestRequired", err)
	}
	if e, ok := c.Get(r2.Key); ok {
		t.Fatalf("reused drafted entry %+v across draft weights", e)
	}
}

func TestSpeculationKey_NonAllZeroWithZeroLeadingByte(t *testing.T) {
	key := cudaspec.ComputeSpeculationKey([]byte("model"), []int32{1, 2}, 1, 0, 2, "fp32", 1)
	key.DraftDigest = [32]byte{31: 1}
	want := &cudaspec.SpeculationCacheEntry{Key: key, DraftPath: []int32{10, 20}, BonusToken: 99}
	c := cudaspec.NewSpeculationCache(4)
	if err := c.Put(want); err != nil {
		t.Fatal(err)
	}
	got, ok := c.Get(key)
	if !ok || got.Key != key || !slices.Equal(got.DraftPath, want.DraftPath) || got.BonusToken != want.BonusToken {
		t.Fatalf("Get = %+v, %v; want %+v", got, ok, want)
	}
}

func TestSpeculationCachePutRequiresDraftDigest(t *testing.T) {
	zero := cudaspec.ComputeSpeculationKey([]byte("model"), []int32{1, 2}, 1, 2, 2, "fp32", 1)
	digest := zero
	digest.DraftDigest = [32]byte{1}
	contents := []struct {
		name  string
		entry cudaspec.SpeculationCacheEntry
	}{
		{"empty", cudaspec.SpeculationCacheEntry{}},
		{"bonus zero", cudaspec.SpeculationCacheEntry{BonusToken: 0}},
		{"bonus", cudaspec.SpeculationCacheEntry{BonusToken: 3}},
		{"draft path", cudaspec.SpeculationCacheEntry{DraftPath: []int32{2}}},
		{"logits", cudaspec.SpeculationCacheEntry{Logits: [][]float32{{1}}}},
	}
	for _, key := range []struct {
		name    string
		key     cudaspec.SpeculationCacheKey
		wantErr error
	}{
		{"zero digest", zero, cudaspec.ErrDraftDigestRequired},
		{"digest", digest, nil},
	} {
		for _, tt := range contents {
			t.Run(key.name+"/"+tt.name, func(t *testing.T) {
				c := cudaspec.NewSpeculationCache(4)
				e := tt.entry
				e.Key = key.key
				err := c.Put(&e)
				if !errors.Is(err, key.wantErr) || (key.wantErr == nil) != (err == nil) {
					t.Fatalf("Put err = %v, want %v", err, key.wantErr)
				}
				got, ok := c.Get(key.key)
				if ok != (key.wantErr == nil) {
					t.Fatalf("Get hit = %v after Put err %v", ok, err)
				}
				if ok && got.BonusToken != tt.entry.BonusToken {
					t.Fatalf("BonusToken = %d, want %d", got.BonusToken, tt.entry.BonusToken)
				}
			})
		}
	}
	if err := cudaspec.NewSpeculationCache(1).Put(nil); err == nil || errors.Is(err, cudaspec.ErrDraftDigestRequired) {
		t.Fatalf("Put(nil) err = %v, want a nil entry error", err)
	}
}

func TestSpeculationCacheRefusalLeavesState(t *testing.T) {
	key := func(gen uint64, draft byte) cudaspec.SpeculationCacheKey {
		k := cudaspec.ComputeSpeculationKey([]byte("model"), []int32{1}, 1, 1, 1, "fp32", gen)
		k.DraftDigest = [32]byte{draft}
		return k
	}
	a, b := key(1, 1), key(2, 1)
	c := cudaspec.NewSpeculationCache(2)
	for _, k := range []cudaspec.SpeculationCacheKey{a, b} {
		if err := c.Put(&cudaspec.SpeculationCacheEntry{Key: k, BonusToken: int32(k.Generation)}); err != nil {
			t.Fatal(err)
		}
	}
	// Refused entries, including ones for keys that differ from cached keys
	// only by their zero draft digest, must neither evict nor replace.
	for _, k := range []cudaspec.SpeculationCacheKey{key(1, 0), key(2, 0), key(3, 0)} {
		for _, e := range []cudaspec.SpeculationCacheEntry{{Key: k}, {Key: k, BonusToken: 9}, {Key: k, DraftPath: []int32{9}}, {Key: k, Logits: [][]float32{{9}}}} {
			if err := c.Put(&e); !errors.Is(err, cudaspec.ErrDraftDigestRequired) {
				t.Fatalf("Put(%+v) err = %v, want %v", e, err, cudaspec.ErrDraftDigestRequired)
			}
		}
	}
	for _, k := range []cudaspec.SpeculationCacheKey{a, b} {
		if e, ok := c.Get(k); !ok || e.BonusToken != int32(k.Generation) {
			t.Fatalf("Get(generation %d) = %+v, %v after refusals", k.Generation, e, ok)
		}
	}
	if n := c.InvalidateGeneration(0); n != 0 {
		t.Fatalf("InvalidateGeneration(0) purged %d, want 0", n)
	}
	// Insertion order is unchanged: the next insert evicts a, not b.
	if err := c.Put(&cudaspec.SpeculationCacheEntry{Key: key(3, 1)}); err != nil {
		t.Fatal(err)
	}
	if _, ok := c.Get(a); ok {
		t.Fatal("oldest entry survived eviction")
	}
	if _, ok := c.Get(b); !ok {
		t.Fatal("newer entry evicted")
	}
	if n := c.InvalidateGeneration(3); n != 2 {
		t.Fatalf("InvalidateGeneration(3) purged %d, want 2", n)
	}
}

func TestSpeculationCacheZeroDraftDigestBonusReuse(t *testing.T) {
	// The bonus token depends on how many drafts were accepted, and so on
	// the draft weights. These cycles draft [2 3] and [2 0] under equal
	// zero-digest request keys and emit different bonus tokens.
	a1, r1 := cycleRequest(t, []float32{0, 0, 2, 0, 0, 0, 0, 3})
	a2, r2 := cycleRequest(t, []float32{0, 0, 2, 0, 5, 0, 0, 0})
	res1, err := runCycle(t, a1, r1)
	if err != nil {
		t.Fatal(err)
	}
	res2, err := runCycle(t, a2, r2)
	if err != nil {
		t.Fatal(err)
	}
	if r1.Key != r2.Key || res1.Outcome.BonusToken == res2.Outcome.BonusToken {
		t.Fatalf("fixture: keys equal %v, bonus %d and %d", r1.Key == r2.Key, res1.Outcome.BonusToken, res2.Outcome.BonusToken)
	}
	c := cudaspec.NewSpeculationCache(4)
	if err := c.Put(&cudaspec.SpeculationCacheEntry{Key: r1.Key, BonusToken: res1.Outcome.BonusToken}); !errors.Is(err, cudaspec.ErrDraftDigestRequired) {
		t.Fatalf("Put without draft digest = %v, want ErrDraftDigestRequired", err)
	}
	if e, ok := c.Get(r2.Key); ok {
		t.Fatalf("reused bonus %d across draft weights; cycle drafted under them emitted %d", e.BonusToken, res2.Outcome.BonusToken)
	}
}
