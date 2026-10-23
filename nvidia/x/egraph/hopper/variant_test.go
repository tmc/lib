package hopper

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

func TestSweepMetadata(t *testing.T) {
	skinny, err := SweepSkinnyGEMM(SkinnyGEMM{Name: "qkv", N: 2304, K: 8192, MaxM: 8, Weight: WeightBF16, Out: BF16})
	if err != nil {
		t.Fatal(err)
	}
	tile, err := SweepTileGEMM(TileGEMM{Name: "qkv", N: 2304, K: 8192, Weight: WeightBF16, Out: BF16})
	if err != nil {
		t.Fatal(err)
	}
	attn, err := SweepPagedAttention(PagedAttention{Name: "attn", Hq: 16, Hkv: 2, QLen: 1, Window: 2048, Page: 1, Q: E4M3, Out: F32})
	if err != nil {
		t.Fatal(err)
	}
	if len(tile) != 12 {
		t.Errorf("%d tile variants, want 12", len(tile))
	}
	for _, set := range [][]Variant{skinny, tile, attn} {
		configs := make(map[string]bool)
		for _, v := range set {
			if v.Key != set[0].Key {
				t.Errorf("%s: key %q differs from %q", v.Name, v.Key, set[0].Key)
			}
			if configs[v.Config] {
				t.Errorf("%s: duplicate config %q", v.Name, v.Config)
			}
			configs[v.Config] = true
			sum := sha256.Sum256([]byte(v.Source))
			if v.SHA256 != hex.EncodeToString(sum[:]) {
				t.Errorf("%s: stale hash", v.Name)
			}
			if v.Launch.Block == 0 || v.Launch.Warps*32 != v.Launch.Block {
				t.Errorf("%s: launch %+v", v.Name, v.Launch)
			}
		}
	}
	if tile[0].Key != skinny[0].Key {
		t.Errorf("skinny and tile GEMM of one shape have keys %q and %q", skinny[0].Key, tile[0].Key)
	}
	dir := t.TempDir()
	if err := WriteVariants(dir, append(skinny, attn...)); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var back []Variant
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if len(back) != len(skinny)+len(attn) {
		t.Fatalf("manifest has %d variants", len(back))
	}
	src, err := os.ReadFile(filepath.Join(dir, back[0].Name+".cu"))
	if err != nil {
		t.Fatal(err)
	}
	if sum := sha256.Sum256(src); hex.EncodeToString(sum[:]) != back[0].SHA256 {
		t.Errorf("%s.cu does not match its manifest hash", back[0].Name)
	}
}

// TestDispatcherEmulated routes small m to a split-K SkinnyGEMM and large m
// to a TileGEMM through the generated dispatcher and launchers.
func TestDispatcherEmulated(t *testing.T) {
	const n, k = 64, 512
	small, err := SkinnyGEMM{Name: "proj_small", N: n, K: k, MaxM: 16, Weight: WeightBF16, Out: F32, SplitK: 2, RowsPerBlock: 2}.Variant()
	if err != nil {
		t.Fatal(err)
	}
	large, err := TileGEMM{Name: "proj_large", N: n, K: k, Weight: WeightBF16, Out: F32, BM: 64, BN: 64}.Variant()
	if err != nil {
		t.Fatal(err)
	}
	src, err := Dispatcher("proj", []Route{{Max: 16, Variant: small}, {Variant: large}})
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(61, 62))
	w := GEMMWeights{BF16: randBF16(r, n*k, 0.05)}
	ws := SkinnyGEMM{Name: "s", N: n, K: k, MaxM: 16, Weight: WeightBF16, Out: F32, SplitK: 2, RowsPerBlock: 2}.WorkspaceBytes()
	for _, m := range []int{5, 40} {
		x := randBF16(r, m*k, 1)
		res := runEmu(t, src, fmt.Sprintf("proj(%d, x, w, out, ws, nullptr)", m),
			emuBuf{"x", "uint16_t", u16Bytes(x)},
			emuBuf{"w", "uint16_t", u16Bytes(w.BF16)},
			emuBuf{"out", "float", make([]byte, 4*m*n)},
			emuBuf{"ws", "float", make([]byte, ws)})
		want, err := TileGEMM{Name: "ref", N: n, K: k, Weight: WeightBF16, Out: F32, BN: 64}.Reference(m, x, nil, nil, []GEMMWeights{w}, nil)
		if err != nil {
			t.Fatal(err)
		}
		got := bytesF32(res["out"])
		if i, ok := closeTo(got, want, 1e-5, 1e-6); !ok {
			t.Fatalf("m=%d: out[%d] = %g, want %g", m, i, got[i], want[i])
		}
	}
}

func TestDispatcherErrors(t *testing.T) {
	a, err := TileGEMM{Name: "a", N: 64, K: 64, Weight: WeightBF16, Out: F32, BN: 64}.Variant()
	if err != nil {
		t.Fatal(err)
	}
	b, err := PagedAttention{Name: "b", Hq: 2, Hkv: 1, QLen: 1, Page: 1, Q: E4M3, Out: F32}.Variant()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Dispatcher("d", []Route{{Variant: a}, {Max: 4, Variant: a}}); err == nil {
		t.Error("unbounded route before a bounded one: no error")
	}
	if _, err := Dispatcher("d", []Route{{Max: 4, Variant: a}, {Variant: b}}); err == nil {
		t.Error("mismatched parameters: no error")
	}
}

func ExampleSweepTileGEMM() {
	vs, _ := SweepTileGEMM(TileGEMM{Name: "proj", N: 8192, K: 1024, Weight: WeightBF16, Out: BF16})
	fmt.Println(len(vs), vs[0].Name, vs[0].Key)
	fmt.Println(vs[0].Config)
	// Output:
	// 12 proj_m64n64s2 gemm/N=8192/K=1024/w=bf16/out=bf16/epi=$0
	// block=256,warps=8,stages=2,bm=64,bn=64
}
