package hopper

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// Launch is a variant's tunable launch configuration. Zero fields do not
// apply to the variant's kernel.
type Launch struct {
	Block        int `json:"block"`                    // threads per block
	Warps        int `json:"warps"`                    // warps per block
	Stages       int `json:"stages,omitempty"`         // cp.async ring depth
	SplitK       int `json:"split_k,omitempty"`        // K splits (SkinnyGEMM)
	SplitKV      int `json:"split_kv,omitempty"`       // keys per split (PagedAttention)
	BM           int `json:"bm,omitempty"`             // output tile rows (TileGEMM)
	BN           int `json:"bn,omitempty"`             // output tile columns (TileGEMM)
	RowsPerBlock int `json:"rows_per_block,omitempty"` // W rows per block (SkinnyGEMM)
	Unroll       int `json:"unroll,omitempty"`         // loads in flight (SkinnyGEMM)
	SharedBytes  int `json:"shared_bytes,omitempty"`   // dynamic shared memory
}

// A Variant is one emitted kernel configuration with the metadata an
// offline tuner needs: Key names the problem (op and shape, the same for
// every variant of it), Config names the tunables, and Source is a
// self-contained translation unit whose host entry is
//
//	void <Entry>(<Params>, cudaStream_t stream)
//
// and, for ctypes, extern "C" <Entry>_c with the same parameters.
type Variant struct {
	Name   string   `json:"name"`
	Op     string   `json:"op"`
	Key    string   `json:"key"`
	Config string   `json:"config"`
	Launch Launch   `json:"launch"`
	Entry  string   `json:"entry"`
	Params []string `json:"params"`
	SHA256 string   `json:"sha256"` // of Source
	Source string   `json:"-"`
}

func newVariant(name, op, key string, l Launch, params []string, src string) Variant {
	v := Variant{Name: name, Op: op, Key: key, Launch: l, Entry: name + "_launch", Params: params}
	v.Config = l.config()
	names := paramNames(params)
	src += fmt.Sprintf("extern \"C\" void %s_c(%s, cudaStream_t stream) {\n    %s(%s, stream);\n}\n",
		v.Entry, strings.Join(params, ", "), v.Entry, strings.Join(names, ", "))
	v.Source = src
	sum := sha256.Sum256([]byte(src))
	v.SHA256 = hex.EncodeToString(sum[:])
	return v
}

func (l Launch) config() string {
	var parts []string
	add := func(k string, v int) {
		if v != 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", k, v))
		}
	}
	add("block", l.Block)
	add("warps", l.Warps)
	add("stages", l.Stages)
	add("split_k", l.SplitK)
	add("split_kv", l.SplitKV)
	add("bm", l.BM)
	add("bn", l.BN)
	add("rows_per_block", l.RowsPerBlock)
	add("unroll", l.Unroll)
	return strings.Join(parts, ",")
}

func paramNames(params []string) []string {
	names := make([]string, len(params))
	for i, p := range params {
		f := strings.Fields(p)
		names[i] = f[len(f)-1]
	}
	return names
}

func epilogueKey(e Lambda, extras []Operand) string {
	s := "epi=" + e.String()
	for _, o := range extras {
		s += fmt.Sprintf(",%s:%v:%v", o.Name, o.DType, o.Broadcast)
	}
	return s
}

// Variant returns g's variant; Params start with int m.
func (g SkinnyGEMM) Variant() (Variant, error) {
	g, err := g.norm()
	if err != nil {
		return Variant{}, err
	}
	src, err := g.Source()
	if err != nil {
		return Variant{}, err
	}
	key := fmt.Sprintf("gemm/N=%d/K=%d/w=%v/out=%v/%s", g.N, g.K, g.Weight, g.Out, epilogueKey(g.Epilogue, g.Extras))
	l := Launch{Block: 32 * g.Warps, Warps: g.Warps, SplitK: g.SplitK, RowsPerBlock: g.RowsPerBlock, Unroll: g.Unroll}
	return newVariant(g.Name, "skinny_gemm", key, l, append([]string{"int m"}, g.params()...), src), nil
}

// Variant returns g's variant; Params start with int m (and, when
// Grouped, int max_rows, int experts).
func (g TileGEMM) Variant() (Variant, error) {
	g, err := g.norm()
	if err != nil {
		return Variant{}, err
	}
	src, err := g.Source()
	if err != nil {
		return Variant{}, err
	}
	key := fmt.Sprintf("gemm/N=%d/K=%d/w=%v/out=%v/%s", g.N, g.K, g.Weight, g.Out, epilogueKey(g.Epilogue, g.Extras))
	if g.Grouped || g.Gather {
		key += fmt.Sprintf("/grouped=%t/gather=%t", g.Grouped, g.Gather)
	}
	l := Launch{Block: 256, Warps: 8, Stages: g.Stages, BM: g.BM, BN: g.BN, SharedBytes: g.SharedBytes()}
	head := []string{"int m"}
	if g.Grouped {
		head = append(head, "int max_rows", "int experts")
	}
	return newVariant(g.Name, "tile_gemm", key, l, append(head, g.params()[1:]...), src), nil
}

// Variant returns a's variant; Params start with int batch, int max_keys.
func (a PagedAttention) Variant() (Variant, error) {
	a, _, err := a.norm()
	if err != nil {
		return Variant{}, err
	}
	src, err := a.Source()
	if err != nil {
		return Variant{}, err
	}
	key := fmt.Sprintf("paged_attention/Hq=%d/Hkv=%d/qlen=%d/window=%d/page=%d/q=%v/out=%v/qscale=%s/smscale=%s",
		a.Hq, a.Hkv, a.QLen, a.Window, a.Page, a.Q, a.Out, floatLiteral(a.QScale), floatLiteral(a.SMScale))
	if a.Gate {
		key += fmt.Sprintf("/gate=%d:%d:%s", a.GateStride, a.GateOffset, floatLiteral(a.GateEps))
	}
	l := Launch{Block: 128, Warps: 4, SplitKV: a.Chunk}
	return newVariant(a.Name, "paged_attention", key, l, append([]string{"int batch", "int max_keys"}, a.params()...), src), nil
}

// SweepSkinnyGEMM returns the valid variants of base over rows per block
// {1, 2, 4}, warps {4, 8, 16}, split-K {1, 2, 4, 8} and unroll {2, 4},
// named base.Name + "_r<R>w<W>s<S>u<U>".
func SweepSkinnyGEMM(base SkinnyGEMM) ([]Variant, error) {
	var vs []Variant
	for _, r := range []int{1, 2, 4} {
		for _, w := range []int{4, 8, 16} {
			for _, s := range []int{1, 2, 4, 8} {
				for _, u := range []int{2, 4} {
					g := base
					g.Name = fmt.Sprintf("%s_r%dw%ds%du%d", base.Name, r, w, s, u)
					g.RowsPerBlock, g.Warps, g.SplitK, g.Unroll = r, w, s, u
					v, err := g.Variant()
					if err != nil {
						continue
					}
					vs = append(vs, v)
				}
			}
		}
	}
	if len(vs) == 0 {
		return nil, fmt.Errorf("sweep %s: no valid variant", base.Name)
	}
	return vs, nil
}

// SweepTileGEMM returns the valid variants of base over BM, BN in {64, 128}
// and 2 to 4 stages, named base.Name + "_m<BM>n<BN>s<S>".
func SweepTileGEMM(base TileGEMM) ([]Variant, error) {
	var vs []Variant
	for _, bm := range []int{64, 128} {
		for _, bn := range []int{64, 128} {
			for s := 2; s <= 4; s++ {
				g := base
				g.Name = fmt.Sprintf("%s_m%dn%ds%d", base.Name, bm, bn, s)
				g.BM, g.BN, g.Stages = bm, bn, s
				v, err := g.Variant()
				if err != nil {
					continue
				}
				vs = append(vs, v)
			}
		}
	}
	if len(vs) == 0 {
		return nil, fmt.Errorf("sweep %s: no valid variant", base.Name)
	}
	return vs, nil
}

// SweepPagedAttention returns base's variants over keys per split
// {128, 256, 512, 1024}, named base.Name + "_kv<C>".
func SweepPagedAttention(base PagedAttention) ([]Variant, error) {
	var vs []Variant
	for _, c := range []int{128, 256, 512, 1024} {
		a := base
		a.Name = fmt.Sprintf("%s_kv%d", base.Name, c)
		a.Chunk = c
		v, err := a.Variant()
		if err != nil {
			return nil, err
		}
		vs = append(vs, v)
	}
	return vs, nil
}

// A Route sends calls whose first argument (m, or batch for attention) is
// at most Max to Variant. Max 0 means no bound.
type Route struct {
	Max     int
	Variant Variant
}

// Dispatcher returns one translation unit holding every routed variant and
//
//	void <name>(<params>, cudaStream_t stream)
//
// (also extern "C" <name>_c) which calls the first route whose Max bounds
// the first argument. Every variant's parameters must be a subsequence of
// the longest one's, which the dispatcher takes; for example a split-K
// SkinnyGEMM's trailing workspace is passed only to it.
func Dispatcher(name string, routes []Route) (string, error) {
	if !cIdent(name) || len(routes) == 0 {
		return "", fmt.Errorf("dispatcher %q: bad name or no routes", name)
	}
	var params []string
	for _, r := range routes {
		if len(r.Variant.Params) > len(params) {
			params = r.Variant.Params
		}
	}
	names := paramNames(params)
	var b strings.Builder
	seen := make(map[string]bool)
	for _, r := range routes {
		if !seen[r.Variant.Name] {
			seen[r.Variant.Name] = true
			fmt.Fprintf(&b, "// variant %s %s %s sha256=%s\n", r.Variant.Name, r.Variant.Key, r.Variant.Config, r.Variant.SHA256)
			b.WriteString(r.Variant.Source)
		}
	}
	fmt.Fprintf(&b, "\ninline void %s(%s, cudaStream_t stream) {\n", name, strings.Join(params, ", "))
	for i, r := range routes {
		args, err := subsequence(r.Variant.Params, params, names)
		if err != nil {
			return "", fmt.Errorf("dispatcher %s: variant %s: %w", name, r.Variant.Name, err)
		}
		call := fmt.Sprintf("%s(%s, stream); return;", r.Variant.Entry, strings.Join(args, ", "))
		if r.Max > 0 {
			fmt.Fprintf(&b, "    if (%s <= %d) { %s }\n", names[0], r.Max, call)
		} else {
			fmt.Fprintf(&b, "    %s\n", call)
			if i != len(routes)-1 {
				return "", fmt.Errorf("dispatcher %s: unbounded route %s is not last", name, r.Variant.Name)
			}
		}
	}
	b.WriteString("}\n")
	fmt.Fprintf(&b, "extern \"C\" void %s_c(%s, cudaStream_t stream) {\n    %s(%s, stream);\n}\n",
		name, strings.Join(params, ", "), name, strings.Join(names, ", "))
	return b.String(), nil
}

func subsequence(sub, full, names []string) ([]string, error) {
	var args []string
	j := 0
	for _, p := range sub {
		for j < len(full) && full[j] != p {
			j++
		}
		if j == len(full) {
			return nil, fmt.Errorf("parameter %q not in dispatcher parameters", p)
		}
		args = append(args, names[j])
		j++
	}
	return args, nil
}

// WriteVariants writes each variant to dir/<name>.cu and the metadata of
// all of them to dir/manifest.json.
func WriteVariants(dir string, vs []Variant) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("write variants: %w", err)
	}
	names := make(map[string]bool)
	for _, v := range vs {
		if names[v.Name] {
			return fmt.Errorf("write variants: duplicate name %s", v.Name)
		}
		names[v.Name] = true
		if err := os.WriteFile(filepath.Join(dir, v.Name+".cu"), []byte(v.Source), 0o644); err != nil {
			return fmt.Errorf("write variants: %w", err)
		}
	}
	sorted := slices.Clone(vs)
	slices.SortFunc(sorted, func(a, b Variant) int { return strings.Compare(a.Name, b.Name) })
	data, err := json.MarshalIndent(sorted, "", "  ")
	if err != nil {
		return fmt.Errorf("write variants: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), append(data, '\n'), 0o644); err != nil {
		return fmt.Errorf("write variants: %w", err)
	}
	return nil
}
