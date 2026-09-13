package cudalowering

import (
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// A Pair binds a PTX kernel to the SASS function it lowered to.
type Pair struct {
	PTX  *cudaptx.Kernel
	SASS *cudasass.Function
}

// A Matching is the result of pairing a PTX module with a disassembly.
// A kernel that appears at one level only is listed as unmatched rather than
// dropped: a comparison that quietly pairs nothing cannot report a divergence.
type Matching struct {
	Pairs         []Pair
	UnmatchedPTX  []*cudaptx.Kernel
	UnmatchedSASS []*cudasass.Function
}

// Match pairs the kernels of m with the functions of d.
//
// Only .entry kernels with a body are considered. A .func device function is
// usually inlined into its callers and has no SASS function of its own, so
// pairing it would report an absence that means nothing.
//
// Names are matched first as written, then through [MatchKey]. A key that names
// more than one kernel on either side is not used, so two kernels are never
// paired on an ambiguous name.
func Match(m *cudaptx.Module, d *cudasass.Disassembly) *Matching {
	res := &Matching{}

	var kernels []*cudaptx.Kernel
	if m != nil {
		for _, k := range m.Kernels {
			if k != nil && k.Entry && k.Body {
				kernels = append(kernels, k)
			}
		}
	}
	var funcs []*cudasass.Function
	if d != nil {
		for _, fn := range d.Functions {
			if fn != nil {
				funcs = append(funcs, fn)
			}
		}
	}

	takenPTX := make(map[*cudaptx.Kernel]bool, len(kernels))
	takenSASS := make(map[*cudasass.Function]bool, len(funcs))

	byName := make(map[string]*cudasass.Function, len(funcs))
	for _, fn := range funcs {
		byName[fn.Name] = fn
	}
	for _, k := range kernels {
		fn := byName[k.Name]
		if fn == nil || takenSASS[fn] {
			continue
		}
		res.Pairs = append(res.Pairs, Pair{PTX: k, SASS: fn})
		takenPTX[k] = true
		takenSASS[fn] = true
	}

	ptxKeys := uniqueKeys(len(kernels), func(i int) (string, bool) {
		if takenPTX[kernels[i]] {
			return "", false
		}
		return MatchKey(kernels[i].Name), true
	})
	sassKeys := uniqueKeys(len(funcs), func(i int) (string, bool) {
		if takenSASS[funcs[i]] {
			return "", false
		}
		return MatchKey(funcs[i].Name), true
	})
	for _, k := range kernels {
		if takenPTX[k] {
			continue
		}
		key := MatchKey(k.Name)
		i, ok := ptxKeys[key]
		if !ok || kernels[i] != k {
			continue
		}
		j, ok := sassKeys[key]
		if !ok {
			continue
		}
		res.Pairs = append(res.Pairs, Pair{PTX: k, SASS: funcs[j]})
		takenPTX[k] = true
		takenSASS[funcs[j]] = true
	}

	for _, k := range kernels {
		if !takenPTX[k] {
			res.UnmatchedPTX = append(res.UnmatchedPTX, k)
		}
	}
	for _, fn := range funcs {
		if !takenSASS[fn] {
			res.UnmatchedSASS = append(res.UnmatchedSASS, fn)
		}
	}
	return res
}

// uniqueKeys maps each key produced by at to its index, dropping keys that more
// than one index produces.
func uniqueKeys(n int, at func(int) (string, bool)) map[string]int {
	res := make(map[string]int, n)
	dup := make(map[string]bool)
	for i := 0; i < n; i++ {
		key, ok := at(i)
		if !ok || key == "" {
			continue
		}
		if _, seen := res[key]; seen {
			dup[key] = true
			continue
		}
		res[key] = i
	}
	for key := range dup {
		delete(res, key)
	}
	return res
}

// MatchKey reduces a kernel symbol to the name the PTX and the SASS for one
// kernel agree on.
//
// nvcc writes the same Itanium-mangled symbol at both levels, so most pairs
// match on the symbol itself. The levels part company once the SASS has been
// through a demangler: cu++filt rewrites .text._Z10atomicDropPfPKfi as
// .text.atomicDrop(float *, const float *, int), which no longer resembles the
// PTX symbol. MatchKey strips either spelling down to the qualified function
// name, "atomicDrop" here, and leaves a name it does not recognize alone.
//
// Template arguments are dropped, so MatchKey("_Z3fooIiEvPT_") and
// MatchKey("foo<int>(int*)") are both "foo". A demangled template kernel is
// still the one case MatchKey cannot rescue on its own: cu++filt writes the
// return type ahead of the name, and the SASS section parser keeps only the
// first word of it.
func MatchKey(name string) string {
	name = strings.TrimSpace(name)
	if key, ok := mangledKey(name); ok {
		return key
	}
	if i := strings.IndexByte(name, '('); i >= 0 {
		name = name[:i]
	}
	if i := strings.IndexByte(name, '<'); i >= 0 {
		name = name[:i]
	}
	return strings.TrimSpace(name)
}

// mangledKey decodes the qualified name of an Itanium-mangled symbol, the only
// part of the encoding this package needs. It reports false for a name it
// cannot read as one, including the extern "C" names that carry no encoding.
func mangledKey(sym string) (string, bool) {
	rest, ok := strings.CutPrefix(sym, "_Z")
	if !ok {
		return "", false
	}
	nested := false
	if r, ok := strings.CutPrefix(rest, "N"); ok {
		nested = true
		rest = r
	}
	var parts []string
	for {
		name, r, ok := lengthPrefixed(rest)
		if !ok {
			break
		}
		parts = append(parts, name)
		rest = r
		if !nested {
			break
		}
	}
	if len(parts) == 0 {
		return "", false
	}
	return strings.Join(parts, "::"), true
}

// lengthPrefixed splits one <length><identifier> component off the front of s,
// returning the identifier and what follows it.
func lengthPrefixed(s string) (name, rest string, ok bool) {
	n := 0
	i := 0
	for ; i < len(s) && s[i] >= '0' && s[i] <= '9'; i++ {
		n = n*10 + int(s[i]-'0')
	}
	if i == 0 || n == 0 || i+n > len(s) {
		return "", s, false
	}
	return s[i : i+n], s[i+n:], true
}
