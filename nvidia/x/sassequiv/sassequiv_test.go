package sassequiv

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// The testdata listings are cuobjdump -sass output, with encodings
// removed, of kernels hopper's generator emits (its testdata/golden),
// compiled by nvcc 13.4 for sm_90a. corpus names whole listings;
// NAME.rN.sass is one function of a listing compiled at -maxrregcount N.
var corpus = []string{"rms_norm_gate", "v_store", "q_norm_rope", "k_norm_rope_store", "moe8_expert_sum"}

func parse(t *testing.T, text string) *cudasass.Disassembly {
	t.Helper()
	d, err := cudasass.Parse(strings.NewReader(text))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func listing(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name + ".sass")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestRegisterLimitNotDifferent compares functions whose register limit
// changed their allocation, spilling and schedule, never their values.
// Neither may be reported Different. Both are Unknown: ptxas computes
// their indexes in different forms and reorders their stores.
func TestRegisterLimitNotDifferent(t *testing.T) {
	for _, pair := range [][2]string{
		{"moe8_qkv_gate3.r255", "moe8_qkv_gate3.r96"},
		{"moe8_router_hilo1.r255", "moe8_router_hilo1.r64"},
	} {
		before, after := parse(t, listing(t, pair[0])), parse(t, listing(t, pair[1]))
		for _, r := range CompareAll(before, after) {
			if r.Verdict == Different {
				t.Errorf("%s against %s: %v", pair[0], pair[1], r)
			}
		}
	}
}

// equalPairs are compilations of one kernel that must compare Equal.
// NAME.lb.sass and NAME.lb1.sass are functions of egraph-owner's
// generated decode kernels compiled with __launch_bounds__(n) and
// __launch_bounds__(n, 1); the minimum block count lets ptxas use more
// registers. NAME.rN.sass is compiled at -maxrregcount N. Both change
// allocation, spilling and schedule, never values.
var equalPairs = [][2]string{
	{"pre_expert_norm_cluster.lb", "pre_expert_norm_cluster.lb1"},
	{"post_expert_sum_cluster.lb", "post_expert_sum_cluster.lb1"},
	{"stock_pre_norm_4096_cluster.lb", "stock_pre_norm_4096_cluster.lb1"},
	{"stock_boundary_attn_cluster.lb", "stock_boundary_attn_cluster.lb1"},
	{"stock_boundary_ffn_cluster.lb", "stock_boundary_ffn_cluster.lb1"},
	{"stock_boundary_ffn_slice1024_cluster.lb", "stock_boundary_ffn_slice1024_cluster.lb1"},
	{"stock_post_expert_sum_4096_cluster.lb", "stock_post_expert_sum_4096_cluster.lb1"},
	{"moe8_router_hilo4.r255", "moe8_router_hilo4.r96"},
	{"moe8_router_hilo5.r255", "moe8_router_hilo5.r96"},
	{"moe8_qkv_gate7.r255", "moe8_qkv_gate7.r96"},
	{"attn_boundary_router_cluster.lb", "attn_boundary_router_cluster.lb1"},
}

// TestNegativeControls edits the second listing of an equal pair in ways
// that change what it computes; each must compare Different.
func TestNegativeControls(t *testing.T) {
	tests := []struct {
		name, pair, old, new string
	}{
		{"fadd of an earlier partial sum", "attn_boundary_router_cluster",
			"/*0e40*/                   FADD R17, R16, R17 ;",
			"/*0e40*/                   FADD R17, R15, R17 ;"},
		{"rms epsilon", "attn_boundary_router_cluster",
			"FFMA R18, R17, R18, 9.9999999392252902908e-09 ;",
			"FFMA R18, R17, R18, 9.9999997473787516356e-06 ;"},
		{"router epsilon", "attn_boundary_router_cluster",
			"FFMA R18, R17, R18, 9.9999997473787516356e-06 ;",
			"FFMA R18, R17, R18, 9.9999999392252902908e-09 ;"},
		{"fadd of an earlier partial sum", "pre_expert_norm_cluster",
			"/*0790*/                   FADD R19, R18, R19 ;",
			"/*0790*/                   FADD R19, R17, R19 ;"},
	}
	for _, tt := range tests {
		t.Run(tt.pair+"/"+tt.name, func(t *testing.T) {
			after := listing(t, tt.pair+".lb1")
			if n := strings.Count(after, tt.old); n != 1 {
				t.Fatalf("%q occurs %d times", tt.old, n)
			}
			after = strings.Replace(after, tt.old, tt.new, 1)
			r := CompareAll(parse(t, listing(t, tt.pair+".lb")), parse(t, after))[0]
			if r.Verdict != Different {
				t.Errorf("got %v, want different", r)
			}
		})
	}
}

func TestEqualPairs(t *testing.T) {
	for _, pair := range equalPairs {
		before, after := parse(t, listing(t, pair[0])), parse(t, listing(t, pair[1]))
		for _, r := range CompareAll(before, after) {
			if r.Verdict != Equal {
				t.Errorf("%s against %s: %v", pair[0], pair[1], r)
			}
		}
	}
}

// TestEqualPairMutantsDiffer mutates the second listing of each equal
// pair, which exercises the rules that make the pair equal: none of the
// mutants may compare Equal to the first. Every eleventh match of each
// mutation is tried.
func TestEqualPairMutantsDiffer(t *testing.T) {
	stats := map[string]map[Verdict]int{}
	for _, pair := range equalPairs {
		before := parse(t, listing(t, pair[0]))
		lines := strings.Split(listing(t, pair[1]), "\n")
		for _, mu := range mutations {
			if stats[mu.name] == nil {
				stats[mu.name] = map[Verdict]int{}
			}
			n := 0
			for i, line := range lines {
				m := mu.re.FindStringSubmatch(line)
				if m == nil {
					continue
				}
				// Line 227 of this listing reads R22 where the
				// branch at 0x08f0 skips its first write.
				undef := pair[1] == "stock_post_expert_sum_4096_cluster.lb1" && i+1 == 227
				if n++; n%11 != 0 && !undef {
					continue
				}
				mutated := append([]string(nil), lines...)
				mutated[i] = mu.edit(m)
				if mutated[i] == line {
					continue
				}
				r := CompareAll(before, parse(t, strings.Join(mutated, "\n")))[0]
				stats[mu.name][r.Verdict]++
				if r.Verdict == Equal {
					t.Errorf("%s: %s at line %d (%s -> %s): %v", pair[1], mu.name, i+1, strings.TrimSpace(line), strings.TrimSpace(mutated[i]), r)
				}
			}
		}
	}
	for _, mu := range mutations {
		s := stats[mu.name]
		t.Logf("%-22s different %3d  unknown %3d  equal %d", mu.name, s[Different], s[Unknown], s[Equal])
	}
}

// A mutation edits one instruction of a listing.
type mutation struct {
	name string
	re   *regexp.Regexp
	edit func(m []string) string // from the submatches, the new line
}

var instRE = `^(\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P[0-9T]\s+)?)`

var mutations = []mutation{
	{"fadd to fmul", regexp.MustCompile(instRE + `FADD((?:\.[A-Z]+)*) (.*)$`),
		func(m []string) string { return m[1] + "FMUL" + m[2] + " " + m[3] }},
	{"fmul to fadd", regexp.MustCompile(instRE + `FMUL((?:\.[A-Z]+)*) (.*)$`),
		func(m []string) string { return m[1] + "FADD" + m[2] + " " + m[3] }},
	{"ffma addend negated", regexp.MustCompile(instRE + `(FFMA(?:\.[A-Z]+)* R\d+, [^,]+, [^,]+, )(R\d+)(.*)$`),
		func(m []string) string { return m[1] + m[2] + "-" + m[3] + m[4] }},
	{"fp source register", regexp.MustCompile(instRE + `((?:FADD|FMUL|FFMA)(?:\.[A-Z]+)* R\d+, )(-?)R(\d+)(.*)$`),
		func(m []string) string { return m[1] + m[2] + m[3] + "R" + bump(m[4]) + m[5] }},
	{"fp immediate", regexp.MustCompile(instRE + `((?:FADD|FMUL|FFMA|FSETP)[^;]*, )(-?\d+(?:\.\d+)?(?:e-?\d+)?)( ;.*|,.*)$`),
		func(m []string) string {
			var f float64
			fmt.Sscan(m[3], &f)
			return m[1] + m[2] + fmt.Sprint(f*1.5+1) + m[4]
		}},
	{"load offset", regexp.MustCompile(instRE + `(LDG[^;]*\[R\d+\.64)(\+0x[0-9a-f]+)?(\][^;]*;.*)$`),
		func(m []string) string { return m[1] + m[2] + "+0x404" + m[4] }},
	{"mufu function", regexp.MustCompile(instRE + `MUFU\.(RSQ|RCP|EX2)(.*)$`),
		func(m []string) string {
			return m[1] + "MUFU." + map[string]string{"RSQ": "RCP", "RCP": "EX2", "EX2": "RSQ"}[m[2]] + m[3]
		}},
	{"store offset", regexp.MustCompile(instRE + `(ST[GS][^;]*\[R\d+(?:\.64)?)(\+0x[0-9a-f]+)?(\][^;]*;.*)$`),
		func(m []string) string { return m[1] + m[2] + "+0x404" + m[4] }},
	{"branch condition negated", regexp.MustCompile(`^(\s+/\*[0-9a-f]+\*/\s+@)(!?)(U?P[0-6]\s+BRA.*)$`),
		func(m []string) string {
			if m[2] == "!" {
				return m[1] + m[3]
			}
			return m[1] + "!" + m[3]
		}},
}

func bump(n string) string {
	var k int
	fmt.Sscan(n, &k)
	return fmt.Sprint(k ^ 1)
}

// TestMutantsDiffer checks that no single-instruction change to a value
// computation compares Equal. Each line a mutation matches is mutated in
// turn.
func TestMutantsDiffer(t *testing.T) {
	stats := map[string]map[Verdict]int{}
	for _, name := range corpus {
		before := parse(t, listing(t, name))
		lines := strings.Split(listing(t, name), "\n")
		for _, mu := range mutations {
			if stats[mu.name] == nil {
				stats[mu.name] = map[Verdict]int{}
			}
			for i, line := range lines {
				m := mu.re.FindStringSubmatch(line)
				if m == nil {
					continue
				}
				mutated := append([]string(nil), lines...)
				mutated[i] = mu.edit(m)
				if mutated[i] == line {
					continue
				}
				after := parse(t, strings.Join(mutated, "\n"))
				for _, r := range CompareAll(before, after) {
					if r.Function != after.Functions[0].Name {
						continue
					}
					stats[mu.name][r.Verdict]++
					if r.Verdict == Unknown && os.Getenv("SASSEQUIV_DEBUG") != "" {
						t.Logf("%s %s line %d: %s", name, mu.name, i+1, r.Reason)
					}
					if r.Verdict == Equal {
						t.Errorf("%s: %s at line %d (%s -> %s) compared equal", name, mu.name, i+1, strings.TrimSpace(line), strings.TrimSpace(mutated[i]))
					}
				}
			}
		}
	}
	for _, mu := range mutations {
		s := stats[mu.name]
		t.Logf("%-22s different %3d  unknown %3d  equal %d", mu.name, s[Different], s[Unknown], s[Equal])
		if s[Different]+s[Unknown] == 0 {
			t.Errorf("%s: matched no instruction", mu.name)
		}
	}
}

var loadRE = regexp.MustCompile(`^(\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P[0-9T]\s+)?(LD[GS][.A-Z0-9]*) [^,]+, )([^ ;]+)( ;.*)$`)

// TestLoadSwapDiffers exchanges the addresses of two loads of one kind,
// which flow analysis that treats every load as an anonymous leaf cannot
// see.
func TestLoadSwapDiffers(t *testing.T) {
	var n, different int
	for _, name := range corpus {
		before := parse(t, listing(t, name))
		lines := strings.Split(listing(t, name), "\n")
		for i := range lines {
			mi := loadRE.FindStringSubmatch(lines[i])
			if mi == nil {
				continue
			}
			for j := i + 1; j < len(lines) && j < i+40; j++ {
				mj := loadRE.FindStringSubmatch(lines[j])
				if mj == nil || mj[2] != mi[2] || mj[3] == mi[3] {
					continue
				}
				mutated := append([]string(nil), lines...)
				mutated[i] = mi[1] + mj[3] + mi[4]
				mutated[j] = mj[1] + mi[3] + mj[4]
				after := parse(t, strings.Join(mutated, "\n"))
				r := CompareAll(before, after)[0]
				n++
				switch r.Verdict {
				case Equal:
					t.Errorf("%s: swapping the addresses at lines %d and %d compared equal", name, i+1, j+1)
				case Different:
					different++
				}
				break
			}
		}
	}
	t.Logf("%d load swaps: %d different, %d unknown", n, different, n-different)
	if n == 0 || different == 0 {
		t.Errorf("%d load swaps, %d different", n, different)
	}
}

func TestSelfIsEqual(t *testing.T) {
	for _, name := range append(corpus, "moe8_qkv_gate3.r96", "moe8_router_hilo1.r64") {
		for _, r := range CompareAll(parse(t, listing(t, name)), parse(t, listing(t, name))) {
			if r.Verdict != Equal {
				t.Errorf("%s against itself: %v", name, r)
			}
		}
	}
}
