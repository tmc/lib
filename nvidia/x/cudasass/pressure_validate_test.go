package cudasass_test

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// malformedPressureHeaders are ".text." label suffixes Parse refuses as
// malformed. Before ParseLifeRangeCount used the same validator, it keyed
// each one's instructions under a guessed kernel name.
var malformedPressureHeaders = []string{
	"/]'//\t}'\u00a0\u200b",
	"\"\\\\\"}\"a//b\"/* c */}[[abi:x] ",
	"((f<\"a//b\"auto:1:))",
	"\"a//b\":: auto:1)",
	"[/* c */L8{\"a//b\"",
	"f<f<'/'f<auto:1L8{\"a//b\"",
	"\"a//b\"{",
	"\"a//b\"::L8{",
}

// literalCommentPressureNames are ".text." label suffixes with "//" inside a
// literal, or followed by a "//" comment, that Parse accepts, each with the
// kernel name Parse reads.
var literalCommentPressureNames = []struct{ header, name string }{
	{"x\"::?//$\t_\"", "x\"::?//$\t_\""},
	{"SL8<L8{\"a//b\"}> h3<L8{\"a//b\"}>()", "SL8<L8{\"a//b\"}> h3<L8{\"a//b\"}>()"},
	{"SLN<L8{\"a//b\"}, (3 > 1) ? 1 : 2> k3<3>()", "SLN<L8{\"a//b\"}, (3 > 1) ? 1 : 2> k3<3>()"},
	{"void m3<3>(SLN<L8{\"a//b\"}, (3 > 1) ? 1 : 2>)", "void m3<3>(SLN<L8{\"a//b\"}, (3 > 1) ? 1 : 2>)"},
	{"f<L8{\"a//b\"}>()", "f<L8{\"a//b\"}>()"},
	{"f<L8{\"a\\\"//\"}>() // c", "f<L8{\"a\\\"//\"}>()"},
	{"f('\\'//') // c", "f('\\'//')"},
	{"f<'//'>()", "f<'//'>()"},
	{"SL8<L8{\"a//b\"}>", "SL8<L8{\"a//b\"}>"},
	{"f(\"a\\\"//\")", "f(\"a\\\"//\")"},
	{"k \"//\"", "k \"//\""},
	{"AT[abi:tg] kt<S<5>{\"'//'\"}>()", "AT[abi:tg] kt<S<5>{\"'//'\"}>()"},
	{"AT[abi:tg] kt<S<5>{\"'//'\"}>() // c", "AT[abi:tg] kt<S<5>{\"'//'\"}>()"},
	{"AT[abi:tg] kt<S<5>{\"\\\"//\\\"\"}>()", "AT[abi:tg] kt<S<5>{\"\\\"//\\\"\"}>()"},
	{"AT[abi:tg] kt<S<5>{\"\\\"//\\\"\"}>() // c", "AT[abi:tg] kt<S<5>{\"\\\"//\\\"\"}>()"},
	{"AT[abi:tg] kt<S<7>{\"//>`|$\"}>()", "AT[abi:tg] kt<S<7>{\"//>`|$\"}>()"},
	{"AT[abi:tg] kt<S<7>{\"//>`|$\"}>() // c", "AT[abi:tg] kt<S<7>{\"//>`|$\"}>()"},
	{"decltype(fp ? 1 : 2) ns::Q<S<3>{\"//\"}, long>::h<bool>(bool)", "decltype(fp ? 1 : 2) ns::Q<S<3>{\"//\"}, long>::h<bool>(bool)"},
	{"decltype(fp ? 1 : 2) ns::Q<S<3>{\"//\"}, long>::h<bool>(bool) // c", "decltype(fp ? 1 : 2) ns::Q<S<3>{\"//\"}, long>::h<bool>(bool)"},
	{"ns::Q<S<3>{\"//\"}, int>::g(int)", "ns::Q<S<3>{\"//\"}, int>::g(int)"},
	{"ns::Q<S<3>{\"//\"}, int>::g(int) // c", "ns::Q<S<3>{\"//\"}, int>::g(int)"},
	{"ns::Q<S<3>{\"//\"}, int>::operator()(int) const", "ns::Q<S<3>{\"//\"}, int>::operator()(int) const"},
	{"ns::Q<S<3>{\"//\"}, int>::operator()(int) const // c", "ns::Q<S<3>{\"//\"}, int>::operator()(int) const"},
	{"void fs<FS{\"a//b\"}>()", "void fs<FS{\"a//b\"}>()"},
	{"void fs<FS{\"a//b\"}>() // c", "void fs<FS{\"a//b\"}>()"},
	{"void k2<S<3>{\"a:\"}, S<4>{\"//b\"}>()", "void k2<S<3>{\"a:\"}, S<4>{\"//b\"}>()"},
	{"void k2<S<3>{\"a:\"}, S<4>{\"//b\"}>() // c", "void k2<S<3>{\"a:\"}, S<4>{\"//b\"}>()"},
	{"void k<S<3>{\"//\"}>()", "void k<S<3>{\"//\"}>()"},
	{"void k<S<3>{\"//\"}>() // c", "void k<S<3>{\"//\"}>()"},
	{"void ns::k<S<3>{\"//\"}>()", "void ns::k<S<3>{\"//\"}>()"},
	{"void ns::k<S<3>{\"//\"}>() // c", "void ns::k<S<3>{\"//\"}>()"},
	{"void ns::k<S<4>{\"//\\\"\"}>()", "void ns::k<S<4>{\"//\\\"\"}>()"},
	{"void ns::k<S<4>{\"//\\\"\"}>() // c", "void ns::k<S<4>{\"//\\\"\"}>()"},
	{"void ns::k<S<5>{\"'//'\"}>()", "void ns::k<S<5>{\"'//'\"}>()"},
	{"void ns::k<S<5>{\"'//'\"}>() // c", "void ns::k<S<5>{\"'//'\"}>()"},
	{"void ns::k<S<5>{\"\\\"//\\\"\"}>()", "void ns::k<S<5>{\"\\\"//\\\"\"}>()"},
	{"void ns::k<S<5>{\"\\\"//\\\"\"}>() // c", "void ns::k<S<5>{\"\\\"//\\\"\"}>()"},
	{"void ns::k<S<7>{\"//>`|$\"}>()", "void ns::k<S<7>{\"//>`|$\"}>()"},
	{"void ns::k<S<7>{\"//>`|$\"}>() // c", "void ns::k<S<7>{\"//>`|$\"}>()"},
	{"void ns::k<S<7>{\"\\\"a//b\\\"\"}>()", "void ns::k<S<7>{\"\\\"a//b\\\"\"}>()"},
	{"void ns::k<S<7>{\"\\\"a//b\\\"\"}>() // c", "void ns::k<S<7>{\"\\\"a//b\\\"\"}>()"},
	{"void ns::k<S<7>{\"a\\\\\\\"//\\\"\"}>()", "void ns::k<S<7>{\"a\\\\\\\"//\\\"\"}>()"},
	{"void ns::k<S<7>{\"a\\\\\\\"//\\\"\"}>() // c", "void ns::k<S<7>{\"a\\\\\\\"//\\\"\"}>()"},
	{"void ns::k<S<9>{\"\\\\|<:\\n//[\"}>()", "void ns::k<S<9>{\"\\\\|<:\\n//[\"}>()"},
	{"void ns::k<S<9>{\"\\\\|<:\\n//[\"}>() // c", "void ns::k<S<9>{\"\\\\|<:\\n//[\"}>()"},
}

// realPressureNames are ".text." label suffixes of the shapes nvdisasm
// prints, each with the kernel name Parse reads.
var realPressureNames = []struct{ header, name string }{
	{"_Z6kernelPf:", "_Z6kernelPf"},
	{"_Z10vector_addPfS_S_i:", "_Z10vector_addPfS_S_i"},
	{"vector_add:", "vector_add"},
	{"scale(float *, int):", "scale(float *, int)"},
	{"ns::scale<int>(int *, int):", "ns::scale<int>(int *, int)"},
	{"ns::Q<S<3>{}, int>::operator()(int) const:", "ns::Q<S<3>{}, int>::operator()(int) const"},
	{"void k<(3 > 1) ? 1 : 2>():", "void k<(3 > 1) ? 1 : 2>()"},
	{"_Z6kernelPf: // note", "_Z6kernelPf"},
}

// lifeRangeListing returns a one-kernel --life-range-mode count listing
// under the label ".text."+header.
func lifeRangeListing(header string) string {
	return ".text." + header + "\n" +
		"        /*0000*/                   EXIT ;  // |   1   |        |        |         |\n"
}

// plainListing returns the same kernel without the live-register table.
func plainListing(header string) string {
	return ".text." + header + "\n        /*0000*/                   EXIT ;\n"
}

func TestLifeRangeCountRefusesMalformedLabel(t *testing.T) {
	if len(malformedPressureHeaders) != 8 {
		t.Fatalf("len(malformedPressureHeaders) = %d, want 8", len(malformedPressureHeaders))
	}
	for _, h := range malformedPressureHeaders {
		t.Run(fmt.Sprintf("%q", h), func(t *testing.T) {
			if d, err := cudasass.ParseString(plainListing(h)); err == nil {
				t.Fatalf("Parse accepted %q: functions %+v", h, d.Functions)
			}
			press, err := cudasass.ParseLifeRangeCount(strings.NewReader(lifeRangeListing(h)))
			if err == nil {
				t.Fatalf("ParseLifeRangeCount accepted %q: keys %q", h, slices.Sorted(maps.Keys(press)))
			}
			if press != nil {
				t.Errorf("ParseLifeRangeCount(%q) map = %v, want nil", h, press)
			}
			if !strings.Contains(err.Error(), "malformed label") {
				t.Errorf("ParseLifeRangeCount(%q) error = %v, want malformed label", h, err)
			}

			// A malformed label after a well-formed kernel is refused too,
			// not folded into the kernel before it.
			listing := lifeRangeListing("k:") + lifeRangeListing(h)
			if press, err := cudasass.ParseLifeRangeCount(strings.NewReader(listing)); err == nil {
				t.Fatalf("ParseLifeRangeCount accepted k then %q: keys %q", h, slices.Sorted(maps.Keys(press)))
			}
		})
	}
}

func TestLifeRangeCountLiteralCommentNames(t *testing.T) {
	if len(literalCommentPressureNames) != 45 {
		t.Fatalf("len(literalCommentPressureNames) = %d, want 45", len(literalCommentPressureNames))
	}
	for _, tt := range literalCommentPressureNames {
		checkPressureKey(t, tt.header, tt.name)
	}
}

func TestLifeRangeCountRealNames(t *testing.T) {
	for _, tt := range realPressureNames {
		checkPressureKey(t, tt.header, tt.name)
	}
}

// checkPressureKey checks that Parse reads header as one kernel named name
// and that ParseLifeRangeCount keys that kernel's counts under the same name.
func checkPressureKey(t *testing.T, header, name string) {
	t.Helper()
	d, err := cudasass.ParseString(plainListing(header))
	if err != nil {
		t.Errorf("Parse(%q): %v", header, err)
		return
	}
	if len(d.Functions) != 1 || d.Functions[0].Name != name {
		t.Errorf("Parse(%q) functions = %+v, want one named %q", header, d.Functions, name)
		return
	}
	press, err := cudasass.ParseLifeRangeCount(strings.NewReader(lifeRangeListing(header)))
	if err != nil {
		t.Errorf("ParseLifeRangeCount(%q): %v", header, err)
		return
	}
	want := map[string]map[uint64]cudasass.RegisterPressure{name: {0: {GPR: 1}}}
	if fmt.Sprint(press) != fmt.Sprint(want) {
		t.Errorf("ParseLifeRangeCount(%q) = %v, want %v", header, press, want)
		return
	}
	if err := cudasass.JoinPressure(d, press); err != nil {
		t.Errorf("JoinPressure(%q): %v", header, err)
	}
}

// TestLifeRangeCountAgreesWithParse checks, over every header above, that
// Parse and ParseLifeRangeCount refuse the same labels and otherwise read the
// same kernel names, alone and after another kernel.
func TestLifeRangeCountAgreesWithParse(t *testing.T) {
	headers := slices.Clone(malformedPressureHeaders)
	for _, tt := range literalCommentPressureNames {
		headers = append(headers, tt.header)
	}
	for _, tt := range realPressureNames {
		headers = append(headers, tt.header)
	}
	refused := 0
	for _, h := range headers {
		for _, prefix := range []string{"", "k:"} {
			plain, life := plainListing(h), lifeRangeListing(h)
			if prefix != "" {
				plain, life = plainListing(prefix)+plain, lifeRangeListing(prefix)+life
			}
			d, perr := cudasass.ParseString(plain)
			press, lerr := cudasass.ParseLifeRangeCount(strings.NewReader(life))
			if (perr == nil) != (lerr == nil) {
				t.Errorf("%q after %q: Parse error %v, ParseLifeRangeCount error %v", h, prefix, perr, lerr)
				continue
			}
			if perr != nil {
				refused++
				continue
			}
			var names []string
			for _, fn := range d.Functions {
				names = append(names, fn.Name)
			}
			slices.Sort(names)
			if keys := slices.Sorted(maps.Keys(press)); !slices.Equal(keys, names) {
				t.Errorf("%q after %q: ParseLifeRangeCount keys %q, Parse names %q", h, prefix, keys, names)
			}
		}
	}
	if want := 2 * len(malformedPressureHeaders); refused != want {
		t.Errorf("refused %d listings, want %d", refused, want)
	}
}

// pressureDirectives are .section directives, without ".section" and the
// blank after it, each with the kernel Parse reads under it, or "" where
// Parse refuses it as malformed. Before ParseLifeRangeCount used the same
// validator, it keyed each refused directive's instructions under a guessed
// kernel name: f<'a>(), k, f<"a>() and k"x.
var pressureDirectives = []struct{ directive, want string }{
	{`.text.f<'a>(),"ax",@progbits`, ""},
	{`.text.k, "a","ax",@progbits`, ""},
	{`.text.f<"a>()`, ""},
	{`.text.k"x`, ""},
	{`.text.k,"ax",@progbits`, "k"},
	{`.text._Z6kernelPf,"ax",@progbits`, "_Z6kernelPf"},
	{`.text.ns::scale<int>(int *, int),"ax",@progbits`, "ns::scale<int>(int *, int)"},
	{`.text.k`, "k"},
	{`.text.k, "a"`, "k"},
	{`.text.f<'a'>(),"ax",@progbits`, "f<'a'>()"},
	{`.text.k, "lit","ax",@progbits`, `k, "lit"`},
}

// TestLifeRangeCountDirectivesAgreeWithParse checks, over pressureDirectives
// and directiveNames, that Parse and ParseLifeRangeCount refuse the same
// .section directives with the same error and otherwise read the same kernel
// names, alone and after another kernel.
func TestLifeRangeCountDirectivesAgreeWithParse(t *testing.T) {
	rows := slices.Concat(pressureDirectives, directiveNames)
	const (
		exit     = "        /*0000*/                   EXIT ;\n"
		exitLife = "        /*0000*/                   EXIT ;  // |   1   |        |        |         |\n"
	)
	refused, wantRefused := 0, 0
	for i, tt := range rows {
		directive := "\t.section\t" + tt.directive + "\n"
		for _, prefix := range []string{"", "before:"} {
			plain, life := directive+exit, directive+exitLife
			if prefix != "" {
				plain, life = plainListing(prefix)+plain, lifeRangeListing(prefix)+life
			}
			d, perr := cudasass.ParseString(plain)
			press, lerr := cudasass.ParseLifeRangeCount(strings.NewReader(life))
			if tt.want == "" {
				wantRefused++
				want := fmt.Sprintf("cudasass: malformed section directive %q", strings.TrimSpace(directive))
				if perr == nil || perr.Error() != want {
					t.Errorf("%d: Parse(%q after %q) error = %v, want %q", i, tt.directive, prefix, perr, want)
				}
				if lerr == nil || lerr.Error() != want {
					t.Errorf("%d: ParseLifeRangeCount(%q after %q) = %q, %v, want error %q", i, tt.directive, prefix, slices.Sorted(maps.Keys(press)), lerr, want)
					continue
				}
				if press != nil {
					t.Errorf("%d: ParseLifeRangeCount(%q after %q) map = %v, want nil", i, tt.directive, prefix, press)
				}
				refused++
				continue
			}
			if perr != nil || lerr != nil {
				t.Errorf("%d: %q after %q: Parse error %v, ParseLifeRangeCount error %v", i, tt.directive, prefix, perr, lerr)
				continue
			}
			want := []string{tt.want}
			if prefix != "" {
				want = append(want, strings.TrimSuffix(prefix, ":"))
			}
			slices.Sort(want)
			var names []string
			for _, fn := range d.Functions {
				names = append(names, fn.Name)
			}
			slices.Sort(names)
			keys := slices.Sorted(maps.Keys(press))
			if !slices.Equal(names, want) || !slices.Equal(keys, want) {
				t.Errorf("%d: %q after %q: Parse names %q, ParseLifeRangeCount keys %q, want %q", i, tt.directive, prefix, names, keys, want)
			}
		}
	}
	if refused != wantRefused || refused == 0 {
		t.Errorf("ParseLifeRangeCount refused %d listings, want %d", refused, wantRefused)
	}
}
