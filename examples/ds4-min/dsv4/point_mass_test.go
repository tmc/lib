package dsv4

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/sampler"
)

type pointMassCase struct {
	name  string
	p     []float32
	draft int
	seed  uint64
	draws []float32
}

type pointMassTestRandom struct {
	rng      sampler.RNG
	scripted []float32
	calls    int
}

func (r *pointMassTestRandom) Float32() float32 {
	r.calls++
	if len(r.scripted) != 0 {
		draw := r.scripted[0]
		r.scripted = r.scripted[1:]
		return draw
	}
	return r.rng.Float32()
}

func pointMassResidualVector() []float32 {
	p := []float32{math.Float32frombits(0x3f000000)}
	for range 8 {
		p = append(p, math.Float32frombits(0x33000001))
	}
	return append(p, math.Float32frombits(0x3efffff7), math.Float32frombits(0x32fffff0), 0)
}

func TestPointMassPinnedC(t *testing.T) {
	fixture, err := os.ReadFile("testdata/point_mass_11e8cad.c")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(fixture)); got != "09e4570bdceac8c828f03796e06dd41bc90d2c27dfa9625a98c9c6f140f4e903" {
		t.Fatalf("pinned fixture hash changed: %s", got)
	}
	for _, body := range []struct{ name, hash string }{
		{"RNG", "593386de00cd0d29d38cad6ca1ab9e52bc6dd7464a048580ddcc5defffce0096"},
		{"SAMPLE", "755fe9cbe95a8db974ffa2611c306cc0b344478ef9684bba771341a7e40c4bef"},
		{"ACCEPT", "193b17a37bcad7437b449cfe98d56072e4a2233a1fdc0923181c18e83258b91f"},
		{"REPLACEMENT", "74f33ade89c7e283975738d9430b096135e91439577d7388a45392cf75e7a5a1"},
	} {
		start := "/* BEGIN PINNED " + body.name + " */\n"
		end := "/* END PINNED " + body.name + " */"
		_, after, ok := strings.Cut(string(fixture), start)
		if !ok {
			t.Fatal("missing pinned body", body.name)
		}
		text, _, ok := strings.Cut(after, end)
		if !ok {
			t.Fatal("missing pinned end", body.name)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256([]byte(text))); got != body.hash {
			t.Fatalf("literal %s body hash changed: %s", body.name, got)
		}
	}
	compiler, err := exec.LookPath("clang")
	if err != nil {
		t.Fatal("pinned point-mass control requires clang")
	}
	binary := filepath.Join(t.TempDir(), "point-mass")
	if out, err := exec.Command(compiler, "-std=c11", "-O2", "-ffp-contract=off", "testdata/point_mass_11e8cad.c", "-lm", "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile point-mass reference: %v\n%s", err, out)
	}
	cases := []pointMassCase{
		{"probability one", []float32{0, 1}, 1, 7, nil},
		{"probability zero", []float32{0, 1}, 0, 7, nil},
		{"accept equality", []float32{.5, .25, .25}, 0, 7, []float32{.5}},
		{"raw float32 acceptance mass", []float32{.5, .5 + 1.0/(1<<24)}, 0, 7, []float32{.5}},
		{"accept above", []float32{.5, .25, .25}, 0, 7, []float32{.5 + 1.0/(1<<24), .25}},
		{"accept below", []float32{.5, .25, .25}, 0, 7, []float32{.5 - 1.0/(1<<24)}},
		{"replacement equality", []float32{.5, .5, 0}, 2, 7, []float32{.5}},
		{"masked zeros", []float32{0, 0, .5, .5}, 1, 7, []float32{0}},
		{"float32 sum boundary", []float32{.5, 1.0 / (1 << 25), .5, 0}, 3, 7, []float32{.5}},
		{"terminal residual fallback normalized", pointMassResidualVector(), 11, 7, []float32{math.Float32frombits(0x3f7fffff)}},
		{"filtered replacement", []float32{.2, 0, .3, .5}, 0, 7, []float32{.9, .375}},
		// These exercise literal C arithmetic fallback guards, not an admitted
		// normalized sampler distribution.
		{"no residual mass", []float32{0, 0, 0}, 1, 7, nil},
		{"overflow fallback best tie", []float32{math.MaxFloat32, math.MaxFloat32, 0}, 2, 7, nil},
	}
	for draft := range 3 {
		for seed := uint64(0); seed < 2000; seed++ {
			cases = append(cases, pointMassCase{"seeded", []float32{.62, .27, .11}, draft, seed, nil})
		}
	}
	for _, n := range []int{17, 513} {
		logits := make([]float32, n)
		for i := range logits {
			logits[i] = float32(i%19) / 4
		}
		p, err := sampler.New(n).Probabilities(logits, sampler.Params{Temperature: 1, TopP: .95, MinP: .05})
		if err != nil {
			t.Fatal(err)
		}
		for seed := uint64(0); seed < 64; seed++ {
			cases = append(cases, pointMassCase{"filtered vocabulary", p, int(seed) % n, seed, nil})
		}
	}
	var input strings.Builder
	for _, tc := range cases {
		fmt.Fprintf(&input, "%d %d %d %d", len(tc.p), tc.draft, tc.seed, len(tc.draws))
		for _, draw := range tc.draws {
			fmt.Fprintf(&input, " %08x", math.Float32bits(draw))
		}
		for _, p := range tc.p {
			fmt.Fprintf(&input, " %08x", math.Float32bits(p))
		}
		input.WriteByte('\n')
	}
	cmd := exec.Command(binary)
	cmd.Stdin = strings.NewReader(input.String())
	output, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	rows := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(rows) != len(cases) {
		t.Fatalf("C results=%d want=%d", len(rows), len(cases))
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var wantToken, wantAccepted, wantDraws int
			var wantState uint64
			if n, err := fmt.Sscan(rows[i], &wantToken, &wantAccepted, &wantState, &wantDraws); err != nil || n != 4 {
				t.Fatalf("malformed C row %d: %q", i, rows[i])
			}
			random := &pointMassTestRandom{rng: sampler.RNG{State: tc.seed}, scripted: append([]float32(nil), tc.draws...)}
			before := append([]float32(nil), tc.p...)
			token, accepted, err := PointMassVerify(tc.p, tc.draft, random)
			if err != nil || token != wantToken || accepted != (wantAccepted != 0) || random.rng.State != wantState || random.calls != wantDraws || !reflect.DeepEqual(tc.p, before) {
				t.Fatalf("case %d %s draft=%d seed=%d: Go token=%d accepted=%v state=%x draws=%d err=%v; C %s", i, tc.name, tc.draft, tc.seed, token, accepted, random.rng.State, random.calls, err, rows[i])
			}
		})
	}
}

func TestPointMassInvalidInput(t *testing.T) {
	for _, tc := range []struct {
		name  string
		p     []float32
		draft int
	}{
		{"empty", nil, 0}, {"draft below", []float32{1}, -1}, {"draft above", []float32{1}, 1},
		{"negative", []float32{.5, -.5}, 0}, {"nan", []float32{.5, float32(math.NaN())}, 0}, {"inf", []float32{.5, float32(math.Inf(1))}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &pointMassTestRandom{rng: sampler.RNG{State: 7}}
			if _, _, err := PointMassVerify(tc.p, tc.draft, r); err == nil || r.calls != 0 {
				t.Fatalf("invalid input consumed randomness: %v", err)
			}
		})
	}
	if _, _, err := PointMassVerify([]float32{1}, 0, nil); err == nil {
		t.Fatal("nil random accepted")
	}
	for i, draw := range []float32{-1, 1, float32(math.NaN()), float32(math.Inf(1))} {
		t.Run("draw"+strconv.Itoa(i), func(t *testing.T) {
			r := &pointMassTestRandom{scripted: []float32{draw}}
			if _, _, err := PointMassVerify([]float32{.5, .5}, 0, r); err == nil || r.calls != 1 {
				t.Fatalf("bad draw: %v", err)
			}
		})
	}
}

func ExamplePointMassVerify() {
	random := sampler.RNG{State: 7}
	token, accepted, _ := PointMassVerify([]float32{0, 1}, 1, &random)
	fmt.Println(token, accepted, random.State)
	// Output: 1 true 7
}
