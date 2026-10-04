package cudafuse

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

type mhcFixture struct {
	name                      string
	config                    MHCConfig
	logits, x, post, residual []float32
}
type mhcAnswer struct {
	mix, out                    []float32
	rowResidual, columnResidual float64
}

// The oracle uses separate matrices and FP64 sums, no kernel helper, no
// denominator scaling, and an explicit matrix contraction rather than flattened
// device indexing. Round only once when handing expected values to lane 69.
func mhcOracle(f mhcFixture) mhcAnswer {
	a := mhcAnswer{mix: make([]float32, len(f.logits)), out: make([]float32, len(f.residual))}
	for b := 0; b < f.config.Batch; b++ {
		var m [4][4]float64
		for r := range m {
			max := math.Inf(-1)
			for k := range m[r] {
				max = math.Max(max, float64(f.logits[b*16+r*4+k]))
			}
			sum := 0.0
			for k := range m[r] {
				m[r][k] = math.Exp(float64(f.logits[b*16+r*4+k]) - max)
				sum += m[r][k]
			}
			for k := range m[r] {
				m[r][k] = m[r][k]/sum + float64(f.config.Epsilon)
			}
		}
		normalize := func(rows bool) {
			var sums [4]float64
			for r := range m {
				for k := range m[r] {
					if rows {
						sums[r] += m[r][k]
					} else {
						sums[k] += m[r][k]
					}
				}
			}
			for r := range m {
				for k := range m[r] {
					q := k
					if rows {
						q = r
					}
					m[r][k] /= sums[q] + float64(f.config.Epsilon)
				}
			}
		}
		normalize(false)
		for round := 1; round < f.config.Repeat; round++ {
			normalize(true)
			normalize(false)
		}
		for r := range m {
			rs, cs := 0.0, 0.0
			for k := range m[r] {
				rs += m[r][k]
				cs += m[k][r]
				a.mix[b*16+r*4+k] = float32(m[r][k])
			}
			a.rowResidual = math.Max(a.rowResidual, math.Abs(rs-1))
			a.columnResidual = math.Max(a.columnResidual, math.Abs(cs-1))
		}
		for n := 0; n < 4; n++ {
			for c := 0; c < f.config.Width; c++ {
				v := float64(f.x[b*f.config.Width+c]) * float64(f.post[b*4+n])
				for stream := 0; stream < 4; stream++ {
					v += m[stream][n] * float64(f.residual[(b*4+stream)*f.config.Width+c])
				}
				a.out[(b*4+n)*f.config.Width+c] = float32(v)
			}
		}
	}
	return a
}

func mhcFixtures() []mhcFixture {
	var fixtures []mhcFixture
	for _, repeat := range []int{1, 10, 16} {
		for _, width := range []int{1, 31, 32, 33, 129} {
			c := MHCConfig{Batch: 2, Width: width, Repeat: repeat, Epsilon: 1e-6}
			if width == 129 {
				c.Batch = 4
			}
			if width == 33 && repeat == 10 {
				c.Epsilon = 0.07
			}
			f := mhcFixture{name: fmt.Sprintf("nonsymmetric-w%d-r%d", width, repeat), config: c, logits: make([]float32, c.Batch*16), x: make([]float32, c.Batch*width), post: make([]float32, c.Batch*4), residual: make([]float32, c.Batch*4*width)}
			for i := range f.logits {
				f.logits[i] = float32((i*7+i/4*3)%19-9) * 0.7
			}
			for i := range f.x {
				f.x[i] = float32(i%11-5) * 0.13
			}
			for i := range f.post {
				f.post[i] = float32(i%7-2) * 0.17
			}
			for i := range f.residual {
				f.residual[i] = float32((i*13+i/width*3)%31-15) * 0.11
			}
			fixtures = append(fixtures, f)
		}
	}
	for _, mode := range []string{"zero", "extreme"} {
		f := fixtures[0]
		f.name = mode
		f.logits = make([]float32, len(f.logits))
		if mode == "extreme" {
			for i := range f.logits {
				if i%3 == 0 {
					f.logits[i] = 10000
				} else {
					f.logits[i] = -10000
				}
			}
		}
		fixtures = append(fixtures, f)
	}
	return fixtures
}

func mhcPolicy(f mhcFixture, source, target string, n int) cudatest.CUDAOutputPolicy {
	input := sha256.New()
	for _, xs := range [][]float32{f.logits, f.x, f.post, f.residual} {
		for _, x := range xs {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(x))
			input.Write(b[:])
		}
	}
	return cudatest.CUDAOutputPolicy{Target: target, DType: "fp32", MathMode: "FP32; --fmad=false; no fast math; scaled positive denominator", SourceDigest: fmt.Sprintf("%x", sha256.Sum256([]byte(source))), InputDigest: fmt.Sprintf("%x", input.Sum(nil)), Elements: n, Absolute: 3e-6, Relative: 3e-5, NaN: "reject", Infinity: "reject", Zero: "equal"}
}

func mhcCompare(f mhcFixture, got mhcAnswer, source, target string) error {
	want := mhcOracle(f)
	for _, pair := range [][2][]float32{{got.mix, want.mix}, {got.out, want.out}} {
		if err := cudatest.CompareCUDAOutputFP32(pair[0], pair[1], mhcPolicy(f, source, target, len(pair[0]))); err != nil {
			return err
		}
	}
	return nil
}

func TestMHCSinkhornReference(t *testing.T) {
	for _, f := range mhcFixtures() {
		t.Run(f.name, func(t *testing.T) {
			sources, err := MHCSource(f.config)
			if err != nil {
				t.Fatal(err)
			}
			got := mhcRunHost(t, mhcBuildHost(t, sources), f)
			if err := mhcCompare(f, got, sources[0].Source+sources[1].Source, "host-emulation"); err != nil {
				t.Fatal(err)
			}
			want := mhcOracle(f)
			t.Logf("FP64 finite-round residual row=%g column=%g", want.rowResidual, want.columnResidual)
		})
	}
}

func TestMHCEpsilonAndRepeatAdmission(t *testing.T) {
	valid := MHCConfig{1, 1, 1, 1e-6}
	for _, repeat := range []int{1, 10, 16} {
		c := valid
		c.Repeat = repeat
		if _, err := MHCSource(c); err != nil {
			t.Fatal(err)
		}
	}
	for _, eps := range []float32{math.SmallestNonzeroFloat32, math.MaxFloat32} {
		c := valid
		c.Epsilon = eps
		if _, err := MHCSource(c); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []MHCConfig{{1, 1, 0, 1e-6}, {1, 1, 17, 1e-6}, {0, 1, 1, 1e-6}, {1, 0, 1, 1e-6}, {math.MaxInt32, 1, 1, 1e-6}, {1, math.MaxInt32, 1, 1e-6}, {1, 1, 1, 0}, {1, 1, 1, -1}, {1, 1, 1, float32(math.NaN())}, {1, 1, 1, float32(math.Inf(1))}} {
		if _, err := MHCSource(c); err == nil {
			t.Fatalf("invalid config accepted: %+v", c)
		}
	}
}

func mhcRanges(c MHCConfig) []mhcRange {
	var out []mhcRange
	ptr := uint64(4096)
	for _, n := range []int{c.Batch * 16, c.Batch * c.Width, c.Batch * 4, c.Batch * 4 * c.Width} {
		out = append(out, mhcRange{ptr, uint64(n) * 4, n})
		ptr += uint64(n)*4 + 64
	}
	return out
}

func TestMHCWidthTailsAndAliasing(t *testing.T) {
	for _, width := range []int{1, 31, 32, 33, 129} {
		c := MHCConfig{4, width, 10, 1e-6}
		ranges := mhcRanges(c)
		if err := mhcValidateRanges(c, ranges); err != nil {
			t.Fatal(err)
		}
		for i := range ranges {
			for j := 0; j < i; j++ {
				bad := append([]mhcRange(nil), ranges...)
				bad[i].ptr = bad[j].ptr
				if mhcValidateRanges(c, bad) == nil {
					t.Fatal("forbidden alias accepted")
				}
			}
		}
		bad := append([]mhcRange(nil), ranges...)
		bad[3].length--
		if mhcValidateRanges(c, bad) == nil {
			t.Fatal("short residual accepted")
		}
	}
}

type mhcMutation struct {
	name                      string
	stage                     int
	old, replacement, fixture string
}

var mhcMutations = []mhcMutation{
	{"swapped-rounds", 0, "mhc_normalize(a,0);\n        mhc_normalize(a,1);", "mhc_normalize(a,1);\n        mhc_normalize(a,0);", "nonsymmetric-w33-r10"},
	{"epsilon-omitted", 0, "/sum+MHC_EPS;", "/sum;", "nonsymmetric-w33-r10"},
	{"unstable-softmax", 0, "expf(logits[b*16+r*4+k]-mx)", "expf(logits[b*16+r*4+k])", "extreme"},
	{"mix-transposed", 1, "mix[b*16+m*4+n]", "mix[b*16+n*4+m]", "nonsymmetric-w1-r1"},
	{"tail-dropped", 1, "i >= (unsigned int)count", "i >= (unsigned int)(count-count%32)", "nonsymmetric-w33-r10"},
}

func mhcMutated(c MHCConfig, m mhcMutation) ([2]FusedKernel, error) {
	s, err := MHCSource(c)
	if err != nil {
		return s, err
	}
	if strings.Count(s[m.stage].Source, m.old) != 1 {
		return s, fmt.Errorf("mhc mutant %s did not match exactly once", m.name)
	}
	s[m.stage].Source = strings.Replace(s[m.stage].Source, m.old, m.replacement, 1)
	return s, nil
}
func mhcTestMutants(t *testing.T, names ...string) {
	for _, m := range mhcMutations {
		selected := false
		for _, name := range names {
			selected = selected || name == m.name
		}
		if !selected {
			continue
		}
		t.Run(m.name, func(t *testing.T) {
			for _, f := range mhcFixtures() {
				if f.name != m.fixture {
					continue
				}
				s, err := mhcMutated(f.config, m)
				if err != nil {
					t.Fatal(err)
				}
				got := mhcRunHost(t, mhcBuildHost(t, s), f)
				if err := mhcCompare(f, got, s[0].Source+s[1].Source, "host-emulation"); err == nil {
					t.Fatal("behavioral mutant survived")
				} else {
					t.Logf("completed output rejected: %v", err)
				}
				return
			}
			t.Fatal("missing mutant fixture")
		})
	}
}
func TestMHCNormalizationOrder(t *testing.T) {
	mhcTestMutants(t, "swapped-rounds", "epsilon-omitted", "unstable-softmax")
}
func TestMHCPostOrientation(t *testing.T) { mhcTestMutants(t, "mix-transposed", "tail-dropped") }

func TestMHCSchema(t *testing.T) {
	sources, _ := MHCSource(MHCConfig{1, 33, 10, 1e-6})
	for _, s := range sources {
		schema, err := mhcSchema(append(append([]string(nil), s.Inputs...), s.Outputs...))
		if err != nil {
			t.Fatal(err)
		}
		if err := schema.Validate(); err != nil {
			t.Fatal(err)
		}
		if len(schema.Params) != len(s.Inputs)+len(s.Outputs)+1 {
			t.Fatal("wrong ABI")
		}
		for i, p := range schema.Params {
			if i == len(schema.Params)-1 {
				if p.Type != "s32" || p.ByteWidth != 4 {
					t.Fatal(p)
				}
			} else if !p.IsPointer || p.ByteWidth != 8 {
				t.Fatal(p)
			}
		}
	}
}
