package dsv41

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
)

func TestRoutedMidPinnedC(t *testing.T) {
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Skip("set DS41_ORACLE for literal source control")
	}
	source, err := exec.Command("git", "-C", repo, "show", "11e8cad4473716a41baf4d62e92ed9764a00fce6:ds4_cuda.cu").Output()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(source)) != "c420f92fecf07af2113bad5b9d74801c5e1d10020f1094120d9f31ba430fc0b5" {
		t.Fatal("routed source hash mismatch")
	}
	f := literalFunction(t, string(source), "swiglu_kernel")
	const main = `int main(){float g[32],u[32],out[32];if(fread(g,4,32,stdin)!=32||fread(u,4,32,stdin)!=32)return 2;blockIdx.x=0;blockDim.x=32;for(unsigned i=0;i<32;i++){threadIdx.x=i;swiglu_kernel(out,g,u,32,10.0f,.375f);}return fwrite(out,4,32,stdout)==32?0:3;}`
	gate, up, want := make([]float32, 32), make([]float32, 32), make([]float32, 32)
	for i := range gate {
		gate[i] = float32(i) - 16
		up[i] = float32(i)*1.7 - 20
	}
	if err := routedMid(want, gate, up, .375); err != nil {
		t.Fatal(err)
	}
	input := append(append([]float32(nil), gate...), up...)
	got := runLiteral(t, operationPrefix+f+main, input)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("weighted mid %d: %g != %g", i, got[i], want[i])
		}
	}
	mutant := strings.Replace(f, "s * u * weight", "s * u", 1)
	if mutant == f {
		t.Fatal("mid weight mutant missing")
	}
	if equalBits(runLiteral(t, operationPrefix+mutant+main, input), want) {
		t.Fatal("mid weight mutant not detected")
	}
}

type patchedReader struct{ patches map[int64][]byte }

func (r patchedReader) ReadAt(dst []byte, offset int64) (int, error) {
	clear(dst)
	for base, data := range r.patches {
		start, end := max(offset, base), min(offset+int64(len(dst)), base+int64(len(data)))
		if start < end {
			copy(dst[start-offset:end-offset], data[start-base:end-base])
		}
	}
	return len(dst), nil
}

func TestHCLaggedPreAndBF16Boundary(t *testing.T) {
	g := modelFixture(t)
	patches := make(map[int64][]byte)
	for _, tensor := range g.Tensors {
		if tensor.Name == "blk.0.attn_norm.weight" || tensor.Name == "blk.0.ffn_norm.weight" {
			data := make([]byte, 5120*4)
			for i := 0; i < 5120; i++ {
				binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(1))
			}
			patches[int64(g.DataOffset+tensor.Offset)] = data
		}
	}
	m, err := Bind(patchedReader{patches}, int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	residual := make([]float32, 4*5120)
	for i := 0; i < 5120; i++ {
		residual[i] = float32(math.Sin(float64(i) * .3))
		residual[5120+i] = float32(math.Cos(float64(i) * .2))
		residual[10240+i] = .25
		residual[15360+i] = -.5
	}
	pre := []float32{1, 0, 0, 0}
	got, split, err := m.hcInput(0, false, residual, pre)
	if err != nil {
		t.Fatal(err)
	}
	want := append([]float32(nil), residual[:5120]...)
	for i := range want {
		want[i] = BF16(want[i])
	}
	if err := dsv4cpu.RMSNorm(want, want, nil, 1e-20); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		want[i] = BF16(want[i])
	}
	if !equalBits(got, want) {
		t.Fatal("HC used current split instead of lagged pre")
	}
	changed, _, err := m.hcInput(0, false, residual, split[:4])
	if err != nil {
		t.Fatal(err)
	}
	if equalBits(got, changed) {
		t.Fatal("pre-weight intervention insensitive")
	}
	// FFN collapse receives attention pre, despite computing a new FFN split.
	ffn, _, err := m.hcInput(0, true, residual, split[:4])
	if err != nil {
		t.Fatal(err)
	}
	if !equalBits(ffn, changed) {
		t.Fatal("FFN lagged-pre contract mismatch")
	}
}

func TestStateAdmissionResetAndIsolation(t *testing.T) {
	g := modelFixture(t)
	m, err := Bind(new(countReader), int64(g.Size), g)
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.NewState(2)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.NewState(2)
	if err != nil {
		t.Fatal(err)
	}
	if a.kv == b.kv || a.engram == b.engram || &a.pool[0] == &b.pool[0] {
		t.Fatal("request state shared")
	}
	for _, token := range []int32{-1, 129280} {
		if _, err := a.Eval(token); err == nil {
			t.Fatal("invalid token admitted")
		}
		if !a.valid || a.position != 0 {
			t.Fatal("admission error corrupted state")
		}
	}
	a.valid = false
	if _, err := a.Eval(0); err == nil {
		t.Fatal("invalid execution state admitted")
	}
	a.Reset()
	if !a.valid || a.position != 0 || !b.valid {
		t.Fatal("reset or isolation failed")
	}
	if _, err := m.NewState(0); err == nil {
		t.Fatal("invalid context admitted")
	}
}

func TestIndexScoresLaneReduction(t *testing.T) {
	q, weights, keys := make([]float32, 32*128), make([]float32, 32), make([]float32, 2*128)
	for i := range q {
		q[i] = 1
	}
	for i := range weights {
		weights[i] = 1
	}
	for i := 0; i < 128; i++ {
		keys[i] = 1
		keys[128+i] = -1
	}
	got, err := indexScores(q, weights, keys)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []float32{64, 0}) {
		t.Fatal(got)
	}
}

func ExampleModel_NewState() {
	var model Model
	_, err := model.NewState(2048)
	fmt.Println(err)
	// Output: invalid V4.1 model
}

func ExampleState_Reset() {
	var state State
	state.Reset()
	_, err := state.Eval(0)
	fmt.Println(err)
	// Output: invalid V4.1 state or token
}

func ExampleState_Eval() {
	var state State
	_, err := state.Eval(0)
	fmt.Println(err)
	// Output: invalid V4.1 state or token
}
