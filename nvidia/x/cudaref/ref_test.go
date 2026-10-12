package cudaref_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestRefMatMul(t *testing.T) {
	// A: 2x2 [[1, 2], [3, 4]]
	a := []float32{1, 2, 3, 4}
	// B: 2x2 [[5, 6], [7, 8]]
	b := []float32{5, 6, 7, 8}

	// C = A x B = [[19, 22], [43, 50]]
	c, err := cudaref.MatMul(a, b, 2, 2, 2)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	expected := []float32{19, 22, 43, 50}
	for i := range expected {
		if c[i] != expected[i] {
			t.Errorf("at %d: want %v, got %v", i, expected[i], c[i])
		}
	}
}

func TestRefSoftmax(t *testing.T) {
	src := []float32{1.0, 2.0, 3.0}
	res := cudaref.Softmax(src)

	var sum float32
	for _, v := range res {
		sum += v
	}

	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax sum != 1.0: %v", sum)
	}
}

func TestRefReductions(t *testing.T) {
	var warpVals [32]float32
	for i := range warpVals {
		warpVals[i] = float32(i + 1)
	}

	// 1 + 2 + ... + 32 = 32*33/2 = 528
	wSum := cudaref.WarpReduceSum(warpVals)
	if wSum != 528 {
		t.Errorf("expected 528, got %v", wSum)
	}
}

func TestNumericalEdgeCases(t *testing.T) {
	// Acceptance test: TestNumericalEdgeCases
	// 1. NaN propagation
	nan := float32(math.NaN())
	unaryRes, err := cudaref.ElementwiseUnary([]float32{nan, 1.0, -1.0}, "relu")
	if err != nil {
		t.Fatalf("ElementwiseUnary failed: %v", err)
	}
	if !math.IsNaN(float64(unaryRes[0])) {
		t.Errorf("expected NaN propagation in relu, got %v", unaryRes[0])
	}
	if unaryRes[1] != 1.0 || unaryRes[2] != 0.0 {
		t.Errorf("unexpected relu outputs: %v", unaryRes)
	}

	// 2. Infinities and Signed Zero in Division
	posZero := float32(0.0)
	negZero := float32(math.Copysign(0.0, -1.0))

	divRes, err := cudaref.ElementwiseBinary([]float32{1.0, 1.0, 0.0}, []float32{posZero, negZero, posZero}, "div")
	if err != nil {
		t.Fatalf("ElementwiseBinary div failed: %v", err)
	}
	if !math.IsInf(float64(divRes[0]), 1) {
		t.Errorf("expected +Inf for 1.0 / +0.0, got %v", divRes[0])
	}
	if !math.IsInf(float64(divRes[1]), -1) {
		t.Errorf("expected -Inf for 1.0 / -0.0, got %v", divRes[1])
	}
	if !math.IsNaN(float64(divRes[2])) {
		t.Errorf("expected NaN for 0.0 / 0.0, got %v", divRes[2])
	}

	// 3. Empty reduction sets via Evaluate
	gEmpty := cudair.NewGraph()
	emptyIn := gEmpty.AddInput("Empty", []int{2, 0}, cudair.Float32)
	sumNode := gEmpty.AddReduce("sum", emptyIn, 1) // reduce over extent 0
	maxNode := gEmpty.AddReduce("max", emptyIn, 1)
	_ = gEmpty.SetOutputs(sumNode, maxNode)

	emptyInputs := map[string][]float32{
		"Empty": {},
	}
	evalRes, err := cudaref.Evaluate(gEmpty, emptyInputs)
	if err != nil {
		t.Fatalf("Evaluate on empty reduction failed: %v", err)
	}
	// Policy: Empty sum -> 0.0
	for _, v := range evalRes[sumNode.Name] {
		if v != 0.0 {
			t.Errorf("empty sum reduction policy violated: want 0.0, got %v", v)
		}
	}
	// Policy: Empty max -> -Inf
	for _, v := range evalRes[maxNode.Name] {
		if !math.IsInf(float64(v), -1) {
			t.Errorf("empty max reduction policy violated: want -Inf, got %v", v)
		}
	}

	// 4. Fully masked attention row
	seqLen := 4
	headDim := 8
	scores := []float32{10.0, 20.0, 30.0, 40.0}
	values := make([]float32, seqLen*headDim)
	for i := range values {
		values[i] = 1.0
	}
	allTrueMask := []bool{true, true, true, true} // completely masked

	attnOut, err := cudaref.OnlineAttentionRow(scores, values, allTrueMask, seqLen, headDim, 2)
	if err != nil {
		t.Fatalf("OnlineAttentionRow failed: %v", err)
	}
	for i, v := range attnOut {
		if v != 0.0 || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("masked row policy violated at %d: want 0.0, got %v", i, v)
		}
	}
}

func TestExactActivationDerivatives(t *testing.T) {
	inputs := []float32{-20, -2, -1e-7, 0, 1e-7, 2, 20}
	relu, err := cudaref.ElementwiseUnary(inputs, "relu_grad")
	if err != nil {
		t.Fatal(err)
	}
	wantRelu := []float32{0, 0, 0, 0, 1, 1, 1}
	for i := range wantRelu {
		if relu[i] != wantRelu[i] {
			t.Errorf("relu_grad(%g) = %g, want %g", inputs[i], relu[i], wantRelu[i])
		}
	}

	gelu, err := cudaref.ElementwiseUnary(inputs, "gelu_grad")
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range inputs {
		x := float64(v)
		cdf := 0.5 * (1 + math.Erf(x/math.Sqrt2))
		pdf := math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
		want := float32(cdf + x*pdf)
		if math.Abs(float64(gelu[i]-want)) > 1e-7 {
			t.Errorf("gelu_grad(%g) = %.9g, want %.9g", v, gelu[i], want)
		}
	}

	nan := float32(math.NaN())
	for _, op := range []string{"relu_grad", "gelu_grad"} {
		got, err := cudaref.ElementwiseUnary([]float32{nan}, op)
		if err != nil || len(got) != 1 || !math.IsNaN(float64(got[0])) {
			t.Errorf("ElementwiseUnary(%s, NaN) = %v, %v; want NaN", op, got, err)
		}
	}
}

func TestReferenceAffine(t *testing.T) {
	// Acceptance test: TestReferenceAffine
	// Affine computation: Y = MatMul(X, W) + B -> Relu -> Gelu
	g := cudair.NewGraph()
	// X: 2x3, W: 3x2, B: 2x2
	x := g.AddInput("X", []int{2, 3}, cudair.Float32)
	w := g.AddInput("W", []int{3, 2}, cudair.Float32)
	b := g.AddInput("B", []int{2, 2}, cudair.Float32)

	xw := g.AddMatMul(x, w)
	y := g.AddBinary("add", xw, b)
	r := g.AddUnary("relu", y)
	out := g.AddUnary("gelu", r)
	if err := g.SetOutputs(out); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}

	// X = [[1, 2, 3],
	//      [-1, 0, 2]]
	xData := []float32{1, 2, 3, -1, 0, 2}
	// W = [[1, -1],
	//      [0,  2],
	//      [2,  1]]
	wData := []float32{1, -1, 0, 2, 2, 1}
	// XW = [[1*1 + 2*0 + 3*2,  1*(-1) + 2*2 + 3*1],
	//       [-1*1 + 0*0 + 2*2, -1*(-1) + 0*2 + 2*1]]
	//    = [[7, 6],
	//       [3, 3]]
	// B = [[-10, 2],
	//      [-3,  1]]
	bData := []float32{-10, 2, -3, 1}
	// Y = XW + B = [[-3, 8],
	//               [0,  4]]
	// Relu(Y) = [[0, 8],
	//            [0, 4]]
	// Gelu(0) = 0.0
	// Gelu(8) = 0.5 * 8 * (1 + erf(8/sqrt(2))) = 4 * 2 = 8.0
	// Gelu(4) = 0.5 * 4 * (1 + erf(4/sqrt(2))) = 2 * 2 = 4.0

	inputs := map[string][]float32{
		"X": xData,
		"W": wData,
		"B": bData,
	}

	results, err := cudaref.Evaluate(g, inputs)
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}

	res := results[out.Name]
	if len(res) != 4 {
		t.Fatalf("expected 4 outputs, got %d", len(res))
	}

	expected := []float32{0.0, 8.0, 0.0, 4.0}
	for i := range expected {
		diff := math.Abs(float64(res[i] - expected[i]))
		if diff > 1e-3 {
			t.Errorf("at index %d: want ~%f, got %f", i, expected[i], res[i])
		}
	}
}

func TestOnlineAttentionMutantControls(t *testing.T) {
	// Mutant controls for online attention:
	// 1. Mutant inverse rescaling
	// 2. Mutant omitted normalizer
	seqLen := 4
	headDim := 2
	tileSize := 2
	scores := []float32{1.0, 5.0, 2.0, 8.0}
	values := []float32{
		1.0, 0.0,
		0.0, 1.0,
		1.0, 1.0,
		2.0, 3.0,
	}
	mask := []bool{false, false, false, false}

	// Reference calculation
	golden, err := cudaref.OnlineAttentionRow(scores, values, mask, seqLen, headDim, tileSize)
	if err != nil {
		t.Fatalf("OnlineAttentionRow failed: %v", err)
	}

	// Check that single-tile computation matches full sequence (mathematical invariance)
	singleTile, err := cudaref.OnlineAttentionRow(scores, values, mask, seqLen, headDim, seqLen)
	if err != nil {
		t.Fatalf("singleTile OnlineAttentionRow failed: %v", err)
	}
	for i := range golden {
		if math.Abs(float64(golden[i]-singleTile[i])) > 1e-5 {
			t.Errorf("tiled attention != single-tile attention at %d: %f vs %f", i, golden[i], singleTile[i])
		}
	}

	// Mutant 1: Inverse rescaling
	mutantRescale, err := cudaref.OnlineAttentionRowMutantRescale(scores, values, mask, seqLen, headDim, tileSize)
	if err != nil {
		t.Fatalf("OnlineAttentionRowMutantRescale failed: %v", err)
	}
	rescaleMismatch := false
	for i := range golden {
		if math.Abs(float64(golden[i]-mutantRescale[i])) > 1e-3 {
			rescaleMismatch = true
			break
		}
	}
	if !rescaleMismatch {
		t.Fatal("mutant control failed: mutant inverse rescaling should diverge from golden reference")
	}

	// Mutant 2: Omitted normalizer
	mutantNoNorm, err := cudaref.OnlineAttentionRowMutantNoNormalizer(scores, values, mask, seqLen, headDim, tileSize)
	if err != nil {
		t.Fatalf("OnlineAttentionRowMutantNoNormalizer failed: %v", err)
	}
	normMismatch := false
	for i := range golden {
		if math.Abs(float64(golden[i]-mutantNoNorm[i])) > 1e-3 {
			normMismatch = true
			break
		}
	}
	if !normMismatch {
		t.Fatal("mutant control failed: mutant omitted normalizer should diverge from golden reference")
	}
}
