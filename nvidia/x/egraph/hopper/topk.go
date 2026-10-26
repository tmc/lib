package hopper

import (
	"fmt"
	"sort"
	"strings"
)

// RowTopK selects, for each row of D values, the K columns with the largest
// keys and writes their indices (I32 [rows, K], best first, ties to the
// lower column) and weights (F32 [rows, K]). One warp owns a row.
//
// With x the row and b the optional per-column bias:
//
//	score = Score(x)            e.g. (sigmoid (mul $0 0.1767766952966369))
//	key   = Key(score, b)       e.g. (add $0 $1)
//	weight = score at the chosen columns, then, with Normalize,
//	         weight / (Σ weights + Eps) · Scale
//
// which is Moe8's biased sigmoid top-8 routing
// (moe8_decode_layer/reference.py:418-424, 462-463): the bias picks experts
// but the weights come from the unbiased scores.
type RowTopK struct {
	Name      string
	D, K      int
	X         DType  // F32 or BF16 rows of D values, contiguous
	Bias      DType  // 0 for none, else F32 or BF16 [D]
	Score     Lambda // $0 = x; zero value means $0
	Key       Lambda // $0 = score, $1 = bias; zero value means $0
	Normalize bool
	Eps       float32
	Scale     float32 // 0 means 1
}

func (k RowTopK) norm() (RowTopK, error) {
	if !cIdent(k.Name) {
		return k, fmt.Errorf("row topk: bad name %q", k.Name)
	}
	if k.D < 32 || k.D > 1024 || k.D%32 != 0 || k.K < 1 || k.K > 32 || k.K > k.D {
		return k, fmt.Errorf("row topk %s: unsupported D %d, K %d", k.Name, k.D, k.K)
	}
	if k.X != F32 && k.X != BF16 || k.Bias != 0 && k.Bias != F32 && k.Bias != BF16 {
		return k, fmt.Errorf("row topk %s: bad dtypes", k.Name)
	}
	if k.Score.root == nil {
		k.Score = Identity
	}
	if k.Key.root == nil {
		k.Key = Identity
	}
	if k.Score.Arity() > 1 || k.Key.Arity() > 2 || k.Key.Arity() == 2 && k.Bias == 0 {
		return k, fmt.Errorf("row topk %s: score takes $0, key takes $0 and the bias $1", k.Name)
	}
	if k.Scale == 0 {
		k.Scale = 1
	}
	return k, nil
}

func (k RowTopK) params() []string {
	p := []string{"int rows", "const " + k.X.cType() + "* __restrict__ x"}
	if k.Bias != 0 {
		p = append(p, "const "+k.Bias.cType()+"* __restrict__ bias")
	}
	return append(p, "int32_t* __restrict__ ids", "float* __restrict__ weights")
}

// Grid returns the launch grid and block for rows rows (8 rows per block).
func (k RowTopK) Grid(rows int) (grid, block [3]int) {
	return [3]int{(rows + 7) / 8, 1, 1}, [3]int{256, 1, 1}
}

// CUDA returns the kernel source (without [Prelude]):
// __global__ void Name(int rows, x, [bias,] ids, weights).
func (k RowTopK) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	score, err := k.Score.CUDA("xv[j]")
	if err != nil {
		return "", err
	}
	key, err := k.Key.CUDA("sc[j]", "bv[j]")
	if err != nil {
		return "", err
	}
	per := k.D / 32
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: top-%d of %d per row by key %s of score %s.\n", k.Name, k.K, k.D, k.Key, k.Score)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n", k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int D = %d, K = %d, PER = %d;\n", k.D, k.K, per)
	b.WriteString(`    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    // Lane l holds columns l*PER .. l*PER+PER-1.
    float xv[PER], bv[PER], sc[PER], key[PER];
`)
	if per%2 == 0 {
		b.WriteString("    eg_load<PER>(x + (size_t)r * D + lane * PER, xv);\n")
	} else {
		fmt.Fprintf(&b, "    for (int j = 0; j < PER; ++j) xv[j] = %s;\n", scalarLoad(k.X, "x[(size_t)r * D + lane * PER + j]"))
	}
	if k.Bias != 0 {
		if per%2 == 0 {
			b.WriteString("    eg_load<PER>(bias + lane * PER, bv);\n")
		} else {
			fmt.Fprintf(&b, "    for (int j = 0; j < PER; ++j) bv[j] = %s;\n", scalarLoad(k.Bias, "bias[lane * PER + j]"))
		}
	} else {
		b.WriteString("#pragma unroll\n    for (int j = 0; j < PER; ++j) bv[j] = 0.0f;\n")
	}
	fmt.Fprintf(&b, "#pragma unroll\n    for (int j = 0; j < PER; ++j) {\n        sc[j] = %s;\n        key[j] = %s;\n    }\n", score, key)
	b.WriteString(`    int my_id = 0;
    float my_w = 0.0f, total = 0.0f;
    for (int s = 0; s < K; ++s) {
        // Lane-local best untaken key, ties to the lower column.
        float best = -__uint_as_float(0x7f800000u);
        int col = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (key[j] > best) { best = key[j]; col = lane * PER + j; }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oc = __shfl_xor_sync(0xffffffffu, col, m);
            if (ob > best || (ob == best && oc < col)) { best = ob; col = oc; }
        }
        const int owner = col / PER;
        float w = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (lane == owner && lane * PER + j == col) { w = sc[j]; key[j] = -__uint_as_float(0x7f800000u); }
        w = __shfl_sync(0xffffffffu, w, owner);
        total += w;
        if (lane == s) { my_id = col; my_w = w; }
    }
`)
	if k.Normalize {
		fmt.Fprintf(&b, "    my_w = my_w / (total + %s) * %s;\n", floatLiteral(k.Eps), floatLiteral(k.Scale))
	}
	b.WriteString(`    if (lane < K) {
        ids[(size_t)r * K + lane] = my_id;
        weights[(size_t)r * K + lane] = my_w;
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (k RowTopK) Launcher() (string, error) {
	if _, err := k.norm(); err != nil {
		return "", err
	}
	return launcher(k.Name, k.params(), func(b *strings.Builder) {
		b.WriteString("    const dim3 grid((rows + 7) / 8), block(256);\n")
	}), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (k RowTopK) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// Reference computes ids and weights for rows rows of x (values as FP32).
func (k RowTopK) Reference(rows int, x, bias []float32) (ids []int32, weights []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, err
	}
	if len(x) != rows*k.D || k.Bias != 0 && len(bias) != k.D {
		return nil, nil, fmt.Errorf("row topk %s reference: bad inputs", k.Name)
	}
	for r := 0; r < rows; r++ {
		cols := make([]int, k.D)
		sc := make([]float32, k.D)
		key := make([]float32, k.D)
		for c := range cols {
			cols[c] = c
			sc[c] = k.Score.Eval(x[r*k.D+c])
			b := float32(0)
			if k.Bias != 0 {
				b = bias[c]
			}
			key[c] = k.Key.Eval(sc[c], b)
		}
		sort.SliceStable(cols, func(i, j int) bool { return key[cols[i]] > key[cols[j]] })
		var total float32
		for _, c := range cols[:k.K] {
			total += sc[c]
		}
		for _, c := range cols[:k.K] {
			w := sc[c]
			if k.Normalize {
				w = w / (total + k.Eps) * k.Scale
			}
			ids = append(ids, int32(c))
			weights = append(weights, w)
		}
	}
	return ids, weights, nil
}
