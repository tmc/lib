package hopper

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// ExpertSum is the MoE post-expert norm and top-K weighted sum
// (moe8_moe_w4a16/reference.py:88-90, 106):
//
//	out[t] = Σ over routes k of token t with r = slots[t·K + k] ≥ 0 of
//	         (y[r] · rsqrt(Σ_j ss[r][j] / Width + Eps)) · weights[t·K + k]
//
// where y [rows, Width] is the down projection with the (1 + post_w) gain
// applied and ss [rows, Width/32] its row sums of squares, as [TileGEMM]
// writes them with SumSq and the epilogue (mul $0 (add 1 $1)); slots is
// [ExpertAlign]'s route-to-row map and weights [RowTopK]'s routing weights.
// The routes of a token are summed in increasing row order, which is
// increasing local expert, the order of reference.py's loop over experts.
// A token with no local route gets zeros.
//
// One block of 256 threads owns a token.
type ExpertSum struct {
	Name  string
	Width int     // columns, a multiple of 32
	K     int     // routes per token, 1 to 32
	Y     DType   // BF16 or F32
	Out   DType   // BF16 or F32
	Eps   float32 // 0 means 1e-8
}

func (s ExpertSum) norm() (ExpertSum, error) {
	if !cIdent(s.Name) {
		return s, fmt.Errorf("expert sum: bad name %q", s.Name)
	}
	if s.Width < 32 || s.Width%32 != 0 || s.K < 1 || s.K > 32 {
		return s, fmt.Errorf("expert sum %s: width %d must be a multiple of 32 and K %d in 1..32", s.Name, s.Width, s.K)
	}
	if s.Y != BF16 && s.Y != F32 || s.Out != BF16 && s.Out != F32 {
		return s, fmt.Errorf("expert sum %s: y and out must be BF16 or F32", s.Name)
	}
	if s.Eps == 0 {
		s.Eps = 1e-8
	}
	return s, nil
}

func (s ExpertSum) params() []string {
	return []string{"int tokens", "const " + s.Y.cType() + "* __restrict__ y", "const float* __restrict__ ss",
		"const int32_t* __restrict__ slots", "const float* __restrict__ weights", s.Out.cType() + "* __restrict__ out"}
}

// Grid returns the grid and block for tokens tokens.
func (s ExpertSum) Grid(tokens int) (grid, block [3]int) {
	return [3]int{tokens, 1, 1}, [3]int{256, 1, 1}
}

// CUDA returns the kernel source (without [Prelude]):
//
//	__global__ void Name(int tokens, y, ss, slots, weights, out)
func (s ExpertSum) CUDA() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: post-expert norm and top-%d weighted sum over %d columns.\n", s.Name, s.K, s.Width)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n", s.Name, strings.Join(s.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int W = %d, K = %d, G = W / 32;\n    constexpr float EPS = %s;\n", s.Width, s.K, floatLiteral(s.Eps))
	b.WriteString(`    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ int row[K];
    __shared__ float inv[K], wt[K];
    // A warp per route: the row's rsqrt of mean square, from its 32-column sums.
    for (int k = warp; k < K; k += 8) {
        const int r = slots[(size_t)t * K + k];
        float sum = 0.0f;
        for (int j = lane; j < G && r >= 0; j += 32) sum += ss[(size_t)r * G + j];
        sum = eg_warp_sum(sum);
        if (lane == 0) {
            row[k] = r;
            inv[k] = rsqrtf(sum / W + EPS);
            wt[k] = weights[(size_t)t * K + k];
        }
    }
    __syncthreads();
    // Routes in increasing row order (local expert order), absent ones last.
    if (tid == 0) {
        for (int i = 1; i < K; ++i) {
            const int r = row[i];
            const float v = inv[i], w = wt[i];
            int j = i;
            for (; j > 0 && (unsigned)row[j - 1] > (unsigned)r; --j) {
                row[j] = row[j - 1]; inv[j] = inv[j - 1]; wt[j] = wt[j - 1];
            }
            row[j] = r; inv[j] = v; wt[j] = w;
        }
    }
    __syncthreads();
    for (int c = tid * 8; c < W; c += 256 * 8) {
        float acc[8] = {};
        for (int k = 0; k < K && row[k] >= 0; ++k) {
            float v[8];
            eg_load<8>(y + (size_t)row[k] * W + c, v);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += v[i] * inv[k] * wt[k];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
`)
	if s.Out == BF16 {
		b.WriteString("            out[(size_t)t * W + c + i] = eg_f32_to_bf16(acc[i]);\n")
	} else {
		b.WriteString("            out[(size_t)t * W + c + i] = acc[i];\n")
	}
	b.WriteString("    }\n}\n")
	return b.String(), nil
}

// Launcher returns
//
//	void Name_launch(int tokens, y, ss, slots, weights, out, cudaStream_t stream)
func (s ExpertSum) Launcher() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	params := s.params()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", s.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    if (tokens > 0) EG_LAUNCH(%s, dim3(tokens), dim3(256), 0, stream, %s);\n}\n",
		s.Name, strings.Join(paramNames(params), ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (s ExpertSum) Source() (string, error) {
	src, err := s.CUDA()
	if err != nil {
		return "", err
	}
	l, err := s.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// Reference computes out [tokens, Width] in FP32 in the kernel's order from
// y values (as stored), ss, slots and weights, rounded to Out.
func (s ExpertSum) Reference(tokens int, y, ss []float32, slots []int32, weights []float32) ([]float32, error) {
	s, err := s.norm()
	if err != nil {
		return nil, err
	}
	g := s.Width / 32
	if len(slots) != tokens*s.K || len(weights) != tokens*s.K || len(y)%s.Width != 0 || len(ss) != len(y)/32 {
		return nil, fmt.Errorf("expert sum %s reference: bad input sizes", s.Name)
	}
	out := make([]float32, tokens*s.Width)
	for t := range tokens {
		type route struct {
			r       int
			inv, wt float32
		}
		var rs []route
		for k := range s.K {
			r := int(slots[t*s.K+k])
			if r < 0 {
				continue
			}
			if (r+1)*s.Width > len(y) {
				return nil, fmt.Errorf("expert sum %s reference: row %d out of range", s.Name, r)
			}
			var sum float32
			for _, v := range ss[r*g : (r+1)*g] {
				sum += v
			}
			inv := float32(1 / math.Sqrt(float64(sum/float32(s.Width)+s.Eps)))
			rs = append(rs, route{r, inv, weights[t*s.K+k]})
		}
		sort.SliceStable(rs, func(i, j int) bool { return rs[i].r < rs[j].r })
		o := out[t*s.Width : (t+1)*s.Width]
		for _, rt := range rs {
			for c, v := range y[rt.r*s.Width : (rt.r+1)*s.Width] {
				o[c] += v * rt.inv * rt.wt
			}
		}
		for c := range o {
			o[c] = s.Out.Round(o[c])
		}
	}
	return out, nil
}
