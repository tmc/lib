package hopper

import (
	"fmt"
	"strings"
)

// ExpertAlign sorts a step's (token, slot) routes by local expert, the
// layout [TileGEMM] reads with Grouped and Gather. Route i = t·K + k goes
// to expert ids[i]; it is local when e = (ids[i] - local_start) mod Experts
// is below Local (moe8_decode_layer/reference.py:466). The kernel writes
//
//	offsets [Local+1]  expert e's routes are sorted rows offsets[e] .. offsets[e+1]
//	rows    [T·K]      sorted row j reads token rows[j] (first offsets[Local] used)
//	slots   [T·K]      route i's sorted row, or -1 when its expert is not local
//
// Within an expert, routes keep their order (stable), so the result does
// not depend on scheduling. One block counts with warp ballots.
type ExpertAlign struct {
	Name    string
	K       int // routes per token
	Experts int // global expert count
	Local   int // local experts, at most 32
	Warps   int // per block; 0 means 32
}

func (a ExpertAlign) norm() (ExpertAlign, error) {
	if !cIdent(a.Name) {
		return a, fmt.Errorf("expert align: bad name %q", a.Name)
	}
	if a.Warps == 0 {
		a.Warps = 32
	}
	if a.K < 1 || a.Local < 1 || a.Local > 32 || a.Experts < a.Local || a.Warps < 1 || a.Warps > 32 {
		return a, fmt.Errorf("expert align %s: unsupported K %d, %d of %d experts, %d warps", a.Name, a.K, a.Local, a.Experts, a.Warps)
	}
	return a, nil
}

func (a ExpertAlign) params() []string {
	return []string{"int tokens", "const int32_t* __restrict__ ids", "const int32_t* __restrict__ local_start",
		"int32_t* __restrict__ offsets", "int32_t* __restrict__ rows", "int32_t* __restrict__ slots"}
}

// CUDA returns the kernel source (without [Prelude]), launched as one
// block of 32·Warps threads:
//
//	__global__ void Name(int tokens, ids, local_start, offsets, rows, slots)
func (a ExpertAlign) CUDA() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: sort top-%d routes by local expert (%d of %d).\n", a.Name, a.K, a.Local, a.Experts)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n", 32*a.Warps, a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int K = %d, E = %d, LOCAL = %d, WARPS = %d, THREADS = 32 * WARPS;\n", a.K, a.Experts, a.Local, a.Warps)
	b.WriteString(`    __shared__ int wc[WARPS][32];
    __shared__ int total[32], start[33];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, routes = tokens * K, ls = local_start[0];
    if (tid < 32) total[tid] = 0;
    auto expert = [&](int i) {
        const int e = ((ids[i] - ls) % E + E) % E;
        return e < LOCAL ? e : -1;
    };
    // Pass 1: each route's rank within its expert, in route order.
    for (int base = 0; base < routes; base += THREADS) {
        const int i = base + tid, e = i < routes ? expert(i) : -1;
        unsigned mine = 0;
        int count = 0;
#pragma unroll 4
        for (int x = 0; x < LOCAL; ++x) {
            const unsigned m = __ballot_sync(0xffffffffu, e == x);
            if (lane == x) count = __popc(m);
            if (e == x) mine = m;
        }
        wc[warp][lane] = count;
        __syncthreads();
        if (tid < LOCAL) {
            int run = total[tid];
            for (int w = 0; w < WARPS; ++w) { const int c = wc[w][tid]; wc[w][tid] = run; run += c; }
            total[tid] = run;
        }
        __syncthreads();
        if (i < routes) slots[i] = e >= 0 ? wc[warp][e] + __popc(mine & ((1u << lane) - 1u)) : -1;
        __syncthreads();
    }
    if (tid == 0) {
        start[0] = 0;
        for (int e = 0; e < LOCAL; ++e) start[e + 1] = start[e] + total[e];
    }
    __syncthreads();
    if (tid <= LOCAL) offsets[tid] = start[tid];
    // Pass 2: absolute rows; each thread revisits the routes it ranked.
    for (int i = tid; i < routes; i += THREADS) {
        const int e = expert(i);
        if (e < 0) continue;
        const int j = start[e] + slots[i];
        slots[i] = j;
        rows[j] = i / K;
    }
}
`)
	return b.String(), nil
}

// Launcher returns void Name_launch(<params...>, cudaStream_t stream).
func (a ExpertAlign) Launcher() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	return launcher(a.Name, a.params(), func(b *strings.Builder) {
		fmt.Fprintf(b, "    const dim3 grid(tokens > 0 ? 1 : 0), block(%d);\n", 32*a.Warps)
	}), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (a ExpertAlign) Source() (string, error) {
	src, err := a.CUDA()
	if err != nil {
		return "", err
	}
	l, err := a.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// Reference returns offsets, rows (offsets[Local] entries) and slots for
// ids [tokens·K].
func (a ExpertAlign) Reference(ids []int32, localStart int32) (offsets, rows, slots []int32, err error) {
	a, err = a.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	if len(ids)%a.K != 0 {
		return nil, nil, nil, fmt.Errorf("expert align %s reference: %d ids for top-%d", a.Name, len(ids), a.K)
	}
	byExpert := make([][]int, a.Local)
	for i, id := range ids {
		e := ((int(id-localStart))%a.Experts + a.Experts) % a.Experts
		if e < a.Local {
			byExpert[e] = append(byExpert[e], i)
		}
	}
	slots = make([]int32, len(ids))
	for i := range slots {
		slots[i] = -1
	}
	offsets = []int32{0}
	for _, routes := range byExpert {
		for _, i := range routes {
			slots[i] = int32(len(rows))
			rows = append(rows, int32(i/a.K))
		}
		offsets = append(offsets, int32(len(rows)))
	}
	return offsets, rows, slots, nil
}
