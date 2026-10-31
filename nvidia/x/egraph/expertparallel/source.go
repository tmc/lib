package expertparallel

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// A region is one named buffer in the workspace.
type region struct {
	name  string
	ctype string
	off   int // bytes from the start of its area
}

// workspace lays out the global area and one area per rank, every buffer
// 256-byte aligned.
type workspace struct {
	global, rank []region
	globalBytes  int
	rankBytes    int
}

func (w *workspace) add(rank bool, name, ctype string, bytes int) {
	area, n := &w.global, &w.globalBytes
	if rank {
		area, n = &w.rank, &w.rankBytes
	}
	*area = append(*area, region{name, ctype, *n})
	*n += (bytes + 255) / 256 * 256
}

func (p *Plan) workspace() *workspace {
	R, S, K, C, H := p.Ranks, p.TokensPerRank(), p.TopK, p.Capacity(), p.Hidden
	w := new(workspace)
	w.add(false, "peers", p.Name+"_peer", R*5*8)
	w.add(false, "local_start", "int32_t", R*4)
	w.add(false, "shared_offsets", "int32_t", 2*4)
	w.add(true, "ids", "int32_t", S*K*4)
	w.add(true, "wts", "float", S*K*4)
	w.add(true, "send_off", "int32_t", (R+1)*4)
	w.add(true, "send_pos", "int32_t", S*K*4)
	w.add(true, "send_route", "int32_t", S*K*4)
	w.add(true, "recv_x", "uint16_t", C*H*2)
	w.add(true, "recv_ids", "int32_t", C*4)
	w.add(true, "recv_w", "float", C*4)
	w.add(true, "y", "float", C*H*4)
	for _, t := range p.Expert.Temps {
		w.add(true, t.Name, ctype(t.DType), size(t.Shape)*t.DType.Bits()/8)
	}
	if p.Shared != nil {
		for _, t := range p.Shared.Temps {
			w.add(true, t.Name, ctype(t.DType), size(t.Shape)*t.DType.Bits()/8)
		}
		w.add(true, "shared_out", "float", S*H*4)
	}
	return w
}

// WorkspaceBytes returns the device workspace [Plan.Source]'s run function
// needs: a global area plus one area per rank.
func (p *Plan) WorkspaceBytes() int {
	w := p.workspace()
	return w.globalBytes + p.Ranks*w.rankBytes
}

func ctype(d hopper.DType) string {
	switch d {
	case hopper.F32:
		return "float"
	case hopper.BF16, hopper.F16:
		return "uint16_t"
	case hopper.I32:
		return "int32_t"
	}
	return "uint8_t"
}

func size(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// Source returns one translation unit with every kernel of the plan and a
// host function
//
//	void Name_run(<Config.Inputs...>, uint16_t* out, void* workspace)
//
// that runs all Ranks ranks on the current device and writes out [Tokens,
// Hidden] BF16. workspace is [Plan.WorkspaceBytes] of device memory. The
// function synchronizes the device before returning. Under EG_HOST_EMU
// (cudahostemu) streams and events are synchronous and the ranks run in
// issue order.
func (p *Plan) Source() (string, error) {
	var b strings.Builder
	b.WriteString(hopper.Prelude)
	route, err := p.Route.CUDA()
	if err != nil {
		return "", err
	}
	routeL, err := p.Route.Launcher()
	if err != nil {
		return "", err
	}
	b.WriteString("\n" + route + "\n" + routeL)
	b.WriteString(p.kernels())
	expert, err := p.Expert.Source()
	if err != nil {
		return "", err
	}
	b.WriteString("\n" + expert)
	if p.Shared != nil {
		shared, err := p.Shared.Source()
		if err != nil {
			return "", err
		}
		b.WriteString("\n" + shared)
	}
	p.host(&b)
	return b.String(), nil
}

// kernels returns the peer table type and the layout, dispatch and
// combine kernels with their launchers.
func (p *Plan) kernels() string {
	n := p.Name
	R, S, K, L, E, C, H := p.Ranks, p.TokensPerRank(), p.TopK, p.LocalExperts(), p.Experts, p.Capacity(), p.Hidden
	r := strings.NewReplacer(
		"NAME", n,
		"$R", fmt.Sprint(R), "$S", fmt.Sprint(S), "$K", fmt.Sprint(K), "$L", fmt.Sprint(L),
		"$E", fmt.Sprint(E), "$C", fmt.Sprint(C), "$H", fmt.Sprint(H),
	)
	return r.Replace(`
// NAME_peer is one rank's buffers as every rank sees them: the peer table
// NVLink symmetric memory provides across GPUs.
struct NAME_peer {
    const int32_t* send_off;   // [R+1] route offsets by destination rank
    const int32_t* send_route; // [S*K] route index at each send position
    const int32_t* ids;        // [S*K] routed global expert ids
    const uint16_t* x;         // [S, H] the rank's token rows
    const float* y;            // [C, H] expert-stage output rows
};

// NAME_layout orders one rank's routes by destination rank, stably:
// send_off [R+1], send_pos [S*K] (route -> send position) and send_route
// [S*K] (send position -> route). One block.
__global__ void __launch_bounds__(256) NAME_layout(
    const int32_t* __restrict__ ids, int32_t* __restrict__ send_off,
    int32_t* __restrict__ send_pos, int32_t* __restrict__ send_route) {
    constexpr int R = $R, N = $S * $K, L = $L;
    __shared__ int start[R + 1];
    const int tid = threadIdx.x;
    for (int d = tid; d < R; d += blockDim.x) {
        int c = 0;
        for (int i = 0; i < N; ++i) c += ids[i] / L == d;
        start[d] = c;
    }
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int d = 0; d < R; ++d) { const int c = start[d]; start[d] = run; run += c; }
        start[R] = run;
    }
    __syncthreads();
    for (int d = tid; d <= R; d += blockDim.x) send_off[d] = start[d];
    for (int d = tid; d < R; d += blockDim.x) {
        int j = start[d];
        for (int i = 0; i < N; ++i) {
            if (ids[i] / L != d) continue;
            send_pos[i] = j;
            send_route[j++] = i;
        }
    }
}

inline void NAME_layout_launch(const int32_t* ids, int32_t* send_off, int32_t* send_pos, int32_t* send_route, cudaStream_t stream) {
    EG_LAUNCH(NAME_layout, dim3(1), dim3(256), 0, stream, ids, send_off, send_pos, send_route);
}

// NAME_dispatch is rank d's all-to-all receive: block s pulls source rank
// s's routes to d into rows base(s) .. base(s)+n of recv_x and recv_ids,
// base(s) the routes of ranks before s. Unused rows get an expert id no
// rank-d expert has; every recv_w is 1.
__global__ void __launch_bounds__(256) NAME_dispatch(
    const NAME_peer* __restrict__ peers, int d,
    uint16_t* __restrict__ recv_x, int32_t* __restrict__ recv_ids, float* __restrict__ recv_w) {
    constexpr int R = $R, K = $K, L = $L, E = $E, C = $C, H = $H;
    const int s = blockIdx.x, tid = threadIdx.x;
    int base = 0, total = 0;
    for (int q = 0; q < R; ++q) {
        const int c = peers[q].send_off[d + 1] - peers[q].send_off[d];
        if (q < s) base += c;
        total += c;
    }
    const NAME_peer src = peers[s];
    const int lo = src.send_off[d], n = src.send_off[d + 1] - lo;
    for (int j = 0; j < n; ++j) {
        const int i = src.send_route[lo + j];
        const eg_u32x4* from = reinterpret_cast<const eg_u32x4*>(src.x + (size_t)(i / K) * H);
        eg_u32x4* to = reinterpret_cast<eg_u32x4*>(recv_x + (size_t)(base + j) * H);
        for (int c = tid; c < H / 8; c += blockDim.x) to[c] = from[c];
        if (tid == 0) recv_ids[base + j] = src.ids[i];
    }
    if (s == 0) {
        for (int j = total + tid; j < C; j += blockDim.x) recv_ids[j] = (d * L + L) % E;
        for (int j = tid; j < C; j += blockDim.x) recv_w[j] = 1.0f;
    }
}

inline void NAME_dispatch_launch(const NAME_peer* peers, int d, uint16_t* recv_x, int32_t* recv_ids, float* recv_w, cudaStream_t stream) {
    EG_LAUNCH(NAME_dispatch, dim3($R), dim3(256), 0, stream, peers, d, recv_x, recv_ids, recv_w);
}

// NAME_combine is rank r's all-to-all return and weighted sum: block t
// pulls token t's K expert rows from the ranks that own the experts and
// accumulates them in increasing expert id with fmaf, as ExpertSum does,
// then adds the shared expert row (when shared is not null) and rounds to
// BF16.
__global__ void __launch_bounds__(256) NAME_combine(
    const NAME_peer* __restrict__ peers, int r,
    const int32_t* __restrict__ ids, const float* __restrict__ wts, const int32_t* __restrict__ send_pos,
    const float* __restrict__ shared, uint16_t* __restrict__ out) {
    constexpr int K = $K, L = $L, H = $H;
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ const float* row[K];
    __shared__ int key[K];
    __shared__ float wt[K];
    if (tid < K) {
        const int i = t * K + tid, e = ids[i], d = e / L;
        int base = 0;
        for (int q = 0; q < r; ++q) base += peers[q].send_off[d + 1] - peers[q].send_off[d];
        row[tid] = peers[d].y + (size_t)(base + send_pos[i] - peers[r].send_off[d]) * H;
        key[tid] = e;
        wt[tid] = wts[i];
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < K; ++i) {
            const float* p = row[i];
            const int e = key[i];
            const float w = wt[i];
            int j = i;
            for (; j > 0 && key[j - 1] > e; --j) { row[j] = row[j - 1]; key[j] = key[j - 1]; wt[j] = wt[j - 1]; }
            row[j] = p; key[j] = e; wt[j] = w;
        }
    }
    __syncthreads();
    for (int c = tid; c < H; c += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc = fmaf(row[k][c], wt[k], acc);
        if (shared) acc += shared[(size_t)t * H + c];
        out[(size_t)t * H + c] = eg_f32_to_bf16(acc);
    }
}

inline void NAME_combine_launch(const NAME_peer* peers, int r, const int32_t* ids, const float* wts, const int32_t* send_pos, const float* shared, uint16_t* out, cudaStream_t stream) {
    EG_LAUNCH(NAME_combine, dim3($S), dim3(256), 0, stream, peers, r, ids, wts, send_pos, shared, out);
}
`)
}

// host writes the event shims and Name_run.
func (p *Plan) host(b *strings.Builder) {
	n := p.Name
	R, S, E, H, L, I := p.Ranks, p.TokensPerRank(), p.Experts, p.Hidden, p.LocalExperts(), p.Inter
	w := p.workspace()
	fmt.Fprintf(b, `
#if defined(EG_HOST_EMU)
typedef int %[1]s_event;
inline void %[1]s_event_create(%[1]s_event*) {}
inline void %[1]s_event_destroy(%[1]s_event) {}
inline void %[1]s_event_record(%[1]s_event, cudaStream_t) {}
inline void %[1]s_stream_wait(cudaStream_t, %[1]s_event) {}
#else
typedef cudaEvent_t %[1]s_event;
inline void %[1]s_event_create(%[1]s_event* e) { cudaEventCreateWithFlags(e, cudaEventDisableTiming); }
inline void %[1]s_event_destroy(%[1]s_event e) { cudaEventDestroy(e); }
inline void %[1]s_event_record(%[1]s_event e, cudaStream_t s) { cudaEventRecord(e, s); }
inline void %[1]s_stream_wait(cudaStream_t s, %[1]s_event e) { cudaStreamWaitEvent(s, e, 0); }
#endif

inline size_t %[1]s_workspace_bytes() { return %[2]d; }
`, n, w.globalBytes+R*w.rankBytes)

	var params []string
	for _, t := range p.Inputs() {
		params = append(params, "const "+ctype(t.DType)+"* "+t.Name)
	}
	params = append(params, "uint16_t* out", "void* workspace")
	fmt.Fprintf(b, "\ninline void %s_run(%s) {\n", n, strings.Join(params, ", "))
	fmt.Fprintf(b, "    constexpr int R = %d;\n", R)
	b.WriteString("    char* ws = static_cast<char*>(workspace);\n")
	for _, g := range w.global {
		fmt.Fprintf(b, "    %s* %s = reinterpret_cast<%s*>(ws + %d);\n", g.ctype, g.name, g.ctype, g.off)
	}
	b.WriteString("    struct rank_bufs {\n")
	for _, g := range w.rank {
		fmt.Fprintf(b, "        %s* %s;\n", g.ctype, g.name)
	}
	b.WriteString("    } rk[R];\n")
	fmt.Fprintf(b, "    %s_peer host_peers[R];\n", n)
	fmt.Fprintf(b, "    int32_t host_ls[R], host_so[2] = {0, %d};\n", S)
	b.WriteString("    for (int r = 0; r < R; ++r) {\n")
	fmt.Fprintf(b, "        char* base = ws + %d + (size_t)r * %d;\n", w.globalBytes, w.rankBytes)
	for _, g := range w.rank {
		fmt.Fprintf(b, "        rk[r].%s = reinterpret_cast<%s*>(base + %d);\n", g.name, g.ctype, g.off)
	}
	fmt.Fprintf(b, "        host_peers[r] = {rk[r].send_off, rk[r].send_route, rk[r].ids, x + (size_t)r * %d, rk[r].y};\n", S*H)
	fmt.Fprintf(b, "        host_ls[r] = r * %d;\n", L)
	b.WriteString("    }\n")
	b.WriteString("    cudaMemcpy(peers, host_peers, sizeof host_peers, cudaMemcpyHostToDevice);\n")
	b.WriteString("    cudaMemcpy(local_start, host_ls, sizeof host_ls, cudaMemcpyHostToDevice);\n")
	b.WriteString("    cudaMemcpy(shared_offsets, host_so, sizeof host_so, cudaMemcpyHostToDevice);\n")
	fmt.Fprintf(b, "    cudaStream_t main_s[R], side_s[R];\n    %s_event routed[R], expert[R], shared_done[R];\n", n)
	fmt.Fprintf(b, `    for (int r = 0; r < R; ++r) {
        cudaStreamCreate(&main_s[r]);
        cudaStreamCreate(&side_s[r]);
        %[1]s_event_create(&routed[r]);
        %[1]s_event_create(&expert[r]);
        %[1]s_event_create(&shared_done[r]);
    }
`, n)

	// Phase 1: route and layout on main; shared expert on side.
	b.WriteString("    for (int r = 0; r < R; ++r) {\n")
	bias := ""
	if p.Route.Bias != 0 {
		bias = "bias, "
	}
	fmt.Fprintf(b, "        %s_launch(%d, logits + (size_t)r * %d, %srk[r].ids, rk[r].wts, main_s[r]);\n", p.Route.Name, S, S*E, bias)
	fmt.Fprintf(b, "        %s_layout_launch(rk[r].ids, rk[r].send_off, rk[r].send_pos, rk[r].send_route, main_s[r]);\n", n)
	fmt.Fprintf(b, "        %s_event_record(routed[r], main_s[r]);\n", n)
	if p.Shared != nil {
		args := p.planArgs(p.Shared, map[string]string{
			"x":              fmt.Sprintf("x + (size_t)r * %d", S*H),
			"shared_offsets": "shared_offsets",
		}, "rk[r].shared_out")
		fmt.Fprintf(b, "        %s_run(%s, side_s[r]);\n", p.Shared.Name, args)
		fmt.Fprintf(b, "        %s_event_record(shared_done[r], side_s[r]);\n", n)
	}
	b.WriteString("    }\n")

	// Phase 2: dispatch and the expert stage, after every rank's layout.
	b.WriteString("    for (int d = 0; d < R; ++d) {\n")
	fmt.Fprintf(b, "        for (int q = 0; q < R; ++q) %s_stream_wait(main_s[d], routed[q]);\n", n)
	fmt.Fprintf(b, "        %s_dispatch_launch(peers, d, rk[d].recv_x, rk[d].recv_ids, rk[d].recv_w, main_s[d]);\n", n)
	args := p.planArgs(p.Expert, map[string]string{
		"recv_x":      "rk[d].recv_x",
		"recv_ids":    "rk[d].recv_ids",
		"recv_w":      "rk[d].recv_w",
		"local_start": "local_start + d",
		"w13_q":       fmt.Sprintf("w13_q + (size_t)d * %d", L*2*I*H/2),
		"w13_sf":      fmt.Sprintf("w13_sf + (size_t)d * %d", L*2*I*H/16),
		"w13_gs":      fmt.Sprintf("w13_gs + (size_t)d * %d", L),
		"w2_q":        fmt.Sprintf("w2_q + (size_t)d * %d", L*H*I/2),
		"w2_sf":       fmt.Sprintf("w2_sf + (size_t)d * %d", L*H*I/16),
		"w2_gs":       fmt.Sprintf("w2_gs + (size_t)d * %d", L),
		"post_w":      "post_w",
	}, "rk[d].y")
	args = strings.ReplaceAll(args, "rk[r].", "rk[d].")
	fmt.Fprintf(b, "        %s_run(%s, main_s[d]);\n", p.Expert.Name, args)
	fmt.Fprintf(b, "        %s_event_record(expert[d], main_s[d]);\n", n)
	b.WriteString("    }\n")

	// Phase 3: combine, after every rank's expert stage and this rank's
	// shared expert.
	b.WriteString("    for (int r = 0; r < R; ++r) {\n")
	fmt.Fprintf(b, "        for (int q = 0; q < R; ++q) %s_stream_wait(main_s[r], expert[q]);\n", n)
	shared := "nullptr"
	if p.Shared != nil {
		fmt.Fprintf(b, "        %s_stream_wait(main_s[r], shared_done[r]);\n", n)
		shared = "rk[r].shared_out"
	}
	fmt.Fprintf(b, "        %s_combine_launch(peers, r, rk[r].ids, rk[r].wts, rk[r].send_pos, %s, out + (size_t)r * %d, main_s[r]);\n", n, shared, S*H)
	b.WriteString("    }\n")
	fmt.Fprintf(b, `    cudaDeviceSynchronize();
    for (int r = 0; r < R; ++r) {
        cudaStreamDestroy(main_s[r]);
        cudaStreamDestroy(side_s[r]);
        %[1]s_event_destroy(routed[r]);
        %[1]s_event_destroy(expert[r]);
        %[1]s_event_destroy(shared_done[r]);
    }
}
`, n)
}

// planArgs returns the argument list of plan's run function: inputs from
// bind (by name, else the parameter of the same name), temps from the rank
// workspace rk[r], and out.
func (p *Plan) planArgs(plan *hopper.Plan, bind map[string]string, out string) string {
	var args []string
	for _, t := range plan.Inputs {
		if a, ok := bind[t.Name]; ok {
			args = append(args, a)
		} else {
			args = append(args, t.Name)
		}
	}
	for _, t := range plan.Temps {
		args = append(args, "rk[r]."+t.Name)
	}
	return strings.Join(append(args, out), ", ")
}
