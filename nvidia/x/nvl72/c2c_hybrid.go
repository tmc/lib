package nvl72

import (
	"encoding/binary"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// C2CCoExecTaskKind classifies work a GPU kernel hands to Grace or Vera CPU
// threads over coherent NVLink-C2C memory.
type C2CCoExecTaskKind uint32

const (
	TaskColdMoESingleToken   C2CCoExecTaskKind = iota // evaluate a cold, LPDDR5X-resident expert in place on the CPU
	TaskGrammarBitmaskDFA                             // advance grammar DFAs and write next-token bitmasks
	TaskRadixPrefixLookup                             // radix-tree prefix lookup and KV page staging
	TaskApproximateANNSearch                          // vector search over an LPDDR5X-resident index
)

func (k C2CCoExecTaskKind) String() string {
	switch k {
	case TaskColdMoESingleToken:
		return "cold_moe"
	case TaskGrammarBitmaskDFA:
		return "grammar_bitmask"
	case TaskRadixPrefixLookup:
		return "radix_lookup"
	case TaskApproximateANNSearch:
		return "ann_search"
	}
	return fmt.Sprintf("C2CCoExecTaskKind(%d)", uint32(k))
}

// C2CHybridConfig describes one CPU-GPU superchip for the co-execution
// model. Bandwidths are GB/s, compute is dense TFLOPS.
type C2CHybridConfig struct {
	Arch                string // "gb200", "gb300" or "vr200"
	ArmCores            int
	LPDDR5XBandwidthGB  float64 // CPU memory bandwidth
	C2CBandwidthGB      float64 // unidirectional CPU-to-GPU NVLink-C2C bandwidth
	DoorbellLatencyNs   float64 // one coherent cache-line handoff
	ExpertWeightMB      float64 // one expert's weights (1 MB = 1e6 bytes)
	WeightBytesPerParam float64 // 1 for FP8, 0.5 for NVFP4 (default 1)
	ArmTFLOPS           float64 // CPU vector peak at the weight precision
	GPUTFLOPS           float64 // GPU tensor-core peak at the weight precision
	HiddenDim           int     // expert output width per token
	OutputBytesPerElem  int     // default 2 (BF16)
}

// C2CPreset returns the model inputs for arch. The bandwidths are the
// published per-superchip figures; ArmTFLOPS (cores x clock x SIMD FMA
// width) and GPUTFLOPS are estimates for this model, not measurements.
func C2CPreset(arch string) (C2CHybridConfig, error) {
	c := C2CHybridConfig{Arch: arch, DoorbellLatencyNs: 250, ExpertWeightMB: 36, WeightBytesPerParam: 1, HiddenDim: 7168, OutputBytesPerElem: 2}
	switch arch {
	case "gb200":
		c.ArmCores, c.LPDDR5XBandwidthGB, c.C2CBandwidthGB = 72, 512, NVL72GraceC2CGBps
		c.ArmTFLOPS, c.GPUTFLOPS = 7, NVL72BlackwellTensorTFLOPS
	case "gb300":
		c.ArmCores, c.LPDDR5XBandwidthGB, c.C2CBandwidthGB = 72, 512, NVL72GraceC2CGBps
		c.ArmTFLOPS, c.GPUTFLOPS = 7, 2500
	case "vr200":
		c.ArmCores, c.LPDDR5XBandwidthGB, c.C2CBandwidthGB = 88, 1200, 900
		c.ArmTFLOPS, c.GPUTFLOPS = 10, 4500
	default:
		return c, fmt.Errorf("c2c hybrid: unknown arch %q (want gb200, gb300 or vr200)", arch)
	}
	return c, nil
}

func (c C2CHybridConfig) norm() (C2CHybridConfig, error) {
	if c.WeightBytesPerParam == 0 {
		c.WeightBytesPerParam = 1
	}
	if c.OutputBytesPerElem == 0 {
		c.OutputBytesPerElem = 2
	}
	if c.LPDDR5XBandwidthGB <= 0 || c.C2CBandwidthGB <= 0 || c.ArmTFLOPS <= 0 || c.GPUTFLOPS <= 0 ||
		c.ExpertWeightMB <= 0 || c.DoorbellLatencyNs < 0 || c.HiddenDim < 1 || c.WeightBytesPerParam <= 0 {
		return c, fmt.Errorf("c2c hybrid: bandwidths, TFLOPS, ExpertWeightMB and HiddenDim must be positive")
	}
	return c, nil
}

// ColdMoEDecision compares evaluating a cold expert in place on the CPU,
// next to its weights in LPDDR5X, with copying the weights over C2C to the
// GPU.
type ColdMoEDecision struct {
	TokenCount       int
	RunOnArmInPlace  bool
	ArmInPlaceTimeUs float64 // 2 doorbells + max(weight read, CPU math) + activations over C2C
	GPUDMATimeUs     float64 // weights over C2C + GPU math
	BandwidthSavedMB float64 // C2C bytes not moved when running on the CPU
}

// DecideColdMoEPlacement evaluates the cost model for tokenCount tokens.
func DecideColdMoEPlacement(cfg C2CHybridConfig, tokenCount int) (ColdMoEDecision, error) {
	cfg, err := cfg.norm()
	if err != nil {
		return ColdMoEDecision{}, err
	}
	if tokenCount < 1 {
		return ColdMoEDecision{}, fmt.Errorf("c2c hybrid: tokenCount %d must be positive", tokenCount)
	}
	weightBytes := cfg.ExpertWeightMB * 1e6
	flops := 2 * float64(tokenCount) * weightBytes / cfg.WeightBytesPerParam
	actBytes := 2 * float64(tokenCount*cfg.HiddenDim*cfg.OutputBytesPerElem) // input in, output back
	us := func(bytes, gbps float64) float64 { return bytes / (gbps * 1e3) }
	d := ColdMoEDecision{TokenCount: tokenCount}
	d.ArmInPlaceTimeUs = 2*cfg.DoorbellLatencyNs/1e3 +
		max(us(weightBytes, cfg.LPDDR5XBandwidthGB), flops/(cfg.ArmTFLOPS*1e6)) +
		us(actBytes, cfg.C2CBandwidthGB)
	d.GPUDMATimeUs = us(weightBytes, cfg.C2CBandwidthGB) + flops/(cfg.GPUTFLOPS*1e6)
	d.RunOnArmInPlace = d.ArmInPlaceTimeUs < d.GPUDMATimeUs
	if d.RunOnArmInPlace {
		d.BandwidthSavedMB = (weightBytes - actBytes) / 1e6
	}
	return d, nil
}

// ColdMoECrossover returns the smallest token count at which copying the
// expert to the GPU is no slower than running it on the CPU, or 0 if the
// CPU is never faster.
func ColdMoECrossover(cfg C2CHybridConfig) (int, error) {
	first, err := DecideColdMoEPlacement(cfg, 1)
	if err != nil || !first.RunOnArmInPlace {
		return 0, err
	}
	for m := 2; m <= 1<<20; m++ {
		d, err := DecideColdMoEPlacement(cfg, m)
		if err != nil {
			return 0, err
		}
		if !d.RunOnArmInPlace {
			return m, nil
		}
	}
	return 0, fmt.Errorf("c2c hybrid: no crossover below 2^20 tokens")
}

// GrammarOverlap is the timing of one decode step whose next-token grammar
// bitmask is computed on the CPU.
type GrammarOverlap struct {
	ReadyUs  float64 // bitmask ready, from the start of the step
	NeededUs float64 // LM head reads it, from the start of the step
	StallUs  float64 // GPU wait, max(0, Ready - Needed)
}

// PlanGrammarOverlap models running a grammar DFA advance on the CPU while
// the GPU runs a step's transformer layers. The DFA can only advance once
// the previous token is sampled, so the work starts with the step: it costs
// a doorbell, armTaskUs on the CPU, and a completion doorbell, and is hidden
// when that fits within layersUs, the time until the LM head.
func PlanGrammarOverlap(cfg C2CHybridConfig, layersUs, armTaskUs float64) (GrammarOverlap, error) {
	if layersUs < 0 || armTaskUs < 0 || cfg.DoorbellLatencyNs < 0 {
		return GrammarOverlap{}, fmt.Errorf("c2c hybrid: negative time")
	}
	o := GrammarOverlap{ReadyUs: 2*cfg.DoorbellLatencyNs/1e3 + armTaskUs, NeededUs: layersUs}
	o.StallUs = max(0, o.ReadyUs-o.NeededUs)
	return o, nil
}

// C2CDoorbellSlot is the 64-byte, cache-line-aligned descriptor the GPU
// and a CPU worker exchange. The GPU fills the fields, then publishes SeqNo
// with st.release.sys; the worker acquires SeqNo, runs the task, and
// publishes CompletionSeq with a release store that the GPU acquires.
type C2CDoorbellSlot struct {
	SeqNo         uint64
	TaskKind      C2CCoExecTaskKind
	NumTokens     uint32
	InputAddr     uint64
	WeightAddr    uint64
	OutputAddr    uint64
	CompletionSeq uint64
}

// Byte offsets of the C2CDoorbellSlot fields; bytes 48..63 are reserved.
const (
	C2CDoorbellSlotBytes = 64
	C2CSlotSeqNo         = 0
	C2CSlotTaskKind      = 8
	C2CSlotNumTokens     = 12
	C2CSlotInputAddr     = 16
	C2CSlotWeightAddr    = 24
	C2CSlotOutputAddr    = 32
	C2CSlotCompletionSeq = 40
)

// MarshalBinary encodes s in its 64-byte little-endian layout.
func (s C2CDoorbellSlot) MarshalBinary() ([]byte, error) {
	b := make([]byte, C2CDoorbellSlotBytes)
	le := binary.LittleEndian
	le.PutUint64(b[C2CSlotSeqNo:], s.SeqNo)
	le.PutUint32(b[C2CSlotTaskKind:], uint32(s.TaskKind))
	le.PutUint32(b[C2CSlotNumTokens:], s.NumTokens)
	le.PutUint64(b[C2CSlotInputAddr:], s.InputAddr)
	le.PutUint64(b[C2CSlotWeightAddr:], s.WeightAddr)
	le.PutUint64(b[C2CSlotOutputAddr:], s.OutputAddr)
	le.PutUint64(b[C2CSlotCompletionSeq:], s.CompletionSeq)
	return b, nil
}

// UnmarshalBinary decodes a 64-byte slot.
func (s *C2CDoorbellSlot) UnmarshalBinary(b []byte) error {
	if len(b) != C2CDoorbellSlotBytes {
		return fmt.Errorf("c2c doorbell: slot is %d bytes, want %d", len(b), C2CDoorbellSlotBytes)
	}
	le := binary.LittleEndian
	*s = C2CDoorbellSlot{
		SeqNo:         le.Uint64(b[C2CSlotSeqNo:]),
		TaskKind:      C2CCoExecTaskKind(le.Uint32(b[C2CSlotTaskKind:])),
		NumTokens:     le.Uint32(b[C2CSlotNumTokens:]),
		InputAddr:     le.Uint64(b[C2CSlotInputAddr:]),
		WeightAddr:    le.Uint64(b[C2CSlotWeightAddr:]),
		OutputAddr:    le.Uint64(b[C2CSlotOutputAddr:]),
		CompletionSeq: le.Uint64(b[C2CSlotCompletionSeq:]),
	}
	return nil
}

// EmitC2CSysDoorbellPTX returns a PTX function
//
//	.func (.reg .u32 status) funcName(.reg .u64 slot, .reg .u64 seq, .reg .u32 kind,
//	    .reg .u32 tokens, .reg .u64 in, .reg .u64 w, .reg .u64 out, .reg .u64 timeout_ns)
//
// that fills a [C2CDoorbellSlot], publishes seq with st.release.sys, and
// polls CompletionSeq with ld.acquire.sys until it equals seq (status 0) or
// %globaltimer passes timeout_ns (status 1). Both accesses must be .sys:
// the peer is a CPU thread, outside the GPU's .gpu scope.
func EmitC2CSysDoorbellPTX(funcName string) (string, error) {
	if !cIdent(funcName) {
		return "", fmt.Errorf("c2c doorbell: bad function name %q", funcName)
	}
	var b strings.Builder
	fmt.Fprintf(&b, ".func (.reg .u32 %%status) %s(\n", funcName)
	b.WriteString("\t.reg .u64 %slot, .reg .u64 %seq, .reg .u32 %kind, .reg .u32 %tokens,\n")
	b.WriteString("\t.reg .u64 %in, .reg .u64 %w, .reg .u64 %out, .reg .u64 %timeout_ns)\n{\n")
	b.WriteString("\t.reg .u64 %t0, %t, %done, %dt;\n\t.reg .pred %p_done, %p_late;\n")
	fmt.Fprintf(&b, "\tst.global.u32 [%%slot+%d], %%kind;\n", C2CSlotTaskKind)
	fmt.Fprintf(&b, "\tst.global.u32 [%%slot+%d], %%tokens;\n", C2CSlotNumTokens)
	fmt.Fprintf(&b, "\tst.global.u64 [%%slot+%d], %%in;\n", C2CSlotInputAddr)
	fmt.Fprintf(&b, "\tst.global.u64 [%%slot+%d], %%w;\n", C2CSlotWeightAddr)
	fmt.Fprintf(&b, "\tst.global.u64 [%%slot+%d], %%out;\n", C2CSlotOutputAddr)
	b.WriteString("\tmov.u64 %t0, %globaltimer;\n")
	fmt.Fprintf(&b, "\tst.release.sys.global.u64 [%%slot+%d], %%seq;\n", C2CSlotSeqNo)
	b.WriteString("$poll:\n")
	fmt.Fprintf(&b, "\tld.acquire.sys.global.u64 %%done, [%%slot+%d];\n", C2CSlotCompletionSeq)
	b.WriteString("\tsetp.eq.u64 %p_done, %done, %seq;\n\t@%p_done bra $ok;\n")
	b.WriteString("\tmov.u64 %t, %globaltimer;\n\tsub.u64 %dt, %t, %t0;\n")
	b.WriteString("\tsetp.gt.u64 %p_late, %dt, %timeout_ns;\n\t@%p_late bra $late;\n")
	b.WriteString("\tnanosleep.u32 32;\n\tbra $poll;\n")
	b.WriteString("$ok:\n\tmov.u32 %status, 0;\n\tret;\n")
	b.WriteString("$late:\n\tmov.u32 %status, 1;\n\tret;\n}\n")
	return b.String(), nil
}

var c2cMemOpRE = regexp.MustCompile(`\b(?:st|ld|atom|red|fence)(?:\.[A-Za-z0-9_:]+)+`)

// AuditC2CCoherence checks the memory-ordering instructions in src, PTX
// text or CUDA source with inline PTX: every release store, acquire load,
// atomic, reduction and fence must be .sys scoped, and src must publish
// with st.release.sys and observe with ld.acquire.sys. A .gpu, .cluster or
// .cta scope orders the access only among GPU threads, so a CPU worker
// could see the doorbell before the descriptor it guards; atom and red
// without a scope default to .gpu.
func AuditC2CCoherence(src string) error {
	var release, acquire bool
	for _, inst := range c2cMemOpRE.FindAllString(src, -1) {
		q := strings.Split(inst, ".")
		op, quals := q[0], q[1:]
		has := func(x string) bool { return slices.Contains(quals, x) }
		ordered := op == "atom" || op == "red" || op == "fence" ||
			(op == "st" && has("release")) || (op == "ld" && has("acquire"))
		if !ordered {
			continue
		}
		scope := ""
		for _, x := range quals {
			switch x {
			case "cta", "cluster", "gpu", "sys":
				scope = x
			}
		}
		if scope == "" {
			return fmt.Errorf("c2c coherence: %s has no scope, want .sys", inst)
		}
		if scope != "sys" {
			return fmt.Errorf("c2c coherence: %s uses .%s scope, want .sys", inst, scope)
		}
		release = release || (op == "st" && has("release"))
		acquire = acquire || (op == "ld" && has("acquire"))
	}
	switch {
	case !release:
		return fmt.Errorf("c2c coherence: no st.release.sys publishes the doorbell")
	case !acquire:
		return fmt.Errorf("c2c coherence: no ld.acquire.sys observes completion")
	}
	return nil
}

// C2CDoorbellKernel emits a persistent GPU kernel and a CPU worker that
// ping-pong Rounds cold-expert tasks through one [C2CDoorbellSlot] without
// relaunching: per round the kernel posts a B=1 matrix-vector task, waits
// for completion with ld.acquire.sys under a %globaltimer timeout, and
// records the round trip in nanoseconds. The worker (eg_c2c_worker, host
// C++) acquires each sequence number, computes y = W x from memory it owns,
// and releases CompletionSeq.
//
// On the GPU the slot, x and y must be in memory both processors address
// coherently (cudaHostAllocMapped, or system memory on Grace and Vera).
// Under cudahostemu the .sys accesses are C++ acquire and release atomics
// on host memory, so the emulation checks the protocol, not C2C latency;
// the worker pauses after each output row there, so a completion released
// before the output is written is observed by the kernel.
type C2CDoorbellKernel struct {
	Name      string
	Rounds    int
	Rows      int // expert output width
	Cols      int // expert input width
	TimeoutNs uint64
}

func (k C2CDoorbellKernel) norm() (C2CDoorbellKernel, error) {
	if k.Name == "" {
		k.Name = "nvl72_c2c_doorbell"
	}
	if k.TimeoutNs == 0 {
		k.TimeoutNs = 2_000_000_000
	}
	switch {
	case !cIdent(k.Name):
		return k, fmt.Errorf("c2c doorbell: bad kernel name %q", k.Name)
	case k.Rounds < 1 || k.Rows < 1 || k.Cols < 1:
		return k, fmt.Errorf("c2c doorbell: Rounds %d, Rows %d, Cols %d must be positive", k.Rounds, k.Rows, k.Cols)
	}
	return k, nil
}

// CUDA returns the kernel and the worker. Kernel parameters: slot (8 u64),
// x [Rounds][Cols], y [Rounds][Rows], lat_ns [Rounds] u64, ysum [Rounds]
// (the kernel's sum of each round's y, read after the acquire), status (u32:
// 0 ok, 1 timeout). Worker: eg_c2c_worker(slot, w [Rows][Cols]) returns 0 on
// success and 1 on its own timeout.
func (k C2CDoorbellKernel) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: %d-round GPU/CPU .sys doorbell ping-pong, %dx%d cold-expert GEMV per round.\n", k.Name, k.Rounds, k.Rows, k.Cols)
	b.WriteString(`#if !defined(EG_HOST_EMU)
#include <chrono>
#include <thread>
#endif
EG_DEV void eg_c2c_st_release_sys_u64(unsigned long long* p, unsigned long long v) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#else
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
#endif
}
EG_DEV unsigned long long eg_c2c_ld_acquire_sys_u64(const unsigned long long* p) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#else
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
#endif
}
// eg_c2c_spin_pause yields the host thread under emulation, where the GPU
// thread and the worker share CPUs; on hardware both sides spin.
EG_DEV void eg_c2c_spin_pause() {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    std::this_thread::yield();
#endif
}
EG_DEV unsigned long long eg_c2c_globaltimer() {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}
`)
	fmt.Fprintf(&b, "constexpr int EG_C2C_ROUNDS = %d, EG_C2C_ROWS = %d, EG_C2C_COLS = %d;\n", k.Rounds, k.Rows, k.Cols)
	fmt.Fprintf(&b, "constexpr unsigned long long EG_C2C_TIMEOUT_NS = %dull;\n", k.TimeoutNs)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s(unsigned long long* slot, const float* x, float* y,\n", k.Name)
	fmt.Fprintf(&b, "    unsigned long long* lat_ns, float* ysum, unsigned int* status) {\n")
	fmt.Fprintf(&b, `    if (threadIdx.x != 0) return;
    unsigned int* s32 = reinterpret_cast<unsigned int*>(slot);
    *status = 0u;
    for (int r = 0; r < EG_C2C_ROUNDS; ++r) {
        const unsigned long long seq = (unsigned long long)r + 1ull;
        s32[%[1]d / 4] = %[2]du;
        s32[%[3]d / 4] = 1u;
        slot[%[4]d / 8] = (unsigned long long)(x + (size_t)r * EG_C2C_COLS);
        slot[%[5]d / 8] = (unsigned long long)(y + (size_t)r * EG_C2C_ROWS);
        const unsigned long long t0 = eg_c2c_globaltimer();
        eg_c2c_st_release_sys_u64(slot + %[6]d / 8, seq);
        while (eg_c2c_ld_acquire_sys_u64(slot + %[7]d / 8) != seq) {
            if (eg_c2c_globaltimer() - t0 > EG_C2C_TIMEOUT_NS) {
                *status = 1u;
                return;
            }
            eg_c2c_spin_pause();
        }
        lat_ns[r] = eg_c2c_globaltimer() - t0;
        // Consume the CPU's output: valid only if the acquire saw it.
        float acc = 0.0f;
        for (int i = 0; i < EG_C2C_ROWS; ++i) acc += y[(size_t)r * EG_C2C_ROWS + i];
        ysum[r] = acc;
    }
}

// eg_c2c_worker serves EG_C2C_ROUNDS doorbells on the calling CPU thread.
inline int eg_c2c_worker(unsigned long long* slot, const float* w) {
    const unsigned int* s32 = reinterpret_cast<const unsigned int*>(slot);
    for (unsigned long long seq = 1; seq <= (unsigned long long)EG_C2C_ROUNDS; ++seq) {
        const auto t0 = std::chrono::steady_clock::now();
        while (__atomic_load_n(slot + %[6]d / 8, __ATOMIC_ACQUIRE) != seq) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::nanoseconds(EG_C2C_TIMEOUT_NS)) return 1;
#if defined(EG_HOST_EMU) || defined(EG_EMU)
            std::this_thread::yield();
#endif
        }
        if (s32[%[1]d / 4] != %[2]du || s32[%[3]d / 4] != 1u) return 2;
        const float* in = reinterpret_cast<const float*>(slot[%[4]d / 8]);
        float* out = reinterpret_cast<float*>(slot[%[5]d / 8]);
        for (int i = 0; i < EG_C2C_ROWS; ++i) {
            float acc = 0.0f;
            for (int j = 0; j < EG_C2C_COLS; ++j) { const float p = w[(size_t)i * EG_C2C_COLS + j] * in[j]; acc += p; }
            out[i] = acc;
#if defined(EG_HOST_EMU) || defined(EG_EMU)
            // Write the output slowly, so that a completion published
            // before the output is visible to the kernel as a wrong sum.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
        }
        __atomic_store_n(slot + %[7]d / 8, seq, __ATOMIC_RELEASE);
    }
    return 0;
}
`, C2CSlotTaskKind, uint32(TaskColdMoESingleToken), C2CSlotNumTokens, C2CSlotInputAddr, C2CSlotOutputAddr, C2CSlotSeqNo, C2CSlotCompletionSeq)
	return b.String(), nil
}

// Source returns [Prelude] and the kernel and worker.
func (k C2CDoorbellKernel) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src, nil
}
