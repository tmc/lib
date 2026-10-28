package nvl72

import (
	"fmt"
	"math"
)

// An AllReduceAlgo is an all-reduce algorithm over one NVLink domain.
type AllReduceAlgo uint8

const (
	// AllReduceOneShot: every rank reads every peer's buffer and sums
	// (one hop; each rank takes in (N-1)·S bytes). The rankgroup
	// DeviceAllReduce kernel is this algorithm.
	AllReduceOneShot AllReduceAlgo = iota + 1
	// AllReduceTwoShot: reduce-scatter then all-gather through peer
	// pointers (two hops; 2·(N-1)/N·S bytes in per rank).
	AllReduceTwoShot
	// AllReduceRing: NCCL's ring (2·(N-1) hops; 2·(N-1)/N·S bytes).
	AllReduceRing
	// AllReduceNVLS: NVLink SHARP, multimem.ld_reduce then multimem.st of
	// each rank's 1/N slice (two hops; the switch reduces and replicates,
	// so about S bytes cross each rank's links per direction).
	AllReduceNVLS
)

func (a AllReduceAlgo) String() string {
	switch a {
	case AllReduceOneShot:
		return "one-shot"
	case AllReduceTwoShot:
		return "two-shot"
	case AllReduceRing:
		return "ring"
	case AllReduceNVLS:
		return "nvls"
	}
	return fmt.Sprintf("AllReduceAlgo(%d)", uint8(a))
}

// An NVLinkFabric is the cost model of one GPU's NVLink domain and memory.
// Every field is a model input, not a measurement.
type NVLinkFabric struct {
	Name string

	// LinkGBps is the per-GPU unidirectional NVLink bandwidth (900 for
	// NVLink 5); HopUs the latency of one dependent NVLink step: a peer
	// write or switch reduction plus the flag that tells the receiver.
	LinkGBps float64
	HopUs    float64
	NVLS     bool // in-switch reduction (multimem) available

	HBMGBps       float64 // per-GPU HBM bandwidth
	HBMEfficiency float64 // fraction of HBMGBps a decode GEMV streams
	TensorTFLOPS  float64 // per-GPU dense BF16

	// KernelGapUs is the idle time between dependent kernels of a step
	// replayed from a CUDA graph; a persistent megakernel has none.
	// CollectiveUs is the extra cost of a separate all-reduce kernel
	// (launch, flag reset); an all-reduce fused into the GEMM epilogue
	// does not pay it.
	KernelGapUs  float64
	CollectiveUs float64
}

// GB300NVLink5 returns the model of a GB300 NVL72 GPU: HBM3e at 8 TB/s,
// NVLink 5 at 900 GB/s per direction through NVSwitch with NVLS. HopUs
// (1 µs), KernelGapUs (1 µs), CollectiveUs (2 µs) and HBMEfficiency (0.8)
// are assumptions, not datasheet values; nothing here was measured on
// GB300.
func GB300NVLink5() NVLinkFabric {
	return NVLinkFabric{
		Name:          "GB300 NVL72 (NVLink 5)",
		LinkGBps:      NVL72NVLink5UniGBps,
		HopUs:         1.0,
		NVLS:          true,
		HBMGBps:       NVL72HBM3eGBps,
		HBMEfficiency: 0.8,
		TensorTFLOPS:  GB300TensorTFLOPS,
		KernelGapUs:   1.0,
		CollectiveUs:  2.0,
	}
}

// GB10 returns the model of one DGX Spark GB10 (LPDDR5X at about 273 GB/s
// peak, no NVLink), the device this repository executes on. Its
// HBMEfficiency is set so the model's TP=1 Llama-3-8B BF16 step matches a
// decode floor of about 63 ms measured on it.
func GB10() NVLinkFabric {
	return NVLinkFabric{
		Name:          "GB10 (DGX Spark)",
		HBMGBps:       273,
		HBMEfficiency: 0.87,
		TensorTFLOPS:  100,
		KernelGapUs:   2.0,
		CollectiveUs:  2.0,
	}
}

// AllReduceUs models the time of algo summing bytes per rank over ranks
// GPUs. One rank costs nothing. NVLS needs f.NVLS.
func (f NVLinkFabric) AllReduceUs(algo AllReduceAlgo, ranks int, bytes float64) (float64, error) {
	if ranks < 1 || bytes < 0 {
		return 0, fmt.Errorf("nvl72: all-reduce of %v bytes over %d ranks", bytes, ranks)
	}
	if ranks == 1 {
		return 0, nil
	}
	if f.LinkGBps <= 0 {
		return 0, fmt.Errorf("nvl72: %s has no NVLink", f.Name)
	}
	n := float64(ranks)
	bw := f.LinkGBps * 1e3 // bytes per µs
	switch algo {
	case AllReduceOneShot:
		return f.HopUs + (n-1)*bytes/bw, nil
	case AllReduceTwoShot:
		return 2*f.HopUs + 2*(n-1)/n*bytes/bw, nil
	case AllReduceRing:
		return 2*(n-1)*f.HopUs + 2*(n-1)/n*bytes/bw, nil
	case AllReduceNVLS:
		if !f.NVLS {
			return 0, fmt.Errorf("nvl72: %s has no NVLS", f.Name)
		}
		return 2*f.HopUs + bytes/bw, nil
	}
	return 0, fmt.Errorf("nvl72: bad all-reduce algorithm %v", algo)
}

// BestAllReduce returns the fastest available algorithm for bytes over
// ranks, as NCCL's tuner picks one.
func (f NVLinkFabric) BestAllReduce(ranks int, bytes float64) (AllReduceAlgo, float64, error) {
	best, bestUs := AllReduceAlgo(0), math.Inf(1)
	for _, a := range []AllReduceAlgo{AllReduceOneShot, AllReduceTwoShot, AllReduceRing, AllReduceNVLS} {
		us, err := f.AllReduceUs(a, ranks, bytes)
		if err != nil {
			if a == AllReduceNVLS || ranks > 1 && f.LinkGBps <= 0 {
				continue
			}
			return 0, 0, err
		}
		if us < bestUs {
			best, bestUs = a, us
		}
	}
	if best == 0 {
		return 0, 0, fmt.Errorf("nvl72: no all-reduce over %d ranks on %s", ranks, f.Name)
	}
	return best, bestUs, nil
}

// A DenseLlama is a dense LLaMA decoder's shape.
type DenseLlama struct {
	Name                           string
	Layers, Hidden, Heads, KVHeads int
	HeadDim, Intermediate, Vocab   int
}

// Llama3_8B and Llama3_70B are the Llama 3 shapes.
var (
	Llama3_8B  = DenseLlama{Name: "Llama-3-8B", Layers: 32, Hidden: 4096, Heads: 32, KVHeads: 8, HeadDim: 128, Intermediate: 14336, Vocab: 128256}
	Llama3_70B = DenseLlama{Name: "Llama-3-70B", Layers: 80, Hidden: 8192, Heads: 64, KVHeads: 8, HeadDim: 128, Intermediate: 28672, Vocab: 128256}
)

// LayerParams returns the parameters of one decoder layer.
func (m DenseLlama) LayerParams() float64 {
	q, kv := m.Heads*m.HeadDim, m.KVHeads*m.HeadDim
	return float64(m.Hidden*(q+2*kv) + q*m.Hidden + 3*m.Hidden*m.Intermediate + 2*m.Hidden)
}

// Params returns the model's parameters, with untied embedding and head.
func (m DenseLlama) Params() float64 {
	return float64(m.Layers)*m.LayerParams() + float64(2*m.Vocab*m.Hidden+m.Hidden)
}

// TPDecodeConfig is one tensor-parallel decode step to model.
type TPDecodeConfig struct {
	Model       DenseLlama
	TP          int
	Batch       int     // sequences decoded together; 0 means 1
	Context     int     // KV positions attended per sequence
	WeightBytes float64 // bytes per weight (2 BF16, 1 FP8, 0.5625 NVFP4 with scales); 0 means 2
	KVBytes     float64 // bytes per cached K or V value; 0 means 2

	// KernelsPerLayer are the dependent kernels of one layer per rank,
	// not counting all-reduces; 0 means 9 (the llama-infer e-graph decode
	// plans). Megakernel removes every gap.
	KernelsPerLayer int
	Megakernel      bool
	// FuseAllReduce fuses each all-reduce into its GEMM epilogue (NVLS
	// multimem.red): no separate collective kernel or gap.
	FuseAllReduce bool
}

// TPDecodeEstimate is the modelled time of one decode step on each rank.
// All fields are model outputs.
type TPDecodeEstimate struct {
	Config        TPDecodeConfig
	Fabric        string
	WeightGB      float64 // weight bytes per rank
	WeightUs      float64 // streaming the rank's weights
	KVUs          float64 // reading the rank's KV cache
	ComputeUs     float64 // tensor-core time, if not hidden (it is, at decode batch)
	GapUs         float64 // dependent-kernel gaps
	AllReduceAlgo AllReduceAlgo
	AllReduceUs   float64 // all 2·layers all-reduces, plus the head's
	StepUs        float64
	TokensPerSec  float64 // per sequence
}

// EstimateTPDecode models one decode step of c on fabric f: each rank
// streams its shard of every layer (heads/TP of attention, intermediate/TP
// of the MLP, a vocab-parallel head with a replicated embedding row), reads
// its KV heads' cache, and joins 2·layers + 1 all-reduces of Batch·Hidden
// BF16 values (the +1 reduces the head's argmax). Memory time and
// all-reduces add; tensor-core time overlaps memory and counts only when
// larger.
func EstimateTPDecode(f NVLinkFabric, c TPDecodeConfig) (TPDecodeEstimate, error) {
	m := c.Model
	if c.Batch == 0 {
		c.Batch = 1
	}
	if c.WeightBytes == 0 {
		c.WeightBytes = 2
	}
	if c.KVBytes == 0 {
		c.KVBytes = 2
	}
	if c.KernelsPerLayer == 0 {
		c.KernelsPerLayer = 9
	}
	switch {
	case c.TP < 1 || m.KVHeads%min(c.TP, m.KVHeads) != 0 || m.Intermediate%c.TP != 0 || m.Heads%c.TP != 0:
		return TPDecodeEstimate{}, fmt.Errorf("nvl72: tp %d does not shard %s", c.TP, m.Name)
	case c.Batch < 1 || c.Context < 0:
		return TPDecodeEstimate{}, fmt.Errorf("nvl72: bad batch %d or context %d", c.Batch, c.Context)
	case f.HBMGBps <= 0 || f.HBMEfficiency <= 0 || f.TensorTFLOPS <= 0:
		return TPDecodeEstimate{}, fmt.Errorf("nvl72: fabric %s lacks memory or compute rates", f.Name)
	}
	tp := float64(c.TP)
	e := TPDecodeEstimate{Config: c, Fabric: f.Name}
	params := float64(m.Layers)*m.LayerParams()/tp + float64(m.Vocab*m.Hidden)/tp
	weightBytes := params*c.WeightBytes + float64(c.Batch*m.Hidden)*c.WeightBytes // + embedding rows
	e.WeightGB = weightBytes / 1e9
	hbm := f.HBMGBps * f.HBMEfficiency * 1e3 // bytes per µs
	e.WeightUs = weightBytes / hbm
	// Beyond KVHeads ranks, KV heads are replicated.
	kvShard := float64(min(c.TP, m.KVHeads))
	kvBytes := 2 * float64(m.Layers*m.KVHeads*m.HeadDim*c.Context*c.Batch) * c.KVBytes / kvShard
	e.KVUs = kvBytes / hbm
	flops := 2 * params * float64(c.Batch)
	if c := e.WeightUs + e.KVUs; flops/(f.TensorTFLOPS*1e6) > c {
		e.ComputeUs = flops/(f.TensorTFLOPS*1e6) - c
	}
	collectives := 2*m.Layers + 1
	if !c.Megakernel {
		kernels := m.Layers*c.KernelsPerLayer + 2 // + embedding, head
		if !c.FuseAllReduce && c.TP > 1 {
			kernels += collectives
		}
		e.GapUs = float64(kernels) * f.KernelGapUs
	}
	if c.TP > 1 {
		algo, us, err := f.BestAllReduce(c.TP, float64(c.Batch*m.Hidden)*2)
		if err != nil {
			return TPDecodeEstimate{}, err
		}
		if !c.FuseAllReduce {
			us += f.CollectiveUs
		}
		e.AllReduceAlgo = algo
		e.AllReduceUs = float64(collectives) * us
	}
	e.StepUs = e.WeightUs + e.KVUs + e.ComputeUs + e.GapUs + e.AllReduceUs
	e.TokensPerSec = 1e6 / e.StepUs
	return e, nil
}
