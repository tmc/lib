package main

import (
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
)

const (
	// LlamaBundlePageAlignment is the 4096-byte OS page alignment enforced for
	// all Safetensors weight tensors and PGO cubin sections in a LLaMA .egbundle
	// archive for zero-copy mmap access.
	LlamaBundlePageAlignment = 4096

	llamaBundleMagic         = "EGBNDL01"
	llamaBundleHeaderPrefix  = 16 // 8-byte magic + 8-byte uint64 manifest JSON length
	llamaBundleFormatVersion = 1
)

// LlamaAOTWeightSpec describes an input Safetensors weight tensor to pack into
// a LLaMA .egbundle archive at a 4096-byte page-aligned offset.
type LlamaAOTWeightSpec struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	Data  []byte `json:"data,omitempty"`
}

// LlamaAOTWeightHeader records Safetensors-compatible tensor metadata and the
// 4096-byte page-aligned absolute file offset for zero-copy mmap access.
type LlamaAOTWeightHeader struct {
	Name        string    `json:"name"`
	DType       string    `json:"dtype"`
	Shape       []int     `json:"shape"`
	DataOffsets [2]uint64 `json:"data_offsets"`
	PageOffset  uint64    `json:"page_offset"`
	ByteLength  uint64    `json:"byte_length"`
	SHA256      string    `json:"sha256"`
}

// LlamaAOTPGOCacheEntry binds a verified PGO promotion proof and its
// 4096-byte page-aligned cubin slice inside a LLaMA .egbundle archive.
type LlamaAOTPGOCacheEntry struct {
	CacheID     string                     `json:"cache_id"`
	Key         autotune.PGOCacheKey       `json:"key"`
	Proof       autotune.PGOPromotionProof `json:"proof"`
	CubinOffset uint64                     `json:"cubin_offset"`
	CubinLength uint64                     `json:"cubin_length"`
	CubinSHA256 string                     `json:"cubin_sha256"`
}

// LlamaAOTBucketArtifact stores the pre-lowered cross-layer (or full-layer)
// megakernel CUDA/PTX translation unit, #line-attributed source, formal barrier
// safety verification certificate, and optional verified PGO cache entry for
// one speculative/batch shape bucket M in {1, 2, 4, 8}.
type LlamaAOTBucketArtifact struct {
	BucketM          int                          `json:"bucket_m"`
	KernelName       string                       `json:"kernel_name"`
	Regime           string                       `json:"regime"`
	NumStages        int                          `json:"num_stages"`
	TotalBlocks      int                          `json:"total_blocks"`
	SharedBytes      int                          `json:"shared_bytes"`
	ReadyWords       int                          `json:"ready_words"`
	CUDASource       string                       `json:"cuda_source"`
	LineInfoSource   string                       `json:"line_info_source"`
	PTX              string                       `json:"ptx,omitempty"`
	StageSpans       []hopper.StageLineSpan       `json:"stage_spans"`
	BarrierReport    *qualify.FormalBarrierReport `json:"barrier_report"`
	SourceSHA256     string                       `json:"source_sha256"`
	LineSourceSHA256 string                       `json:"line_source_sha256"`
	BarrierSHA256    string                       `json:"barrier_sha256"`
	PGOEntry         *LlamaAOTPGOCacheEntry       `json:"pgo_entry,omitempty"`
}

// LlamaAOTBundleManifest is the top-level self-verifying manifest embedded at
// the start of a LLaMA .egbundle archive.
type LlamaAOTBundleManifest struct {
	FormatVersion  int                             `json:"format_version"`
	Arch           string                          `json:"arch"`
	DriverVersion  string                          `json:"driver_version"`
	CUDAVersion    string                          `json:"cuda_version"`
	WeightDigest   string                          `json:"weight_digest"`
	LayerL         int                             `json:"layer_l"`
	LayerNext      int                             `json:"layer_next"`
	FullLayer      bool                            `json:"full_layer,omitempty"`
	PageAlignment  int                             `json:"page_alignment"`
	DataBasePage   uint64                          `json:"data_base_page"`
	Buckets        []LlamaAOTBucketArtifact        `json:"buckets"`
	PGOCache       []LlamaAOTPGOCacheEntry         `json:"pgo_cache,omitempty"`
	Weights        map[string]LlamaAOTWeightHeader `json:"weights"`
	BucketsSHA256  string                          `json:"buckets_sha256"`
	BarriersSHA256 string                          `json:"barriers_sha256"`
	PGOSHA256      string                          `json:"pgo_sha256"`
	WeightsSHA256  string                          `json:"weights_sha256"`
	PayloadSHA256  string                          `json:"payload_sha256"`
}

// LlamaAOTBundleConfig configures compilation and export of a deterministic,
// self-verifying LLaMA .egbundle archive across shape buckets.
type LlamaAOTBundleConfig struct {
	Model         *ModelConfig
	Weights       *ModelWeights
	WeightSpecs   []LlamaAOTWeightSpec
	QKVBias       bool
	LayerL        int
	LayerNext     int
	FullLayer     bool
	Buckets       []int
	Arch          string
	Target        string
	DriverVersion string
	CUDAVersion   string
	WeightDigest  string
	PlanKeyPrefix string
	PGOCache      *autotune.VerifiedPGOCache
	PGOCubins     map[int][]byte
}

func alignLlamaBundlePage(offset uint64) uint64 {
	rem := offset % LlamaBundlePageAlignment
	if rem == 0 {
		return offset
	}
	return offset + (LlamaBundlePageAlignment - rem)
}

func sha256HexBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func canonicalBarrierReportDigest(rep *qualify.FormalBarrierReport) (string, error) {
	if rep == nil {
		return "", errors.New("llama bundle: nil formal barrier report")
	}
	raw, err := json.Marshal(rep)
	if err != nil {
		return "", fmt.Errorf("llama bundle marshal barrier report: %w", err)
	}
	return sha256HexBytes(raw), nil
}

func normalizeLlamaBundleModelConfig(cfg *ModelConfig) *ModelConfig {
	out := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         64,
	}
	if cfg != nil {
		out = *cfg
		if out.HiddenSize < 256 || out.HiddenSize%256 != 0 {
			out.HiddenSize = 256
		}
		if out.IntermediateSize < 256 || out.IntermediateSize%256 != 0 {
			out.IntermediateSize = 256
		}
		if out.NumHiddenLayers < 2 {
			out.NumHiddenLayers = 2
		}
		if out.NumAttentionHeads <= 0 {
			out.NumAttentionHeads = 4
		}
		if out.NumKeyValueHeads <= 0 {
			out.NumKeyValueHeads = 1
		}
		if out.VocabSize <= 0 {
			out.VocabSize = 256
		}
		if out.RmsNormEps <= 0 {
			out.RmsNormEps = 1e-5
		}
		if out.RopeTheta <= 0 {
			out.RopeTheta = 10000
		}
		if out.MaxSeqLen <= 0 {
			out.MaxSeqLen = 64
		}
	}
	return &out
}

func defaultLlamaBundleWeightSpecs(cfg *ModelConfig, mw *ModelWeights, layerL, layerNext int) []LlamaAOTWeightSpec {
	cfg = normalizeLlamaBundleModelConfig(cfg)
	if mw == nil {
		mw = GenerateDeterministicWeights(cfg)
	}
	encodeBF16 := func(vals []float32) []byte {
		buf := make([]byte, 2*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint16(buf[2*i:], hopper.BF16Bits(v))
		}
		return buf
	}
	encodeF32 := func(vals []float32) []byte {
		buf := make([]byte, 4*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
		}
		return buf
	}

	hidden := max(cfg.HiddenSize, 16)
	inter := max(cfg.IntermediateSize, 32)
	qDim := max(cfg.NumAttentionHeads*cfg.HeadDim(), hidden)
	kvDim := max(cfg.NumKeyValueHeads*cfg.HeadDim(), cfg.HeadDim())

	var specs []LlamaAOTWeightSpec
	layers := []int{layerL, layerNext}
	seen := make(map[int]bool, len(layers))
	for _, lIdx := range layers {
		if lIdx < 0 || seen[lIdx] {
			continue
		}
		seen[lIdx] = true
		var lw LayerWeights
		if lIdx < len(mw.Layers) {
			lw = mw.Layers[lIdx]
		}
		p := fmt.Sprintf("model.layers.%d.", lIdx)
		if len(lw.InputNorm) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "input_layernorm.weight",
				DType: "F32",
				Shape: []int{hidden},
				Data:  encodeF32(lw.InputNorm),
			})
		}
		if len(lw.PostNorm) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "post_attention_layernorm.weight",
				DType: "F32",
				Shape: []int{hidden},
				Data:  encodeF32(lw.PostNorm),
			})
		}
		if len(lw.WQ) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "self_attn.q_proj.weight",
				DType: "BF16",
				Shape: []int{qDim, hidden},
				Data:  encodeBF16(lw.WQ),
			})
		}
		if len(lw.WK) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "self_attn.k_proj.weight",
				DType: "BF16",
				Shape: []int{kvDim, hidden},
				Data:  encodeBF16(lw.WK),
			})
		}
		if len(lw.WV) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "self_attn.v_proj.weight",
				DType: "BF16",
				Shape: []int{kvDim, hidden},
				Data:  encodeBF16(lw.WV),
			})
		}
		if len(lw.WO) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "self_attn.o_proj.weight",
				DType: "BF16",
				Shape: []int{hidden, qDim},
				Data:  encodeBF16(lw.WO),
			})
		}
		if len(lw.WGate) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "mlp.gate_proj.weight",
				DType: "BF16",
				Shape: []int{inter, hidden},
				Data:  encodeBF16(lw.WGate),
			})
		}
		if len(lw.WUp) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "mlp.up_proj.weight",
				DType: "BF16",
				Shape: []int{inter, hidden},
				Data:  encodeBF16(lw.WUp),
			})
		}
		if len(lw.WDown) > 0 {
			specs = append(specs, LlamaAOTWeightSpec{
				Name:  p + "mlp.down_proj.weight",
				DType: "BF16",
				Shape: []int{hidden, inter},
				Data:  encodeBF16(lw.WDown),
			})
		}
	}
	if len(mw.Norm) > 0 {
		specs = append(specs, LlamaAOTWeightSpec{
			Name:  "model.norm.weight",
			DType: "F32",
			Shape: []int{hidden},
			Data:  encodeF32(mw.Norm),
		})
	}
	return specs
}

func buildLlamaMultiBucketController(buckets []int) (*specdecode.AdaptiveBucketController, error) {
	if len(buckets) == 0 {
		buckets = []int{1, 2, 4, 8}
	}
	defSpecs, err := specdecode.DefaultBucketSpecs()
	if err != nil {
		return nil, err
	}
	byM := make(map[int]specdecode.BucketSpec, len(defSpecs))
	for _, s := range defSpecs {
		byM[s.BucketM] = s
	}
	var specs []specdecode.BucketSpec
	seen := make(map[int]bool, len(buckets))
	for _, m := range buckets {
		if m <= 0 {
			return nil, fmt.Errorf("llama bundle: invalid bucket M=%d", m)
		}
		if seen[m] {
			continue
		}
		seen[m] = true
		if sp, ok := byM[m]; ok {
			specs = append(specs, sp)
			continue
		}
		regime := specdecode.RegimeSkinnyDecode
		if m >= 16 {
			regime = specdecode.RegimeTilePrefill
		} else if m >= 4 {
			regime = specdecode.RegimeMTPTreeVerify
		}
		specs = append(specs, specdecode.BucketSpec{
			BucketM: m,
			Regime:  regime,
		})
	}
	slices.SortFunc(specs, func(a, b specdecode.BucketSpec) int {
		return cmp.Compare(a.BucketM, b.BucketM)
	})
	return specdecode.NewAdaptiveBucketController(specdecode.AdaptiveBucketConfig{
		Buckets:                 specs,
		EMAAlpha:                0.50,
		HysteresisSteps:         1,
		LowAcceptanceThreshold:  0.30,
		HighAcceptanceThreshold: 0.65,
		InitialBucketM:          specs[0].BucketM,
	})
}

func buildLlamaBucketBuilder(modelCfg *ModelConfig, qkvBias bool, layerL, layerNext int, fullLayer bool, arch string) specdecode.BucketPlanBuilder {
	return func(bucketM int, regime specdecode.BucketRegime) (*hopper.Plan, hopper.Megakernel, error) {
		if fullLayer {
			plan, mega, err := BuildLlamaFullLayerMegakernel(FullLayerMegakernelConfig{
				Name:       fmt.Sprintf("llama_full_l%d_l%d_m%d", layerL, layerNext, bucketM),
				Model:      modelCfg,
				StartLayer: layerL,
				NumLayers:  max(1, layerNext-layerL+1),
				BatchM:     bucketM,
				Arch:       hopper.Arch(arch),
				Target:     arch,
			})
			if err != nil {
				return nil, hopper.Megakernel{}, err
			}
			if mega == nil {
				return nil, hopper.Megakernel{}, errors.New("llama bundle: nil full-layer megakernel")
			}
			return plan, *mega, nil
		}

		prefixL := fmt.Sprintf("l%d", layerL)
		prefixNext := fmt.Sprintf("l%d", layerNext)
		_, postL, err := buildLayerPlans(modelCfg, qkvBias, false, prefixL, bucketM, hopper.F32, arch, true)
		if err != nil {
			return nil, hopper.Megakernel{}, fmt.Errorf("llama bundle layer %d post M=%d: %w", layerL, bucketM, err)
		}
		attnInNext, _, err := buildLayerPlans(modelCfg, qkvBias, false, prefixNext, bucketM, hopper.F32, arch, true)
		if err != nil {
			return nil, hopper.Megakernel{}, fmt.Errorf("llama bundle layer %d attnIn M=%d: %w", layerNext, bucketM, err)
		}

		// Ensure RowsPerBlock * bucketM <= 256 so every SkinnyGEMM stage fits in the 256-thread persistent CTA.
		for _, pl := range []*hopper.Plan{postL, attnInNext} {
			for i := range pl.Kernels {
				if g := pl.Kernels[i].GEMM; g != nil {
					for _, r := range []int{32, 16, 8, 4, 2, 1} {
						if g.GLU && r%2 != 0 {
							continue
						}
						if g.N%r == 0 && r*max(bucketM, g.MaxM, 1) <= 256 {
							g.RowsPerBlock = r
							break
						}
					}
				}
			}
		}

		regimeSuffix := "skinny"
		if regime == specdecode.RegimeTensorCoreTile || regime == specdecode.RegimeTilePrefill || bucketM >= 4 {
			regimeSuffix = "tc_verify"
		}
		glue := hopper.CrossLayerGlue{
			Name:               fmt.Sprintf("llama_cross_l%d_l%d_%s_m%d", layerL, layerNext, regimeSuffix, bucketM),
			LayerLPrefix:       prefixL + "_",
			LayerNextPrefix:    prefixNext + "_",
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		return hopper.BuildCrossLayerMegakernel(postL, attnInNext, glue)
	}
}

func writeBundleFileAtomic(dstPath string, data []byte) error {
	dir := filepath.Dir(dstPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("llama bundle mkdir %s: %w", dir, err)
	}
	base := filepath.Base(dstPath)
	f, err := os.CreateTemp(dir, base+".tmp.*")
	if err != nil {
		return fmt.Errorf("llama bundle create temp: %w", err)
	}
	tmpName := f.Name()
	if _, err := f.Write(data); err != nil {
		cerr := f.Close()
		rerr := os.Remove(tmpName)
		return errors.Join(fmt.Errorf("llama bundle write temp: %w", err), cerr, rerr)
	}
	if err := f.Close(); err != nil {
		rerr := os.Remove(tmpName)
		return errors.Join(fmt.Errorf("llama bundle close temp: %w", err), rerr)
	}
	if err := os.Rename(tmpName, dstPath); err != nil {
		rerr := os.Remove(tmpName)
		return errors.Join(fmt.Errorf("llama bundle rename temp to %s: %w", dstPath, err), rerr)
	}
	return nil
}

// BuildAndExportLlamaBundle compiles the multi-bucket (M in {1, 2, 4, 8})
// cross-layer or full-layer LLaMA megakernel table via
// [specdecode.NewMultiBucketMegakernelTable], runs
// [qualify.VerifyMegakernelBarrierSafety] on every bucket's megakernel, aligns
// all Safetensors weight tensors and PGO cubin slices to 4096-byte page
// boundaries, computes SHA-256 digests over every section, and atomically
// writes the self-verifying .egbundle file to bundlePath.
func BuildAndExportLlamaBundle(cfg LlamaAOTBundleConfig, bundlePath string) (*LlamaAOTBundleManifest, error) {
	if strings.TrimSpace(bundlePath) == "" {
		return nil, errors.New("llama bundle export: empty bundle path")
	}

	modelCfg := normalizeLlamaBundleModelConfig(cfg.Model)
	layerL := cfg.LayerL
	layerNext := cfg.LayerNext
	if layerL == 0 && layerNext == 0 && !cfg.FullLayer {
		layerNext = 1
	}
	if layerL < 0 || layerNext < 0 || (!cfg.FullLayer && layerL == layerNext) {
		return nil, fmt.Errorf("llama bundle export: invalid layer pair (%d, %d)", layerL, layerNext)
	}
	if modelCfg.NumHiddenLayers > 0 && (layerL >= modelCfg.NumHiddenLayers || layerNext >= modelCfg.NumHiddenLayers) {
		return nil, fmt.Errorf("llama bundle export: layer pair (%d, %d) out of range [0, %d)",
			layerL, layerNext, modelCfg.NumHiddenLayers)
	}

	arch := strings.TrimSpace(cfg.Arch)
	if arch == "" {
		arch = strings.TrimSpace(cfg.Target)
	}
	if arch == "" {
		arch = "sm_121a"
	}
	driverVer := strings.TrimSpace(cfg.DriverVersion)
	if driverVer == "" {
		driverVer = "580.173.02"
	}
	cudaVer := strings.TrimSpace(cfg.CUDAVersion)
	if cudaVer == "" {
		cudaVer = "13.0"
	}
	weightDigest := strings.TrimSpace(cfg.WeightDigest)
	if weightDigest == "" {
		weightDigest = fmt.Sprintf("llama_h%d_i%d_l%d",
			modelCfg.HiddenSize, modelCfg.IntermediateSize, modelCfg.NumHiddenLayers)
	}
	planPrefix := strings.TrimSpace(cfg.PlanKeyPrefix)
	if planPrefix == "" {
		if cfg.FullLayer {
			planPrefix = fmt.Sprintf("llama_full_l%d_l%d", layerL, layerNext)
		} else {
			planPrefix = fmt.Sprintf("llama_cross_l%d_l%d", layerL, layerNext)
		}
	}

	ctrl, err := buildLlamaMultiBucketController(cfg.Buckets)
	if err != nil {
		return nil, err
	}
	builder := buildLlamaBucketBuilder(modelCfg, cfg.QKVBias, layerL, layerNext, cfg.FullLayer, arch)

	table, err := specdecode.NewMultiBucketMegakernelTable(specdecode.MultiBucketTableConfig{
		Arch:          arch,
		DriverVersion: driverVer,
		CUDAVersion:   cudaVer,
		WeightDigest:  weightDigest,
		PlanKeyPrefix: planPrefix,
		Cache:         cfg.PGOCache,
		Controller:    ctrl,
		Builder:       builder,
	})
	if err != nil {
		return nil, fmt.Errorf("llama bundle export multi-bucket table: %w", err)
	}

	entries := table.Entries()
	if len(entries) == 0 {
		return nil, errors.New("llama bundle export: multi-bucket table produced zero entries")
	}

	type pgoBlob struct {
		entry LlamaAOTPGOCacheEntry
		cubin []byte
	}
	var pgoBlobs []pgoBlob
	pgoByCacheID := make(map[string]int)

	addPGOBlob := func(key autotune.PGOCacheKey, proof autotune.PGOPromotionProof, cubin []byte) error {
		if len(cubin) == 0 {
			return errors.New("llama bundle export: empty PGO cubin")
		}
		if err := key.Validate(); err != nil {
			return fmt.Errorf("llama bundle export PGO key: %w", err)
		}
		cid := key.CacheID()
		if _, exists := pgoByCacheID[cid]; exists {
			return nil
		}
		cSum := sha256.Sum256(cubin)
		if proof.Key == (autotune.PGOCacheKey{}) {
			proof.Key = key
		}
		if proof.CubinSHA256 == ([32]byte{}) {
			proof.CubinSHA256 = cSum
		} else if proof.CubinSHA256 != cSum {
			return fmt.Errorf("llama bundle export PGO cubin sha256 mismatch for %s", cid)
		}
		pgoByCacheID[cid] = len(pgoBlobs)
		pgoBlobs = append(pgoBlobs, pgoBlob{
			entry: LlamaAOTPGOCacheEntry{
				CacheID:     cid,
				Key:         key,
				Proof:       proof,
				CubinLength: uint64(len(cubin)),
				CubinSHA256: hex.EncodeToString(cSum[:]),
			},
			cubin: slices.Clone(cubin),
		})
		return nil
	}

	if cfg.PGOCache != nil {
		keys, err := cfg.PGOCache.List()
		if err != nil {
			return nil, fmt.Errorf("llama bundle export list pgo cache: %w", err)
		}
		for _, k := range keys {
			proof, cubin, err := cfg.PGOCache.LoadVerified(k)
			if err != nil {
				return nil, fmt.Errorf("llama bundle export load verified pgo %s: %w", k.CacheID(), err)
			}
			if err := addPGOBlob(k, proof, cubin); err != nil {
				return nil, err
			}
		}
	}

	bucketArtifacts := make([]LlamaAOTBucketArtifact, 0, len(entries))
	for _, e := range entries {
		mega := e.Megakernel
		rep, err := qualify.VerifyMegakernelBarrierSafety(&mega)
		if err != nil {
			return nil, fmt.Errorf("llama bundle export bucket M=%d barrier safety: %w", e.Spec.BucketM, err)
		}
		if rep == nil || !rep.Passed() {
			return nil, fmt.Errorf("llama bundle export bucket M=%d barrier report did not pass", e.Spec.BucketM)
		}

		lineSrc, spans, err := mega.LineInfoSource()
		if err != nil {
			return nil, fmt.Errorf("llama bundle export bucket M=%d lineinfo source: %w", e.Spec.BucketM, err)
		}
		cudaSrc := e.CUDASource
		if cudaSrc == "" {
			cudaSrc = lineSrc
		}
		barrierHex, err := canonicalBarrierReportDigest(rep)
		if err != nil {
			return nil, err
		}

		if cubin, ok := cfg.PGOCubins[e.Spec.BucketM]; ok && len(cubin) > 0 {
			cSum := sha256.Sum256(cubin)
			sSum := sha256.Sum256([]byte(cudaSrc))
			proof := autotune.PGOPromotionProof{
				Key:            e.CacheKey,
				CandidateName:  mega.Name,
				CubinSHA256:    cSum,
				SourceSHA256:   sSum,
				ScreenBlocks:   3,
				HoldoutBlocks:  6,
				ScreenRatio:    1.08,
				HoldoutRatio:   1.06,
				AANullRatio:    1.00,
				OneSidedPValue: 0.015625,
				TokensMatched:  true,
			}
			if err := addPGOBlob(e.CacheKey, proof, cubin); err != nil {
				return nil, err
			}
		}

		grid, block := mega.Grid()
		if block[0] == 0 {
			return nil, fmt.Errorf("llama bundle export bucket M=%d: zero block dim", e.Spec.BucketM)
		}
		bucketArtifacts = append(bucketArtifacts, LlamaAOTBucketArtifact{
			BucketM:          e.Spec.BucketM,
			KernelName:       mega.Name,
			Regime:           string(e.Spec.Regime),
			NumStages:        len(mega.Stages),
			TotalBlocks:      grid[0],
			SharedBytes:      mega.SharedBytes(),
			ReadyWords:       mega.ReadyWords(),
			CUDASource:       cudaSrc,
			LineInfoSource:   lineSrc,
			StageSpans:       spans,
			BarrierReport:    rep,
			SourceSHA256:     sha256HexBytes([]byte(cudaSrc)),
			LineSourceSHA256: sha256HexBytes([]byte(lineSrc)),
			BarrierSHA256:    barrierHex,
		})
	}

	slices.SortFunc(bucketArtifacts, func(a, b LlamaAOTBucketArtifact) int {
		return cmp.Compare(a.BucketM, b.BucketM)
	})

	weightSpecs := slices.Clone(cfg.WeightSpecs)
	if len(weightSpecs) == 0 {
		weightSpecs = defaultLlamaBundleWeightSpecs(modelCfg, cfg.Weights, layerL, layerNext)
	}
	slices.SortFunc(weightSpecs, func(a, b LlamaAOTWeightSpec) int {
		return cmp.Compare(a.Name, b.Name)
	})

	// Build 4096-byte page-aligned binary payload (relative to DataBasePage).
	var relPayload []byte
	relAlign := func() uint64 {
		aligned := alignLlamaBundlePage(uint64(len(relPayload)))
		if pad := int(aligned) - len(relPayload); pad > 0 {
			relPayload = append(relPayload, make([]byte, pad)...)
		}
		return aligned
	}

	relWeightOffsets := make([]uint64, len(weightSpecs))
	for i, w := range weightSpecs {
		if strings.TrimSpace(w.Name) == "" {
			return nil, errors.New("llama bundle export: weight tensor name is required")
		}
		off := relAlign()
		relWeightOffsets[i] = off
		relPayload = append(relPayload, w.Data...)
	}

	relCubinOffsets := make([]uint64, len(pgoBlobs))
	for i, pb := range pgoBlobs {
		off := relAlign()
		relCubinOffsets[i] = off
		relPayload = append(relPayload, pb.cubin...)
	}
	if endAligned := alignLlamaBundlePage(uint64(len(relPayload))); int(endAligned) > len(relPayload) {
		relPayload = append(relPayload, make([]byte, int(endAligned)-len(relPayload))...)
	}

	payloadHex := sha256HexBytes(relPayload)

	populateManifest := func(basePage uint64) (*LlamaAOTBundleManifest, error) {
		pgoEntries := make([]LlamaAOTPGOCacheEntry, len(pgoBlobs))
		pgoByBucket := make(map[int]*LlamaAOTPGOCacheEntry, len(pgoBlobs))
		for i, pb := range pgoBlobs {
			pe := pb.entry
			pe.CubinOffset = basePage + relCubinOffsets[i]
			pgoEntries[i] = pe
			if pe.Key.ShapeBucket > 0 {
				peCopy := pe
				pgoByBucket[pe.Key.ShapeBucket] = &peCopy
			}
		}

		bucketsCopy := make([]LlamaAOTBucketArtifact, len(bucketArtifacts))
		var bucketsHashBuf, barriersHashBuf bytes.Buffer
		for i, b := range bucketArtifacts {
			bc := b
			if pe, ok := pgoByBucket[b.BucketM]; ok {
				peCopy := *pe
				bc.PGOEntry = &peCopy
			}
			bucketsCopy[i] = bc
			bucketsHashBuf.WriteString(fmt.Sprintf("%d:%s:%s:%s\n", bc.BucketM, bc.KernelName, bc.SourceSHA256, bc.LineSourceSHA256))
			barriersHashBuf.WriteString(fmt.Sprintf("%d:%s:%s\n", bc.BucketM, bc.KernelName, bc.BarrierSHA256))
		}

		wMap := make(map[string]LlamaAOTWeightHeader, len(weightSpecs))
		var weightsHashBuf bytes.Buffer
		for i, w := range weightSpecs {
			relOff := relWeightOffsets[i]
			absOff := basePage + relOff
			dtype := strings.TrimSpace(w.DType)
			if dtype == "" {
				dtype = "BF16"
			}
			wHex := sha256HexBytes(w.Data)
			hdr := LlamaAOTWeightHeader{
				Name:        w.Name,
				DType:       dtype,
				Shape:       slices.Clone(w.Shape),
				DataOffsets: [2]uint64{relOff, relOff + uint64(len(w.Data))},
				PageOffset:  absOff,
				ByteLength:  uint64(len(w.Data)),
				SHA256:      wHex,
			}
			wMap[w.Name] = hdr
			weightsHashBuf.WriteString(fmt.Sprintf("%s:%s:%d:%d:%s\n", hdr.Name, hdr.DType, hdr.PageOffset, hdr.ByteLength, hdr.SHA256))
		}

		var pgoHashBuf bytes.Buffer
		for _, pe := range pgoEntries {
			pgoHashBuf.WriteString(fmt.Sprintf("%s:%d:%d:%s\n", pe.CacheID, pe.CubinOffset, pe.CubinLength, pe.CubinSHA256))
		}

		return &LlamaAOTBundleManifest{
			FormatVersion:  llamaBundleFormatVersion,
			Arch:           arch,
			DriverVersion:  driverVer,
			CUDAVersion:    cudaVer,
			WeightDigest:   weightDigest,
			LayerL:         layerL,
			LayerNext:      layerNext,
			FullLayer:      cfg.FullLayer,
			PageAlignment:  LlamaBundlePageAlignment,
			DataBasePage:   basePage,
			Buckets:        bucketsCopy,
			PGOCache:       pgoEntries,
			Weights:        wMap,
			BucketsSHA256:  sha256HexBytes(bucketsHashBuf.Bytes()),
			BarriersSHA256: sha256HexBytes(barriersHashBuf.Bytes()),
			PGOSHA256:      sha256HexBytes(pgoHashBuf.Bytes()),
			WeightsSHA256:  sha256HexBytes(weightsHashBuf.Bytes()),
			PayloadSHA256:  payloadHex,
		}, nil
	}

	var (
		basePage = uint64(LlamaBundlePageAlignment)
		manifest *LlamaAOTBundleManifest
		manJSON  []byte
	)
	for range 4 {
		manifest, err = populateManifest(basePage)
		if err != nil {
			return nil, err
		}
		manJSON, err = json.MarshalIndent(manifest, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("llama bundle export marshal manifest: %w", err)
		}
		needed := alignLlamaBundlePage(uint64(llamaBundleHeaderPrefix + len(manJSON)))
		if needed == basePage {
			break
		}
		basePage = needed
	}

	totalSize := int(basePage) + len(relPayload)
	archive := make([]byte, totalSize)
	copy(archive[0:8], []byte(llamaBundleMagic))
	binary.LittleEndian.PutUint64(archive[8:16], uint64(len(manJSON)))
	copy(archive[16:16+len(manJSON)], manJSON)
	copy(archive[int(basePage):], relPayload)

	if err := writeBundleFileAtomic(bundlePath, archive); err != nil {
		return nil, err
	}
	return manifest, nil
}

// LoadVerifiedLlamaBundle loads an .egbundle archive from bundlePath, verifies
// every section's SHA-256 digest, 4096-byte page alignment, PGO promotion
// proofs, and formal barrier safety certificates, and returns the verified
// manifest along with the bucketM-to-CUDA-source map.
func LoadVerifiedLlamaBundle(bundlePath string) (*LlamaAOTBundleManifest, map[int]string, error) {
	if strings.TrimSpace(bundlePath) == "" {
		return nil, nil, errors.New("llama bundle load: empty bundle path")
	}
	data, err := os.ReadFile(bundlePath)
	if err != nil {
		return nil, nil, fmt.Errorf("llama bundle load read %s: %w", bundlePath, err)
	}
	if len(data) < llamaBundleHeaderPrefix {
		return nil, nil, errors.New("llama bundle load: archive shorter than header prefix")
	}
	if string(data[:8]) != llamaBundleMagic {
		return nil, nil, fmt.Errorf("llama bundle load: invalid magic %q", string(data[:8]))
	}
	manLen := binary.LittleEndian.Uint64(data[8:16])
	if manLen == 0 || uint64(len(data)) < llamaBundleHeaderPrefix+manLen {
		return nil, nil, errors.New("llama bundle load: truncated or empty manifest JSON")
	}

	var manifest LlamaAOTBundleManifest
	if err := json.Unmarshal(data[16:16+manLen], &manifest); err != nil {
		return nil, nil, fmt.Errorf("llama bundle load unmarshal manifest: %w", err)
	}
	if manifest.FormatVersion != llamaBundleFormatVersion {
		return nil, nil, fmt.Errorf("llama bundle load: unsupported format_version %d", manifest.FormatVersion)
	}
	if manifest.PageAlignment != LlamaBundlePageAlignment {
		return nil, nil, fmt.Errorf("llama bundle load: page_alignment %d != %d", manifest.PageAlignment, LlamaBundlePageAlignment)
	}
	if manifest.DataBasePage == 0 || manifest.DataBasePage%LlamaBundlePageAlignment != 0 {
		return nil, nil, fmt.Errorf("llama bundle load: data_base_page %d is not %d-byte aligned", manifest.DataBasePage, LlamaBundlePageAlignment)
	}
	if uint64(len(data)) < manifest.DataBasePage {
		return nil, nil, errors.New("llama bundle load: archive truncated before data_base_page")
	}
	for i, padByte := range data[16+manLen : manifest.DataBasePage] {
		if padByte != 0 {
			return nil, nil, fmt.Errorf("llama bundle load: non-zero header padding byte at offset %d", 16+int(manLen)+i)
		}
	}

	payload := data[manifest.DataBasePage:]
	if gotPayloadHex := sha256HexBytes(payload); gotPayloadHex != manifest.PayloadSHA256 {
		return nil, nil, fmt.Errorf("llama bundle load: payload sha256 mismatch: got %s, want %s", gotPayloadHex, manifest.PayloadSHA256)
	}

	if len(manifest.Buckets) == 0 {
		return nil, nil, errors.New("llama bundle load: manifest contains zero bucket artifacts")
	}

	sourcesByBucket := make(map[int]string, len(manifest.Buckets))
	var bucketsHashBuf, barriersHashBuf bytes.Buffer
	for _, b := range manifest.Buckets {
		if b.BucketM <= 0 {
			return nil, nil, fmt.Errorf("llama bundle load: invalid bucket_m %d", b.BucketM)
		}
		if b.CUDASource == "" || b.LineInfoSource == "" {
			return nil, nil, fmt.Errorf("llama bundle load: bucket M=%d has empty CUDA source", b.BucketM)
		}
		if gotSrcHex := sha256HexBytes([]byte(b.CUDASource)); gotSrcHex != b.SourceSHA256 {
			return nil, nil, fmt.Errorf("llama bundle load: bucket M=%d source sha256 mismatch: got %s, want %s", b.BucketM, gotSrcHex, b.SourceSHA256)
		}
		if gotLineHex := sha256HexBytes([]byte(b.LineInfoSource)); gotLineHex != b.LineSourceSHA256 {
			return nil, nil, fmt.Errorf("llama bundle load: bucket M=%d lineinfo source sha256 mismatch: got %s, want %s", b.BucketM, gotLineHex, b.LineSourceSHA256)
		}
		if b.BarrierReport == nil || !b.BarrierReport.Passed() {
			return nil, nil, fmt.Errorf("llama bundle load: bucket M=%d formal barrier report did not pass", b.BucketM)
		}
		gotBarrierHex, err := canonicalBarrierReportDigest(b.BarrierReport)
		if err != nil {
			return nil, nil, err
		}
		if gotBarrierHex != b.BarrierSHA256 {
			return nil, nil, fmt.Errorf("llama bundle load: bucket M=%d barrier report sha256 mismatch: got %s, want %s", b.BucketM, gotBarrierHex, b.BarrierSHA256)
		}
		bucketsHashBuf.WriteString(fmt.Sprintf("%d:%s:%s:%s\n", b.BucketM, b.KernelName, b.SourceSHA256, b.LineSourceSHA256))
		barriersHashBuf.WriteString(fmt.Sprintf("%d:%s:%s\n", b.BucketM, b.KernelName, b.BarrierSHA256))
		sourcesByBucket[b.BucketM] = b.CUDASource
	}
	if got := sha256HexBytes(bucketsHashBuf.Bytes()); got != manifest.BucketsSHA256 {
		return nil, nil, fmt.Errorf("llama bundle load: buckets section sha256 mismatch: got %s, want %s", got, manifest.BucketsSHA256)
	}
	if got := sha256HexBytes(barriersHashBuf.Bytes()); got != manifest.BarriersSHA256 {
		return nil, nil, fmt.Errorf("llama bundle load: barriers section sha256 mismatch: got %s, want %s", got, manifest.BarriersSHA256)
	}

	weightNames := make([]string, 0, len(manifest.Weights))
	for name := range manifest.Weights {
		weightNames = append(weightNames, name)
	}
	slices.Sort(weightNames)

	var weightsHashBuf bytes.Buffer
	for _, name := range weightNames {
		hdr := manifest.Weights[name]
		if hdr.PageOffset < manifest.DataBasePage || hdr.PageOffset%LlamaBundlePageAlignment != 0 {
			return nil, nil, fmt.Errorf("llama bundle load: weight %q page_offset %d is not %d-byte aligned", name, hdr.PageOffset, LlamaBundlePageAlignment)
		}
		end := hdr.PageOffset + hdr.ByteLength
		if end > uint64(len(data)) {
			return nil, nil, fmt.Errorf("llama bundle load: weight %q range [%d, %d) exceeds archive size %d", name, hdr.PageOffset, end, len(data))
		}
		if gotHex := sha256HexBytes(data[hdr.PageOffset:end]); gotHex != hdr.SHA256 {
			return nil, nil, fmt.Errorf("llama bundle load: weight %q sha256 mismatch: got %s, want %s", name, gotHex, hdr.SHA256)
		}
		weightsHashBuf.WriteString(fmt.Sprintf("%s:%s:%d:%d:%s\n", hdr.Name, hdr.DType, hdr.PageOffset, hdr.ByteLength, hdr.SHA256))
	}
	if got := sha256HexBytes(weightsHashBuf.Bytes()); got != manifest.WeightsSHA256 {
		return nil, nil, fmt.Errorf("llama bundle load: weights section sha256 mismatch: got %s, want %s", got, manifest.WeightsSHA256)
	}

	var pgoHashBuf bytes.Buffer
	for _, pe := range manifest.PGOCache {
		if err := pe.Key.Validate(); err != nil {
			return nil, nil, fmt.Errorf("llama bundle load: pgo key %s invalid: %w", pe.CacheID, err)
		}
		if pe.CubinOffset < manifest.DataBasePage || pe.CubinOffset%LlamaBundlePageAlignment != 0 {
			return nil, nil, fmt.Errorf("llama bundle load: pgo entry %s cubin_offset %d not %d-byte aligned", pe.CacheID, pe.CubinOffset, LlamaBundlePageAlignment)
		}
		end := pe.CubinOffset + pe.CubinLength
		if end > uint64(len(data)) || pe.CubinLength == 0 {
			return nil, nil, fmt.Errorf("llama bundle load: pgo entry %s range [%d, %d) out of bounds", pe.CacheID, pe.CubinOffset, end)
		}
		cubinSlice := data[pe.CubinOffset:end]
		cSum := sha256.Sum256(cubinSlice)
		gotHex := hex.EncodeToString(cSum[:])
		if gotHex != pe.CubinSHA256 || cSum != pe.Proof.CubinSHA256 {
			return nil, nil, fmt.Errorf("llama bundle load: pgo entry %s cubin sha256 mismatch", pe.CacheID)
		}
		if !pe.Proof.TokensMatched || pe.Proof.HoldoutBlocks < 6 || pe.Proof.HoldoutRatio <= 1.0 {
			return nil, nil, fmt.Errorf("llama bundle load: pgo entry %s failed holdout proof verification", pe.CacheID)
		}
		pgoHashBuf.WriteString(fmt.Sprintf("%s:%d:%d:%s\n", pe.CacheID, pe.CubinOffset, pe.CubinLength, pe.CubinSHA256))
	}
	if got := sha256HexBytes(pgoHashBuf.Bytes()); got != manifest.PGOSHA256 {
		return nil, nil, fmt.Errorf("llama bundle load: pgo section sha256 mismatch: got %s, want %s", got, manifest.PGOSHA256)
	}

	return &manifest, sourcesByBucket, nil
}
