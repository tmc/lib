//go:build tpu

package main

import (
	"context"
	"errors"
	"fmt"
	"time"
    "unsafe"

	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/openxla"
	"github.com/tmc/lib/tpu"
)

type BatchedQKVResult struct {
	Positions                      []int
	Embedding, Normalized, Q, K, V []float32
	Dispatches                     int
}

func RunBatchedQKV(*ModelEngine, Config, int, []int32, []int) (BatchedQKVResult, error) {
	return BatchedQKVResult{}, errors.New("unsupported")
}

type BatchedAttentionCache struct{}

func (*BatchedAttentionCache) ImportSequence(int, int, *SequenceKVCache) error {
	return errors.New("unsupported")
}

type SamplerConfig struct {
	Method      string
	Temperature float32
}

type BatchedModelTail struct{}

type BatchedModelTailResult struct {
	Normalized []float32
	Logits     []float32
	Tokens     []int32
	Dispatches int
}

func (*ModelEngine) NewBatchedModelTail(Config, int) (*BatchedModelTail, error) {
	return nil, errors.New("unsupported")
}

func (*BatchedModelTail) Free() {}

func (*BatchedModelTail) Run(*ModelEngine, Config, []float32, []SamplerConfig) (BatchedModelTailResult, error) {
	return BatchedModelTailResult{}, errors.New("unsupported")
}

func RunBatchedModelTailFromChain(*ModelEngine, Config, *BatchedModelTail, BatchedLayerChainResult, []SamplerConfig) (BatchedModelTailResult, error) {
	return BatchedModelTailResult{}, errors.New("unsupported")
}

type BatchedAttentionResult struct {
	Q, K, Attention []float32
	Positions, Rows []int
	Dispatches      int
}
type BatchedLayerResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
}
type BatchedLayerChainResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
	DeviceBytes     int64
}

type SegmentedLayerChainResult struct {
	Hidden          []float32
	Positions, Rows []int
	Dispatches      int
	DeviceBytes     int64
}

func RunBatchedLayerChain(*ModelEngine, Config, *BatchedAttentionCache, []int32, []int, []int) (BatchedLayerChainResult, error) {
	return BatchedLayerChainResult{}, errors.New("unsupported")
}

func RunSegmentedLayerChain(context.Context, *ModelEngine, Config, *BatchedAttentionCache, PackedPrefill) (SegmentedLayerChainResult, error) {
	return SegmentedLayerChainResult{}, errors.New("unsupported")
}

func RunSegmentedModelTailFromChain(*ModelEngine, Config, *BatchedModelTail, SegmentedLayerChainResult, []SamplerConfig) (BatchedModelTailResult, error) {
	return BatchedModelTailResult{}, errors.New("unsupported")
}

func RunBatchedLayer(*ModelEngine, Config, int, *BatchedAttentionCache, []int32, []int, []int) (BatchedLayerResult, error) {
	return BatchedLayerResult{}, errors.New("unsupported")
}

func (*ModelEngine) NewBatchedAttentionCache(Config, int) (*BatchedAttentionCache, error) {
	return nil, errors.New("unsupported")
}
func (*BatchedAttentionCache) Free() {}
func RunBatchedAttention(*ModelEngine, Config, int, *BatchedAttentionCache, []float32, []float32, []float32, []int, []int) (BatchedAttentionResult, error) {
	return BatchedAttentionResult{}, errors.New("unsupported")
}

func RunSegmentedAttention(context.Context, *ModelEngine, Config, int, *BatchedAttentionCache, PackedPrefill, []float32, []float32, []float32) (BatchedAttentionResult, error) {
	return BatchedAttentionResult{}, errors.New("unsupported")
}

type dummyStream struct{}
func (s *dummyStream) Synchronize(ctx context.Context) error { return nil }

type ModelEngine struct {
	client *openxla.ClientWrapper
	Plan   *ExecutionPlan
    mCfg   ModelConfig
    tokenizer Tokenizer
    metrics *RuntimeMetrics
    stream *dummyStream
    benchmarkSerial bool
    egraphKnown bool
    egraphSats int
    recorder *metalprof.FlightRecorder
    exec *openxla.ExecutableWrapper
}

func (eng *ModelEngine) CheckpointReceipt() CheckpointReceipt {
	return CheckpointReceipt{}
}

func (eng *ModelEngine) FlightRecorder() *metalprof.FlightRecorder {
	return nil
}

func (eng *ModelEngine) IsF16() bool {
	return false
}

func (eng *ModelEngine) WeightReport() WeightReport {
	return WeightReport{}
}

func (eng *ModelEngine) ReadLogits() ([]float32, error) {
	return nil, errors.New("unsupported")
}

func (eng *ModelEngine) ResetKV() error {
	return nil
}

func (eng *ModelEngine) MutateWeightForTest(layerIndex, elemIndex int, delta float32) (func(), error) {
	return nil, errors.New("unsupported")
}

func GetOrCompileEngine(cfg Config) (*ModelEngine, error) {
	return GetOrCompileTPUEngine(cfg)
}

func GetOrCompileCudaEngine(cfg Config) (*ModelEngine, error) {
    return nil, errors.New("cuda unsupported in tpu build")
}

func GetOrCompileTPUEngine(cfg Config) (*ModelEngine, error) {
	if err := tpu.Load(""); err != nil {
		return nil, fmt.Errorf("load libtpu: %w", err)
	}
	tpuApi, err := tpu.GetApi()
	if err != nil {
		return nil, fmt.Errorf("tpu.GetApi: %w", err)
	}
	
	client, err := openxla.NewClientFromApi((*openxla.Api)(unsafe.Pointer(tpuApi)))
	if err != nil {
		return nil, fmt.Errorf("NewClientFromApi: %w", err)
	}

    // Attempt to upload weights if we had them.
    // Assuming deterministic mock weights for now to demonstrate OpenXLA/TPU buffers.
    devs, err := client.AddressableDevices()
    if err != nil || len(devs) == 0 {
        return nil, fmt.Errorf("no addressable tpu devices")
    }
    
    // StableHLO compilation of a basic forward pass
    mlir := `module @llama {
  func.func @forward(%arg0: tensor<1x2048x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<1x2048x4096xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [2] x [0] : (tensor<1x2048x4096xf32>, tensor<4096x4096xf32>) -> tensor<1x2048x4096xf32>
    return %0 : tensor<1x2048x4096xf32>
  }
}`
    exec, err := client.Compile([]byte(mlir), "mlir", nil)
    if err != nil {
        return nil, fmt.Errorf("compile stablehlo: %w", err)
    }

	return &ModelEngine{
		client: client,
		Plan:   BuildExecutionPlan(cfg),
        tokenizer: NewByteTokenizer(),
        metrics: NewRuntimeMetrics(256),
        stream: &dummyStream{},
        exec: exec,
	}, nil
}

func (eng *ModelEngine) ForwardPassMockTPU() error {
    devs, err := eng.client.AddressableDevices()
    if err != nil || len(devs) == 0 {
        return fmt.Errorf("no addressable tpu devices")
    }
    dev := devs[0]

    // Allocate buffers
    in1 := make([]float32, 1 * 2048 * 4096)
    in2 := make([]float32, 4096 * 4096)
    
    b1, err := eng.client.BufferFromHost(unsafe.Pointer(&in1[0]), openxla.Buffer_Type_F32, []int64{1, 2048, 4096}, dev)
    if err != nil {
        return err
    }
    defer b1.Destroy()

    b2, err := eng.client.BufferFromHost(unsafe.Pointer(&in2[0]), openxla.Buffer_Type_F32, []int64{4096, 4096}, dev)
    if err != nil {
        return err
    }
    defer b2.Destroy()

    // Execute
    outputs, err := eng.exec.Execute([][]*openxla.BufferWrapper{{b1, b2}})
    if err != nil {
        return err
    }
    
    // Read back
    if len(outputs) > 0 && len(outputs[0]) > 0 {
        outBuffer := outputs[0][0]
        defer outBuffer.Destroy()
        outHost := make([]float32, 1 * 2048 * 4096)
        if err := outBuffer.ToHost(unsafe.Pointer(&outHost[0]), uintptr(len(outHost)*4)); err != nil {
            return err
        }
    }
    
    return nil
}


type SequenceKVCache struct{ Cache interface{} }

func (eng *ModelEngine) NewSequenceKV(cfg Config) (*SequenceKVCache, error) {
	return &SequenceKVCache{}, nil
}

func (skv *SequenceKVCache) Free() {}

func runPlatform(cfg Config) error {
	return RunTPUSmokeTest(cfg)
}

func GPUPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32) (int32, error) {
	return 0, errors.New("unsupported")
}

func GPUForwardPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, tokenID int32, pos int) (int32, error) {
	return 0, errors.New("unsupported")
}

func GPUPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) (int32, error) {
	return 0, errors.New("unsupported")
}

func GPUBatchedPrefillPass(eng *ModelEngine, cfg Config, promptTokens []int32) (int32, error) {
	return 0, errors.New("unsupported")
}

func GPUBatchedPrefillPassKV(eng *ModelEngine, cfg Config, skv *SequenceKVCache, promptTokens []int32, startPos int) (int32, error) {
	return 0, errors.New("unsupported")
}

func GPUForwardPass(eng *ModelEngine, cfg Config, tokenID int32, pos int) (int32, error) {
	return 0, errors.New("unsupported")
}

func GPUBatchDecodeStep(eng *ModelEngine, cfg Config, batchKVs []*SequenceKVCache, batchTokens []int32, batchPositions []int) ([]int32, error) {
	return nil, errors.New("unsupported")
}

func PrefillStep(cfg Config, promptTokens []int32) (time.Duration, error) {
	return 0, errors.New("unsupported")
}

func DecodeStep(cfg Config, tokenID int32, pos int) (time.Duration, error) {
	return 0, errors.New("unsupported")
}

// Free releases on-device executables and client resources.
func (eng *ModelEngine) Free() {
	if eng == nil {
		return
	}
	if eng.exec != nil {
		_ = eng.exec.Destroy()
		eng.exec = nil
	}
	if eng.client != nil {
		_ = eng.client.Destroy()
		eng.client = nil
	}
}

// RunTPUSmokeTest executes a standalone matmul smoke test on the TPU device.
func RunTPUSmokeTest(cfg Config) error {
	eng, err := GetOrCompileTPUEngine(cfg)
	if err != nil {
		return fmt.Errorf("tpu initialization: %w", err)
	}
	defer eng.Free()

	fmt.Println("=== TPU Backend initialized ===")
	fmt.Println("Running TPU forward pass test...")
	if err := eng.ForwardPassMockTPU(); err != nil {
		return fmt.Errorf("TPU forward pass execution failed: %w", err)
	}
	fmt.Println("TPU forward pass executed successfully!")
	return nil
}

// Generate reports unsupported until full LLaMA inference is implemented on TPU.
// For matmul pipeline verification, use RunTPUSmokeTest.
func (eng *ModelEngine) Generate(ctx context.Context, cfg Config, prompt string, maxTokens int, onToken func(tokenID int32, tokenText string) bool) error {
	return errors.New("unsupported: TPU LLaMA token generation is not yet implemented; use RunTPUSmokeTest for matmul verification")
}

func (eng *ModelEngine) GenerateTokenIDs(ctx context.Context, cfg Config, prompt string, maxTokens int) ([]int32, error) {
	return nil, errors.New("unsupported")
}

func RunBenchSuite(cfg Config, count int) error {
	return errors.New("unsupported")
}

func defaultNodeKeyPath() string { return "" }
func defaultPeerStatePath() string { return "" }
func runInferbench(ctx context.Context, cfg Config, spec string) error { return errors.New("unsupported") }


// UploadWeightsToTPU uploads CPU weights to Cloud TPU memory via PJRT.
func UploadWeightsToTPU(dev *openxla.Device, client *openxla.ClientWrapper, w *ModelWeights, cfg *ModelConfig) ([]*openxla.BufferWrapper, error) {
    if w == nil || len(w.EmbedTokens) == 0 {
        return nil, nil // No weights provided
    }
    // Allocate the embedding table for demonstration
    embBuf, err := client.BufferFromHost(unsafe.Pointer(&w.EmbedTokens[0]), openxla.Buffer_Type_F32, []int64{int64(cfg.VocabSize), int64(cfg.HiddenSize)}, dev)
    if err != nil {
        return nil, fmt.Errorf("failed to upload embed_tokens: %w", err)
    }
    return []*openxla.BufferWrapper{embBuf}, nil
}
