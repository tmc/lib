//go:build linux && tpu

package main

import (
	"context"
	"errors"
	"os"
	"strings"
	"fmt"
	"time"
    "unsafe"

	"github.com/tmc/lib/apple/x/metal/metalprof"
	"github.com/tmc/lib/openxla"
	"github.com/tmc/lib/tpu"
)

var (
	ErrLibTPUNotFound          = errors.New("tpu: libtpu library not found")
	ErrNoAddressableTPUDevices = errors.New("openxla: no addressable tpu devices found")
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
		if errors.Is(err, os.ErrNotExist) || strings.Contains(err.Error(), "no such file") || strings.Contains(err.Error(), "failed to load libtpu.so") {
			return nil, fmt.Errorf("%w: %v", ErrLibTPUNotFound, err)
		}
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

    devs, err := client.AddressableDevices()
    if err != nil {
        _ = client.Destroy()
        return nil, fmt.Errorf("query addressable devices: %w", err)
    }
    if len(devs) == 0 {
        _ = client.Destroy()
        return nil, ErrNoAddressableTPUDevices
    }
    
    // StableHLO compilation of a basic forward matmul: (2x4) x (4x3) -> (2x3)
    mlir := `module @smoke_matmul {
  func.func @main(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}`
    exec, err := client.Compile([]byte(mlir), "mlir", nil)
    if err != nil {
        _ = client.Destroy()
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
    if eng == nil || eng.client == nil || eng.exec == nil {
        return errors.New("tpu engine not initialized")
    }
    devs, err := eng.client.AddressableDevices()
    if err != nil {
        return fmt.Errorf("query addressable devices: %w", err)
    }
    if len(devs) == 0 {
        return ErrNoAddressableTPUDevices
    }
    dev := devs[0]

    // Asymmetric nonzero inputs: A (2x4) and B (4x3)
    // A:
    // [1.0, 2.0, 3.0, 4.0]
    // [5.0, 6.0, 7.0, 8.0]
    in1 := []float32{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    }
    // B:
    // [1.0, 0.5, 2.0]
    // [0.0, 1.5, 1.0]
    // [2.0, 0.0, 1.0]
    // [1.0, 2.0, 0.5]
    in2 := []float32{
        1.0, 0.5, 2.0,
        0.0, 1.5, 1.0,
        2.0, 0.0, 1.0,
        1.0, 2.0, 0.5,
    }

    // Expected C = A x B (2x3):
    // C[0,0] = 1*1 + 2*0 + 3*2 + 4*1 = 1 + 0 + 6 + 4 = 11.0
    // C[0,1] = 1*0.5 + 2*1.5 + 3*0 + 4*2 = 0.5 + 3.0 + 0 + 8.0 = 11.5
    // C[0,2] = 1*2 + 2*1 + 3*1 + 4*0.5 = 2 + 2 + 3 + 2 = 9.0
    // C[1,0] = 5*1 + 6*0 + 7*2 + 8*1 = 5 + 0 + 14 + 8 = 27.0
    // C[1,1] = 5*0.5 + 6*1.5 + 7*0 + 8*2 = 2.5 + 9.0 + 0 + 16.0 = 27.5
    // C[1,2] = 5*2 + 6*1 + 7*1 + 8*0.5 = 10 + 6 + 7 + 4 = 27.0
    expectedC := []float32{
        11.0, 11.5, 9.0,
        27.0, 27.5, 27.0,
    }

    b1, err := eng.client.BufferFromHost(unsafe.Pointer(&in1[0]), openxla.Buffer_Type_F32, []int64{2, 4}, dev)
    if err != nil {
        return fmt.Errorf("bufferFromHost in1: %w", err)
    }
    defer b1.Destroy()

    b2, err := eng.client.BufferFromHost(unsafe.Pointer(&in2[0]), openxla.Buffer_Type_F32, []int64{4, 3}, dev)
    if err != nil {
        return fmt.Errorf("bufferFromHost in2: %w", err)
    }
    defer b2.Destroy()

    // Execute
    outputs, err := eng.exec.Execute([][]*openxla.BufferWrapper{{b1, b2}})
    if err != nil {
        return fmt.Errorf("execute: %w", err)
    }
    if len(outputs) != 1 || len(outputs[0]) != 1 {
        return fmt.Errorf("unexpected output cardinality: got %d devices, %d outputs on dev 0", len(outputs), len(outputs[0]))
    }

    outBuffer := outputs[0][0]
    defer outBuffer.Destroy()

    outHost := make([]float32, 6)
    if err := outBuffer.ToHost(unsafe.Pointer(&outHost[0]), uintptr(len(outHost)*4)); err != nil {
        return fmt.Errorf("toHost: %w", err)
    }

    for i := range expectedC {
        diff := outHost[i] - expectedC[i]
        if diff < -1e-4 || diff > 1e-4 {
            return fmt.Errorf("matmul numerical mismatch at index %d: got %f, want %f", i, outHost[i], expectedC[i])
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
func (eng *ModelEngine) Free() error {
	if eng == nil {
		return nil
	}
	var errs []error
	if eng.exec != nil {
		if err := eng.exec.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("destroy executable: %w", err))
		}
		eng.exec = nil
	}
	if eng.client != nil {
		if err := eng.client.Destroy(); err != nil {
			errs = append(errs, fmt.Errorf("destroy client: %w", err))
		}
		eng.client = nil
	}
	return errors.Join(errs...)
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
