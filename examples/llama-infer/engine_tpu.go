//go:build linux && tpu

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
	"github.com/tmc/lib/x/xla"
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

// ModelEngine owns its model and client. Calls must be externally synchronized.
type ModelEngine struct {
	pjrt            *xla.Engine
	decoder         *pjrtModel
	checkpoint      CheckpointReceipt
	checkpointEOS   []int32
	client          *openxla.ClientWrapper
	Plan            *ExecutionPlan
	mCfg            ModelConfig
	tokenizer       Tokenizer
	metrics         *RuntimeMetrics
	stream          *dummyStream
	benchmarkSerial bool
	egraphKnown     bool
	egraphSats      int
	recorder        *metalprof.FlightRecorder
	exec            *openxla.ExecutableWrapper
}

func (eng *ModelEngine) CheckpointReceipt() CheckpointReceipt {
	if eng == nil {
		return CheckpointReceipt{}
	}
	return eng.checkpoint
}

func (eng *ModelEngine) FlightRecorder() *metalprof.FlightRecorder {
	return nil
}

func (eng *ModelEngine) IsF16() bool {
	return false
}

func (eng *ModelEngine) WeightReport() WeightReport {
	if eng == nil || eng.decoder == nil || eng.decoder.model == nil {
		return WeightReport{}
	}
	return WeightReport{DType: "F32", Storage: "F32", PerElem: 4, Bytes: eng.decoder.weightBytes}
}

func (eng *ModelEngine) ReadLogits() ([]float32, error) {
	if eng == nil {
		return nil, errors.New("tpu: engine is closed")
	}
	return eng.decoder.readLogits()
}

func (eng *ModelEngine) ResetKV() error {
	if eng == nil {
		return errors.New("tpu: engine is closed")
	}
	return eng.decoder.reset()
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
	p, tokenizer, err := preparePJRTModel(cfg)
	if err != nil {
		return nil, err
	}
	if err := tpu.Load(""); err != nil {
		return nil, fmt.Errorf("load libtpu: %w", classifyTPULoadError(err))
	}
	api, err := tpu.GetApi()
	if err != nil {
		return nil, err
	}
	return newTPUModelEngine(cfg, p, tokenizer, api)
}

func newTPUModelEngine(cfg Config, p *pjrtCheckpoint, tokenizer Tokenizer, api *openxla.Api) (*ModelEngine, error) {
	engine, err := xla.NewEngineFromAPI(api)
	if err != nil {
		return nil, err
	}
	devices, err := engine.Devices()
	if err != nil {
		return nil, errors.Join(err, engine.Close())
	}
	if len(devices) == 0 {
		return nil, errors.Join(ErrNoAddressableTPUDevices, engine.Close())
	}
	decoder, err := newPJRTModel(context.Background(), engine, &p.config, p.weights, p.theta)
	if err != nil {
		return nil, errors.Join(err, engine.Close())
	}
	return &ModelEngine{pjrt: engine, decoder: decoder, mCfg: p.config,
		checkpoint: p.weights.Consumption, checkpointEOS: p.eos, tokenizer: tokenizer,
		Plan: BuildExecutionPlan(cfg), metrics: NewRuntimeMetrics(256), stream: &dummyStream{}}, nil
}

func newTPUSmokeEngine(cfg Config) (*ModelEngine, error) {
	if err := tpu.Load(""); err != nil {
		return nil, fmt.Errorf("load libtpu: %w", classifyTPULoadError(err))
	}
	tpuApi, err := tpu.GetApi()
	if err != nil {
		return nil, fmt.Errorf("tpu.GetApi: %w", err)
	}

	client, err := openxla.NewClientFromApi(tpuApi)
	if err != nil {
		return nil, fmt.Errorf("NewClientFromApi: %w", err)
	}

	devs, err := client.AddressableDevices()
	if err != nil {
		dErr := client.Destroy()
		return nil, errors.Join(fmt.Errorf("query addressable devices: %w", err), dErr)
	}
	if len(devs) == 0 {
		dErr := client.Destroy()
		return nil, errors.Join(ErrNoAddressableTPUDevices, dErr)
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
		dErr := client.Destroy()
		return nil, errors.Join(fmt.Errorf("compile stablehlo: %w", err), dErr)
	}

	return &ModelEngine{
		client:    client,
		Plan:      BuildExecutionPlan(cfg),
		tokenizer: NewByteTokenizer(),
		metrics:   NewRuntimeMetrics(256),
		stream:    &dummyStream{},
		exec:      exec,
	}, nil
}

func (eng *ModelEngine) ForwardPassMockTPU() (err error) {
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
	defer func() {
		if dErr := b1.Destroy(); dErr != nil {
			err = errors.Join(err, fmt.Errorf("destroy b1: %w", dErr))
		}
	}()

	b2, b2Err := eng.client.BufferFromHost(unsafe.Pointer(&in2[0]), openxla.Buffer_Type_F32, []int64{4, 3}, dev)
	if b2Err != nil {
		err = fmt.Errorf("bufferFromHost in2: %w", b2Err)
		return err
	}
	defer func() {
		if dErr := b2.Destroy(); dErr != nil {
			err = errors.Join(err, fmt.Errorf("destroy b2: %w", dErr))
		}
	}()

	outputs, execErr := eng.exec.Execute([][]*openxla.BufferWrapper{{b1, b2}})
	if execErr != nil {
		err = fmt.Errorf("execute: %w", execErr)
		return err
	}
	// Arrange cleanup for ALL returned buffers upfront before cardinality check to prevent leaks
	defer func() {
		for _, devOutputs := range outputs {
			for _, b := range devOutputs {
				if b != nil {
					if dErr := b.Destroy(); dErr != nil {
						err = errors.Join(err, fmt.Errorf("destroy output buffer: %w", dErr))
					}
				}
			}
		}
	}()

	if cardErr := validateOutputCardinality(outputs); cardErr != nil {
		err = cardErr
		return err
	}

	outBuffer := outputs[0][0]
	outHost := make([]float32, 6)
	if toHostErr := outBuffer.ToHost(unsafe.Pointer(&outHost[0]), uintptr(len(outHost)*4)); toHostErr != nil {
		err = fmt.Errorf("toHost: %w", toHostErr)
		return err
	}

	err = checkMatmulOutput(outHost, expectedC)
	return err
}

type SequenceKVCache struct{ Cache interface{} }

func (eng *ModelEngine) NewSequenceKV(cfg Config) (*SequenceKVCache, error) {
	return &SequenceKVCache{}, nil
}

func (skv *SequenceKVCache) Free() {}

func runPlatform(cfg Config) (err error) {
	if cfg.ServeMode || cfg.HTTPAddr != "" || cfg.PeerAddr != "" {
		return errors.New("tpu: serving and mesh modes are unsupported")
	}
	if cfg.ModelPath != "" {
		return errors.New("tpu: checkpoint text tokenization is unsupported; use GenerateFromTokenIDs")
	}
	eng, err := GetOrCompileTPUEngine(cfg)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, eng.Free()) }()
	n := cfg.GenTokens
	if n <= 0 {
		n = 10
	}
	err = eng.Generate(context.Background(), cfg, cfg.Prompt, n, func(_ int32, text string) bool { fmt.Print(text); return true })
	fmt.Println()
	return err
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
	errs = append(errs, eng.decoder.close(), eng.pjrt.Close())
	eng.decoder, eng.pjrt = nil, nil
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
func RunTPUSmokeTest(cfg Config) (err error) {
	eng, initErr := newTPUSmokeEngine(cfg)
	if initErr != nil {
		return fmt.Errorf("tpu initialization: %w", initErr)
	}
	defer func() {
		if freeErr := eng.Free(); freeErr != nil {
			err = errors.Join(err, fmt.Errorf("engine cleanup: %w", freeErr))
		}
	}()

	fmt.Println("=== TPU Backend initialized ===")
	fmt.Println("Running TPU forward pass test...")
	if fwdErr := eng.ForwardPassMockTPU(); fwdErr != nil {
		return fmt.Errorf("TPU forward pass execution failed: %w", fwdErr)
	}
	fmt.Println("TPU forward pass executed successfully!")
	return nil
}

// Generate emits greedy tokens for synthetic byte-tokenizer models. Checkpoint
// text generation requires a qualified tokenizer; use GenerateFromTokenIDs
// with externally tokenized input until that tokenizer is available.
func (eng *ModelEngine) Generate(ctx context.Context, cfg Config, prompt string, maxTokens int, onToken func(int32, string) bool) error {
	if eng == nil || eng.decoder == nil {
		return errors.New("tpu: engine is closed")
	}
	if eng.tokenizer == nil {
		return errors.New("tpu: checkpoint text tokenization is unsupported; use GenerateFromTokenIDs")
	}
	if onToken == nil {
		return errors.New("tpu: token callback is required")
	}
	ids, err := eng.tokenizer.Encode(prompt)
	if err != nil {
		return err
	}
	if cfg.BOSTokenID != 0 {
		ids = append([]int32{cfg.BOSTokenID}, ids...)
	}
	eos := cfg.EOSTokenIDs
	if len(eos) == 0 && cfg.EOSTokenID != 0 {
		eos = []int32{cfg.EOSTokenID}
	}
	var decodeErr error
	err = eng.decoder.generate(ctx, ids, maxTokens, eos, func(id int32) bool {
		text, err := eng.tokenizer.Decode([]int32{id})
		if err != nil {
			decodeErr = err
			return false
		}
		return onToken(id, text)
	})
	return errors.Join(err, decodeErr)
}

// GenerateFromTokenIDs starts a fresh greedy request with exact input IDs.
// Callers include any BOS or chat-template tokens in prompt. Checkpoint EOS
// IDs terminate generation. Zero maxTokens performs no inference.
func (eng *ModelEngine) GenerateFromTokenIDs(ctx context.Context, prompt []int32, maxTokens int, onToken func(int32) bool) error {
	if eng == nil {
		return errors.New("tpu: engine is closed")
	}
	return eng.decoder.generate(ctx, prompt, maxTokens, eng.checkpointEOS, onToken)
}

func (eng *ModelEngine) GenerateTokenIDs(ctx context.Context, cfg Config, prompt string, maxTokens int) ([]int32, error) {
	var ids []int32
	err := eng.Generate(ctx, cfg, prompt, maxTokens, func(id int32, _ string) bool { ids = append(ids, id); return true })
	return ids, err
}

func RunBenchSuite(cfg Config, count int) error {
	return errors.New("unsupported")
}

func defaultNodeKeyPath() string   { return "" }
func defaultPeerStatePath() string { return "" }
func runInferbench(ctx context.Context, cfg Config, spec string) error {
	return errors.New("unsupported")
}

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
