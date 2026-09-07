//go:build tpu

package main
import "context"
import "github.com/tmc/lib/examples/internal/continuousbatch"
import "github.com/tmc/lib/examples/internal/inferbench"

func benchmarkPhaseRange(phase string) func() { return func(){} }
func getBenchmarkEngine(cfg Config) (*ModelEngine, error) { return GetOrCompileTPUEngine(cfg) }
func benchmarkDeviceID(eng *ModelEngine) (string, error) { return "tpu", nil }
func benchmarkPhysicalTelemetry() (bool, float64, float64, float64) { return false, 0, 0, 0 }
func resetBenchmarkCaches(eng *ModelEngine) {}
func benchmarkObservedOutputSupported() bool { return false }
func benchmarkPeakMemorySupport(eng *ModelEngine) (bool, error) { return false, nil }
func getPeakDeviceMemoryBytes(eng *ModelEngine) int64 { return 0 }
func benchmarkSynchronize(ctx context.Context, eng *ModelEngine) error { return nil }
func benchmarkBackend(cfg Config, eng *ModelEngine) (servingBackend, error) { return nil, nil }
func benchmarkMemoryReset(eng *ModelEngine) {}
func benchmarkPeakMemory(eng *ModelEngine) (int64, error) { return 0, nil }

type observedOutputCollector struct{}
func newObservedOutputCollector() *observedOutputCollector { return &observedOutputCollector{} }
func (c *observedOutputCollector) AddLayer(layer int, kind string, data []float32) {}
func (c *observedOutputCollector) AddLogits(data []float32) {}
func (c *observedOutputCollector) Flush() error { return nil }
func (c *observedOutputCollector) Close() error { return nil }

func (c *observedOutputCollector) observeKV(state *continuousbatch.State, cache *BatchedAttentionCache, cfg Config) {}
func (c *observedOutputCollector) fail(err error) {}
func (c *observedOutputCollector) observeLogits(states []*continuousbatch.State, logits []float32, vocabSize int) error { return nil }
func (c *observedOutputCollector) outputs(prompts []string) ([]inferbench.ObservedOutput, error) { return nil, nil }

type platformServingBackend struct{}
func newPlatformServingBackend(eng *ModelEngine, cfg Config, p int) (servingBackend, error) { return &platformServingBackend{}, nil }
func (s *platformServingBackend) Shutdown(context.Context) error { return nil }
func (s *platformServingBackend) Prefill(promptTokens []int32) (int32, error) { return 0, nil }
func (s *platformServingBackend) PrefillBatch(batchTokens [][]int32) ([]int32, error) { return nil, nil }
func (s *platformServingBackend) DecodeBatch(batchTokens []int32) ([]int32, error) { return nil, nil }
func (s *platformServingBackend) Release(id int64) {}
func (s *platformServingBackend) Close() error { return nil }


type servingBackend interface {
	Shutdown(context.Context) error
}
