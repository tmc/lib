//go:build darwin

package main

import (
	"encoding/json"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"testing"
)

type persistentGQAStructuralReceipt struct {
	Schema                    string   `json:"schema"`
	Device                    string   `json:"device"`
	BinarySHA256              string   `json:"binary_sha256"`
	ModelSHA256               string   `json:"model_sha256"`
	Batch                     int      `json:"batch"`
	Pair                      int      `json:"pair"`
	Order                     []string `json:"order"`
	Arm                       string   `json:"arm"`
	Requests                  []string `json:"requests"`
	Commands                  int      `json:"commands"`
	Dispatches                int      `json:"dispatches"`
	AttentionDispatches       int      `json:"attention_dispatches"`
	AttentionThreadgroups     int      `json:"attention_threadgroups"`
	ThreadsPerThreadgroup     int      `json:"threads_per_threadgroup"`
	ThreadgroupMemoryBytes    int      `json:"threadgroup_memory_bytes"`
	WorkspaceBytes            int64    `json:"workspace_bytes"`
	DescriptorAddress         uint64   `json:"descriptor_address"`
	RowAddress                uint64   `json:"row_address"`
	KAddresses                []uint64 `json:"k_addresses"`
	VAddresses                []uint64 `json:"v_addresses"`
	FullLogitCoordinates      int      `json:"full_logit_coordinates"`
	HiddenCoordinates         int      `json:"hidden_coordinates"`
	PopulatedKVCoordinates    int      `json:"populated_kv_coordinates"`
	GPUStartTime              float64  `json:"gpu_start_time"`
	GPUEndTime                float64  `json:"gpu_end_time"`
	GPUCommandNanoseconds     int64    `json:"gpu_command_nanoseconds"`
	HostSubmitWaitNanoseconds int64    `json:"host_submit_wait_nanoseconds"`
}

func TestPersistentGQAPairedStructuralTrace(t *testing.T) {
	model := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if model == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	binarySHA256 := os.Getenv("LLAMA_GQA_BINARY_SHA256")
	modelSHA256 := os.Getenv("LLAMA_GQA_MODEL_SHA256")
	if binarySHA256 == "" || modelSHA256 == "" {
		t.Fatal("LLAMA_GQA_BINARY_SHA256 and LLAMA_GQA_MODEL_SHA256 must identify the measured artifacts")
	}
	pairs := 2
	if value := os.Getenv("LLAMA_GQA_STRUCTURAL_PAIRS"); value != "" {
		var err error
		pairs, err = strconv.Atoi(value)
		if err != nil || pairs < 1 || pairs > 100 {
			t.Fatal("LLAMA_GQA_STRUCTURAL_PAIRS must be in [1,100]")
		}
	}
	cfg := Llama3_2_1BConfig(model)
	cfg.Precision, cfg.MaxSeqLen = "f16", 64
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	enablePersistentGQATestRoute(t, eng) // Compilation precedes every arm.
	eng.testRequireGPUTimestamps = true
	rng := rand.New(rand.NewSource(45))
	medianByBatch := make(map[int][2]int64)

	for _, width := range []int{1, 2, 4} {
		t.Run("B"+itoaTest(width), func(t *testing.T) {
			runtime, leases, tokens, positions, rows := prepareStructuralPrefixes(t, eng, cfg, width)
			defer runtime.Close()
			for _, lease := range leases {
				defer lease.Release()
			}
			candidateCache, _ := eng.NewBatchedAttentionCache(cfg, width)
			defer candidateCache.Free()
			currentCache, _ := eng.NewBatchedAttentionCache(cfg, width)
			defer currentCache.Free()
			candidateTail, _ := eng.NewBatchedModelTail(cfg, width)
			defer candidateTail.Free()
			currentTail, _ := eng.NewBatchedModelTail(cfg, width)
			defer currentTail.Free()
			initial := structuralAddresses(runtime, candidateCache, currentCache)
			ledger := &physicalWaveLedger{}
			before := ledger.Snapshot()
			var currentGPU, persistentGPU []int64
			for pair := 0; pair < pairs; pair++ {
				clearStructuralCache(candidateCache)
				clearStructuralCache(currentCache)
				order := []string{"current", "persistent"}
				if rng.Intn(2) == 1 {
					order[0], order[1] = order[1], order[0]
				}
				type armResult struct {
					chain BatchedLayerChainResult
					tail  BatchedModelTailResult
					cache *BatchedAttentionCache
				}
				results := make(map[string]armResult, 2)
				for _, arm := range order {
					eng.testPersistentGQA = arm == "persistent"
					cache, tail := currentCache, currentTail
					if arm == "persistent" {
						cache, tail = candidateCache, candidateTail
					}
					chain, err := runtime.Run(t.Context(), eng, cfg, cache, leases, tokens, positions, rows)
					if err != nil {
						t.Fatal(err)
					}
					hidden := append([]float32(nil), chain.Hidden...)
					tailResult, err := tail.Run(eng, cfg, hidden, nil)
					if err != nil {
						t.Fatal(err)
					}
					chain.Hidden = hidden
					results[arm] = armResult{chain, tailResult, cache}
					requests := make([]string, width)
					for i := range requests {
						requests[i] = "row-" + strconv.Itoa(i)
					}
					if err := ledger.Record("metal-"+arm, requests, uint64(chain.Dispatches+tailResult.Dispatches), map[string]uint64{"metal_prefix": uint64(runtime.Stats().DeviceReadBytes)}, map[string]uint64{"metal_row_cache": uint64(chain.DeviceBytes)}); err != nil {
						t.Fatal(err)
					}
					receipt := makeStructuralReceipt(cfg, width, pair, order, arm, chain, tailResult, runtime, cache, eng.dev.Name(), binarySHA256, modelSHA256)
					encoded, _ := json.Marshal(receipt)
					t.Log(string(encoded))
				}
				current, persistent := results["current"], results["persistent"]
				currentGPU = append(currentGPU, gpuCommandNanoseconds(t, current.chain))
				persistentGPU = append(persistentGPU, gpuCommandNanoseconds(t, persistent.chain))
				checkVector(t, "hidden", persistent.chain.Hidden, current.chain.Hidden, 4e-4)
				checkVector(t, "logits", persistent.tail.Logits, current.tail.Logits, 8e-4)
				if len(persistent.tail.Tokens) != len(current.tail.Tokens) {
					t.Fatal("token lengths differ")
				}
				for i := range persistent.tail.Tokens {
					if persistent.tail.Tokens[i] != current.tail.Tokens[i] {
						t.Fatalf("tokens differ: %v %v", persistent.tail.Tokens, current.tail.Tokens)
					}
				}
				for layer := range cfg.NumLayers {
					checkVector(t, "K", persistent.cache.K[layer].HostSlice(), current.cache.K[layer].HostSlice(), 1e-6)
					checkVector(t, "V", persistent.cache.V[layer].HostSlice(), current.cache.V[layer].HostSlice(), 1e-6)
				}
				if got := structuralAddresses(runtime, candidateCache, currentCache); !equalUint64s(got, initial) {
					t.Fatalf("buffer residency changed: got=%v want=%v", got, initial)
				}
			}
			waves, err := ledger.Since(before)
			if err != nil || len(waves) != 2*pairs {
				t.Fatalf("schema-v4 waves=%d want=%d err=%v", len(waves), 2*pairs, err)
			}
			medianByBatch[width] = [2]int64{medianInt64(currentGPU), medianInt64(persistentGPU)}
			t.Logf("gpu-command-only B=%d pairs=%d current_median_ns=%d persistent_median_ns=%d host_wait_not_included=true", width, pairs, medianByBatch[width][0], medianByBatch[width][1])
		})
	}
	improved, regressed := 0, false
	for _, width := range []int{1, 2, 4} {
		m := medianByBatch[width]
		delta := float64(m[0]-m[1]) / float64(m[0])
		if delta >= 0.10 {
			improved++
		}
		if delta < -0.05 {
			regressed = true
		}
	}
	t.Logf("preregistered_gate_10pct_shapes=%d regression_over_5pct=%v", improved, regressed)
}

func prepareStructuralPrefixes(t *testing.T, eng *ModelEngine, cfg Config, width int) (*productionPrefixRuntime, []*productionPrefixLease, []int32, []int, []int) {
	t.Helper()
	runtime, err := newProductionPrefixRuntime(eng, cfg, width)
	if err != nil {
		t.Fatal(err)
	}
	leases := make([]*productionPrefixLease, width)
	tokens := make([]int32, width)
	positions := make([]int, width)
	rows := make([]int, width)
	for row := range width {
		n := 17 + row
		prefix := make([]int32, n)
		prefix[0] = cfg.BOSTokenID
		for i := 1; i < n; i++ {
			prefix[i] = int32(200 + row*37 + i)
		}
		source, _ := eng.NewBatchedAttentionCache(cfg, 1)
		packed := packTestPrompts(t, cfg, [][]int32{prefix})
		chain, err := RunSegmentedLayerChain(t.Context(), eng, cfg, source, packed)
		if err != nil {
			t.Fatal(err)
		}
		tail, _ := eng.NewBatchedModelTail(cfg, 1)
		result, err := tail.Run(eng, cfg, append([]float32(nil), chain.Hidden...), nil)
		tail.Free()
		if err != nil {
			t.Fatal(err)
		}
		if err := runtime.Put(t.Context(), prefix, result.Tokens[0], source, 0); err != nil {
			t.Fatal(err)
		}
		source.Free()
		lease, _, err := runtime.Acquire(append(prefix, result.Tokens[0]))
		if err != nil || lease == nil {
			t.Fatal(err)
		}
		leases[row] = lease
		tokens[row] = result.Tokens[0]
		positions[row] = n
		rows[row] = row
	}
	return runtime, leases, tokens, positions, rows
}

func clearStructuralCache(cache *BatchedAttentionCache) {
	for layer := range cache.K {
		clear(cache.K[layer].HostSlice())
		clear(cache.V[layer].HostSlice())
	}
}
func structuralAddresses(runtime *productionPrefixRuntime, caches ...*BatchedAttentionCache) []uint64 {
	s := runtime.workspace.stats()
	out := []uint64{s.DescriptorAddress, s.RowAddress}
	for _, cache := range caches {
		for layer := range cache.K {
			out = append(out, uint64(cache.K[layer].Buffer().GpuAddress())+uint64(cache.K[layer].Offset()), uint64(cache.V[layer].Buffer().GpuAddress())+uint64(cache.V[layer].Offset()))
		}
	}
	return out
}
func equalUint64s(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func makeStructuralReceipt(cfg Config, width, pair int, order []string, arm string, chain BatchedLayerChainResult, tail BatchedModelTailResult, runtime *productionPrefixRuntime, cache *BatchedAttentionCache, device, binarySHA256, modelSHA256 string) persistentGQAStructuralReceipt {
	ws := runtime.workspace.stats()
	r := persistentGQAStructuralReceipt{Schema: "llama-infer.physical-wave/v4", Device: device, BinarySHA256: binarySHA256, ModelSHA256: modelSHA256, Batch: width, Pair: pair, Order: append([]string(nil), order...), Arm: arm, Commands: 2, Dispatches: chain.Dispatches + tail.Dispatches, AttentionDispatches: cfg.NumLayers, AttentionThreadgroups: cfg.NumLayers * width * cfg.NumHeads, ThreadsPerThreadgroup: 128, WorkspaceBytes: ws.Bytes, DescriptorAddress: ws.DescriptorAddress, RowAddress: ws.RowAddress, FullLogitCoordinates: len(tail.Logits), HiddenCoordinates: len(chain.Hidden), PopulatedKVCoordinates: width * cfg.NumLayers * 2 * cfg.NumKVHeads * cfg.HeadDim, GPUStartTime: chain.GPUStartTime, GPUEndTime: chain.GPUEndTime, GPUCommandNanoseconds: int64((chain.GPUEndTime - chain.GPUStartTime) * 1e9), HostSubmitWaitNanoseconds: int64(chain.HostSubmitWait)}
	if arm == "persistent" {
		r.AttentionThreadgroups = chain.CandidateAttentionGroups
		r.ThreadgroupMemoryBytes = 16464
	}
	for layer := range cache.K {
		r.KAddresses = append(r.KAddresses, uint64(cache.K[layer].Buffer().GpuAddress())+uint64(cache.K[layer].Offset()))
		r.VAddresses = append(r.VAddresses, uint64(cache.V[layer].Buffer().GpuAddress())+uint64(cache.V[layer].Offset()))
	}
	r.Requests = make([]string, width)
	for i := range r.Requests {
		r.Requests[i] = "row-" + strconv.Itoa(i)
	}
	return r
}

func gpuCommandNanoseconds(t *testing.T, chain BatchedLayerChainResult) int64 {
	t.Helper()
	n := int64((chain.GPUEndTime - chain.GPUStartTime) * 1e9)
	if n <= 0 {
		t.Fatal("nonpositive GPU command timestamp interval")
	}
	return n
}

func medianInt64(values []int64) int64 {
	v := append([]int64(nil), values...)
	slices.Sort(v)
	if len(v)%2 == 1 {
		return v[len(v)/2]
	}
	return (v[len(v)/2-1] + v[len(v)/2]) / 2
}
