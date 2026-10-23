package cudaplan_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cublasltx"
	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// These records describe this fixture's actual direct submissions, not public
// cudaagreement executor records or cudaplan.Completion receipts. The existing
// agreement/planner pointer-only ABI cannot describe these production kernels.
type hybridDirectRecord struct {
	entry, backend, digest string
	grid, block            [3]uint32
	shared                 uint32
	outputs                []string
	completed              bool
}

var errHybridMissingDispatch = errors.New("hybrid: missing direct dispatch")

func checkHybridDirectRecords(records []hybridDirectRecord, kernels []hybridKernel, digests map[string]string) error {
	if len(records) != len(kernels)+1 {
		return errHybridMissingDispatch
	}
	for i, k := range kernels {
		r := records[i]
		if r.entry != k.entry || r.backend != "direct-cuda-launch" || r.digest == "" || r.digest != digests[k.entry] || !r.completed ||
			r.grid != [3]uint32{hybridTokens, 1, 1} || r.block != [3]uint32{32, 1, 1} || r.shared != k.shared || !slices.Equal(r.outputs, k.outputs) {
			return fmt.Errorf("hybrid: direct receipt mismatch at %d", i)
		}
	}
	head := records[len(kernels)]
	if head.entry != "logits" || head.backend != "cublasltx.MatMulRowMajor" || !head.completed || !slices.Equal(head.outputs, []string{"logits"}) {
		return errors.New("hybrid: wrong logits backend")
	}
	return nil
}

func TestHybridTransformerDirectReceiptControls(t *testing.T) {
	// CPU-only contract controls are not evidence of any actual submission.
	kernels, err := hybridKernels()
	if err != nil {
		t.Fatal(err)
	}
	var records []hybridDirectRecord
	digests := make(map[string]string)
	for _, k := range kernels {
		digests[k.entry] = "test-only-digest"
		records = append(records, hybridDirectRecord{k.entry, "direct-cuda-launch", "test-only-digest", [3]uint32{hybridTokens, 1, 1}, [3]uint32{32, 1, 1}, k.shared, slices.Clone(k.outputs), true})
	}
	records = append(records, hybridDirectRecord{entry: "logits", backend: "cublasltx.MatMulRowMajor", outputs: []string{"logits"}, completed: true})
	if err := checkHybridDirectRecords(records, kernels, digests); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func([]hybridDirectRecord){
		func(r []hybridDirectRecord) { r[1].backend = "fallback" },
		func(r []hybridDirectRecord) { r[1].completed = false },
		func(r []hybridDirectRecord) { r[1].shared = 0 },
		func(r []hybridDirectRecord) { r[1].digest = "stale-artifact" },
		func(r []hybridDirectRecord) { r[1].outputs = nil },
		func(r []hybridDirectRecord) { r[5].backend = "host-reference" },
	} {
		bad := slices.Clone(records)
		change(bad)
		if checkHybridDirectRecords(bad, kernels, digests) == nil {
			t.Fatal("fabricated fallback/omission accepted")
		}
	}
	if checkHybridDirectRecords(records[:5], kernels, digests) == nil {
		t.Fatal("omitted logits dispatch accepted")
	}
}

// TestHybridTransformerNative directly executes the unchanged production
// projection/tree-attention/residual/FFN kernels and the cuBLASLt logits head.
// It is deliberately separately gated and supplies no graph/agreement, timing,
// whole-decoder or GateW/GateA qualification. It must never run in CPU selectors.
func TestHybridTransformerNative(t *testing.T) {
	dev, stream, target, owners := admitWorkflowNativeDevice(t, "CUDAPLAN_HYBRID_NATIVE")
	f := newHybridFixture()
	want, err := hybridReference(f)
	if err != nil {
		t.Fatal(err)
	}
	kernels, err := hybridKernels()
	if err != nil {
		t.Fatal(err)
	}
	retainError := func(err error) {
		var owner io.Closer
		if errors.As(err, &owner) && owner != nil {
			owners.add(workflowCompletion, "hybrid compiler recovery", owner.Close)
		}
	}
	mods := cudaagreement.NewModules(dev)
	owners.add(workflowModule, "hybrid production modules", mods.Close)
	artifacts := make(map[string]*cudakernel.Artifact)
	digests := make(map[string]string)
	compiled := make(map[string]*nvrtcx.CompiledModule)
	for _, k := range kernels {
		cm := compiled[k.source]
		if cm == nil {
			cm, err = nvrtcx.Compile(k.source, nvrtcx.WithName("hybrid_production.cu"), nvrtcx.WithPhysicalArchTarget(target))
			if err != nil {
				retainError(err)
				t.Fatal(err)
			}
			compiled[k.source] = cm
		}
		art, err := cm.ToArtifact(k.entry, k.schema, target)
		if err != nil {
			t.Fatal(err)
		}
		artifacts[k.entry] = art
		digests[k.entry] = art.Digest()
	}
	engine, err := cublasltx.NewEngine(dev, cublasltx.WithWorkspaceBytes(1<<20))
	if engine != nil {
		owners.add(workflowPlan, "hybrid logits engine", engine.Close)
	}
	if err != nil {
		t.Fatal(err)
	}
	data := make(map[string]cudamem.Slice[float32])
	allocFloat := func(name string, values []float32) {
		b, err := cudamem.Alloc[float32](dev, len(values))
		if b.Ptr() != 0 {
			owners.add(workflowBuffer, name, b.Free)
		}
		if err != nil {
			t.Fatal(err)
		}
		data[name] = b
		if err := b.CopyFromHost(values); err != nil {
			t.Fatal(err)
		}
	}
	for name, v := range f.data {
		allocFloat(name, v)
	}
	for name, v := range want {
		allocFloat(name, make([]float32, len(v)))
	}
	rows, err := cudamem.Alloc[int32](dev, hybridTokens)
	if rows.Ptr() != 0 {
		owners.add(workflowBuffer, "hybrid row map", rows.Free)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := rows.CopyFromHost(f.rows); err != nil {
		t.Fatal(err)
	}
	gens, err := cudamem.Alloc[uint64](dev, hybridTokens)
	if gens.Ptr() != 0 {
		owners.add(workflowBuffer, "hybrid generations", gens.Free)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := gens.CopyFromHost(f.generations); err != nil {
		t.Fatal(err)
	}
	masks, err := cudamem.Alloc[uint64](dev, hybridTokens)
	if masks.Ptr() != 0 {
		owners.add(workflowBuffer, "hybrid masks", masks.Free)
	}
	if err != nil {
		t.Fatal(err)
	}
	status, err := cudamem.Alloc[int32](dev, hybridTokens)
	if status.Ptr() != 0 {
		owners.add(workflowBuffer, "hybrid row status", status.Free)
	}
	if err != nil {
		t.Fatal(err)
	}
	ptrs := map[string]driver.DevicePtr{"token_rows": rows.Ptr(), "row_generations": gens.Ptr(), "token_generations": gens.Ptr(), "masks": masks.Ptr(), "row_status": status.Ptr()}
	for name, b := range data {
		ptrs[name] = b.Ptr()
	}
	run := func(mutant string) ([]hybridDirectRecord, error) {
		for name, v := range f.data {
			values := slices.Clone(v)
			if mutant == "projection" && name == "q_weight" {
				values[0] += 1
			}
			if err := data[name].CopyFromHost(values); err != nil {
				return nil, err
			}
		}
		maskValues := slices.Clone(f.masks)
		if mutant == "mask" {
			maskValues[2] = 4
		}
		if err := masks.CopyFromHost(maskValues); err != nil {
			return nil, err
		}
		if err := status.CopyFromHost(make([]int32, hybridTokens)); err != nil {
			return nil, err
		}
		for name, v := range want {
			poison := make([]float32, len(v))
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := data[name].CopyFromHost(poison); err != nil {
				return nil, err
			}
		}
		var records []hybridDirectRecord
		for _, k := range kernels {
			if mutant == "intermediate" && k.entry == "TreeFlashAttentionCUDA" {
				continue
			}
			var args []any
			for _, p := range k.schema.Params {
				if p.IsPointer {
					args = append(args, ptrs[p.Name])
					continue
				}
				switch p.Name {
				case "dim", "q_dim", "kv_dim", "head_dim":
					args = append(args, int32(hybridDim))
				case "intermediate_dim":
					args = append(args, int32(hybridFFN))
				case "num_tokens", "metadata_rows", "num_nodes":
					args = append(args, int32(hybridTokens))
				case "num_heads":
					args = append(args, int32(1))
				case "epsilon":
					args = append(args, hybridEpsilon)
				case "scale":
					args = append(args, hybridScale)
				default:
					return records, fmt.Errorf("hybrid: unbound scalar %s", p.Name)
				}
			}
			art := artifacts[k.entry]
			if _, err := art.Schema().Pack(args...); err != nil {
				return records, err
			}
			fn, err := mods.Function(art)
			if err != nil {
				return records, err
			}
			if err := cudalaunch.OwnedKernel(fn).Grid(hybridTokens, 1, 1).Block(32, 1, 1).SharedMem(k.shared).On(stream).Launch(args...); err != nil {
				return records, err
			}
			// Record only after the actual submission; completion is marked
			// separately after the final stream drain succeeds.
			records = append(records, hybridDirectRecord{k.entry, "direct-cuda-launch", art.Digest(), [3]uint32{hybridTokens, 1, 1}, [3]uint32{32, 1, 1}, k.shared, slices.Clone(k.outputs), false})
		}
		if err := engine.MatMulRowMajor(stream, data["hidden"], hybridTokens, hybridDim, data["head_weight"], hybridVocab, data["logits"]); err != nil {
			return records, err
		}
		records = append(records, hybridDirectRecord{entry: "logits", backend: "cublasltx.MatMulRowMajor", outputs: []string{"logits"}})
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if err := stream.Synchronize(ctx); err != nil {
			return records, err
		}
		for i := range records {
			records[i].completed = true
		}
		if err := checkHybridDirectRecords(records, kernels, digests); err != nil {
			if mutant == "intermediate" && errors.Is(err, errHybridMissingDispatch) {
				// Verify the omitted producer also left the actual downstream
				// logits poisoned. A copy/drain failure is infrastructure failure.
				got := make([]float32, hybridTokens*hybridVocab)
				if copyErr := data["logits"].CopyToHost(got); copyErr != nil {
					return records, copyErr
				}
				poisoned := false
				for _, v := range got {
					poisoned = poisoned || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)
				}
				if !poisoned {
					return records, errors.New("hybrid: omitted attention did not poison downstream logits")
				}
			}
			return records, err
		}
		rowStatus := make([]int32, hybridTokens)
		if err := status.CopyToHost(rowStatus); err != nil {
			return records, err
		}
		for _, s := range rowStatus {
			if s != 0 {
				return records, errors.New("hybrid: generation/row admission failed")
			}
		}
		for name, reference := range want {
			got := make([]float32, len(reference))
			if err := data[name].CopyToHost(got); err != nil {
				return records, err
			}
			if err := checkHybridValues(got, reference); err != nil {
				return records, fmt.Errorf("%s: %w", name, err)
			}
		}
		return records, nil
	}
	records, err := run("")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("target=%s direct-production hybrid receipts=%+v; logits and all materialized intermediates compared", target, records)
	for _, mutant := range []string{"projection", "mask", "intermediate"} {
		actual, err := run(mutant)
		wantCount, wantError := 6, errHybridNumericalMismatch
		if mutant == "intermediate" {
			wantCount, wantError = 5, errHybridMissingDispatch
		}
		if len(actual) != wantCount || !errors.Is(err, wantError) {
			t.Fatalf("native %s control failed before semantic refusal: records=%d err=%v", mutant, len(actual), err)
		}
		for _, r := range actual {
			if !r.completed {
				t.Fatalf("native %s control has unfinished submission: %+v", mutant, r)
			}
		}
		if mutant == "intermediate" {
			remaining := append(slices.Clone(kernels[:1]), kernels[2:]...)
			if err := checkHybridDirectRecords(actual, remaining, digests); err != nil {
				t.Fatalf("omitted-attention control changed other submissions: %v", err)
			}
		}
	}
}
