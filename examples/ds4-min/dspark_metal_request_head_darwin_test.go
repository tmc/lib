//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	applemetal "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

type requestHeadArm struct {
	Name          string
	Threshold     float32
	Poison        bool
	From, To      string
	MismatchField int
}

func requestHeadArms() []requestHeadArm {
	return []requestHeadArm{
		{Name: "zero-accept", Threshold: 0, MismatchField: -1},
		{Name: "confidence-decline", Threshold: .5001, MismatchField: -1},
		{Name: "decline-before-nan", Threshold: .5001, Poison: true, MismatchField: -1},
		{Name: "zero-nan-reject", Threshold: 0, Poison: true, MismatchField: -1},
		{Name: "unquantized-token", From: "if(type!=8){", To: "if(true){", MismatchField: 2},
		{Name: "unquantized-decline", Threshold: .5001, From: "if(type!=8){", To: "if(true){", MismatchField: 0},
		{Name: "premature-logit-scan", Threshold: .5001, Poison: true, From: "if(probability<threshold)return;", To: "if(!isfinite(logits[row*vocab])){result[1]=1;return;}if(probability<threshold)return;", MismatchField: 1},
		{Name: "sigmoid-as-raw", From: "confidence_logits[row]=z;", To: "confidence_logits[row]=1.0f/(1.0f+exp(-z));", MismatchField: 6},
	}
}

type requestHeadOracle struct {
	Name   string
	Values [8]float64 // count, status, IDs, probabilities, raw confidence
}

type requestTensorSeal struct {
	Name       string
	Type       uint32
	Dimensions []uint64
	Bytes      int
	SHA256     string
}

type requestHeadManifest struct {
	PinnedC            string
	OracleSourceSHA256 string
	Arms               []requestHeadArm
	Oracle             []requestHeadOracle
	Finite, Poison     []requestTensorSeal
}

func requestRecipeSeals(poison bool) ([]requestTensorSeal, error) {
	var seals []requestTensorSeal
	for _, r := range requestHeadRecipes(poison) {
		digest, err := r.digest()
		if err != nil {
			return nil, err
		}
		seals = append(seals, requestTensorSeal{r.name, r.typeID, r.dimensions, r.size, fmt.Sprintf("%x", digest)})
	}
	return seals, nil
}

// Failed cleanup retains the complete request, including parent Device, so
// finalizers cannot free dependencies of an unfinished command buffer.
var requestHeadRetained struct {
	sync.Mutex
	owners []*requestHeadOwner
}

type requestHeadOwner struct {
	dev           *libmetal.Device
	ops           *metalPinnedOps
	arena         *metalmem.Arena
	weights       *metalDSparkWeights
	dense         *metalDSparkDense
	attention     *metalDSparkAttention
	batch         *metalBatch
	target        *metalRuntime
	originalHead  applemetal.MTLComputePipelineState
	mutantHead    applemetal.MTLComputePipelineState
	mutantLibrary applemetal.MTLLibrary
	pending       io.Closer
	closed        bool
}

func (f *requestHeadOwner) Close() error {
	if f == nil || f.closed {
		return nil
	}
	if f.batch != nil {
		if !f.batch.done {
			f.batch.endEncoding()
			f.batch.ops.hc.complete(f.batch)
			f.batch.done = true
		}
		if f.batch.buffer.Status() != applemetal.MTLCommandBufferStatusNotEnqueued && !metalGroupedTerminal(f.batch) {
			return fmt.Errorf("request head fixture: nonterminal command retains all owners")
		}
	}
	if f.pending != nil {
		if err := f.pending.Close(); err != nil {
			return err
		}
		f.pending = nil
	}
	if f.attention != nil {
		if f.originalHead != nil {
			f.attention.head = f.originalHead
		}
		f.attention.batches = nil // terminal, or permanently canceled above
		if err := f.attention.Close(); err != nil {
			return err
		}
		f.attention = nil
	}
	if f.dense != nil {
		f.dense.batches = nil
		if err := f.dense.Close(); err != nil {
			return err
		}
		f.dense = nil
	}
	if f.mutantHead != nil {
		objectivec.ObjectFromID(f.mutantHead.GetID()).Release()
		f.mutantHead = nil
	}
	if f.mutantLibrary != nil {
		objectivec.ObjectFromID(f.mutantLibrary.GetID()).Release()
		f.mutantLibrary = nil
	}
	if f.weights != nil {
		if err := f.weights.Close(); err != nil {
			return err
		}
		f.weights = nil
	}
	if f.arena != nil {
		f.arena.Free()
		f.arena = nil
	}
	if f.ops != nil {
		f.ops.Close()
		f.ops = nil
	}
	f.target = nil
	if f.dev != nil {
		if err := f.dev.Close(); err != nil {
			return err
		}
		f.dev = nil
	}
	f.closed = true
	return nil
}

func (f *requestHeadOwner) retain() {
	requestHeadRetained.Lock()
	requestHeadRetained.owners = append(requestHeadRetained.owners, f)
	requestHeadRetained.Unlock()
}

func newRequestHeadOwner(arm requestHeadArm, seals []requestTensorSeal) (f *requestHeadOwner, err error) {
	f = &requestHeadOwner{}
	defer func() {
		if err != nil {
			var closer io.Closer
			if errors.As(err, &closer) {
				f.pending = closer
			}
		}
	}()
	f.dev, err = libmetal.NewDevice()
	if err != nil {
		return f, err
	}
	f.ops, err = newMetalPinnedOps(f.dev)
	if err != nil {
		return f, err
	}
	f.arena = metalmem.NewArena(f.dev)
	f.weights = &metalDSparkWeights{device: f.dev, model: &dsparkModel{block: 2, stages: 3, rank: 32}, tensors: make(map[string]boundMetalTensor)}
	f.target = &metalRuntime{dev: f.dev, borrowed: true}
	recipes := requestHeadRecipes(arm.Poison)
	if len(seals) != len(recipes) {
		return f, fmt.Errorf("request head fixture: incomplete tensor seals")
	}
	for i, r := range recipes {
		seal := seals[i]
		if seal.Name != r.name || seal.Type != r.typeID || seal.Bytes != r.size || fmt.Sprint(seal.Dimensions) != fmt.Sprint(r.dimensions) {
			return f, fmt.Errorf("request head fixture: tensor metadata mismatch")
		}
		memory := metalmem.AllocIn[byte](f.arena, r.name, r.size)
		if err := f.arena.Err(); err != nil {
			return f, err
		}
		if err := r.fill(memory.HostSlice()); err != nil {
			return f, err
		}
		digest := sha256.Sum256(memory.HostSlice())
		if fmt.Sprintf("%x", digest) != seal.SHA256 {
			return f, fmt.Errorf("request head fixture: raw tensor hash mismatch")
		}
		tensor := boundMetalTensor{name: r.name, typeID: r.typeID, dimensions: r.dimensions, data: memory}
		if i == 0 {
			f.target.output.weight = tensor
		} else {
			f.weights.tensors[r.name] = tensor
		}
	}
	f.dense, err = newMetalDSparkDense(f.dev, f.weights)
	if err != nil {
		return f, err
	}
	f.attention, err = newMetalDSparkAttention(f.dev, f.weights)
	if err != nil {
		return f, err
	}
	if arm.From != "" {
		if strings.Count(metalDSparkAttentionMSL, arm.From) != 1 {
			return f, fmt.Errorf("request head fixture: nonunique mutation site")
		}
		source := strings.Replace(metalDSparkAttentionMSL, arm.From, arm.To, 1)
		f.mutantLibrary, err = f.dev.CompileLibrary(source)
		if err != nil {
			return f, err
		}
		f.mutantHead, err = f.dev.NewComputePipeline(f.mutantLibrary, "dspark_proposal_head")
		if err != nil {
			return f, err
		}
		f.originalHead, f.attention.head = f.attention.head, f.mutantHead
	}
	f.batch, err = newMetalBatch(f.ops)
	return f, err
}

func requestHeadMismatch(got, want [8]float64) (int, [8]bool) {
	var fields [8]bool
	n := 0
	for i := range got {
		tolerance := 0.0
		if i == 4 || i == 5 {
			tolerance = 2e-6
		}
		fields[i] = math.IsNaN(got[i]) || math.IsInf(got[i], 0) || math.Abs(got[i]-want[i]) > tolerance
		if fields[i] {
			n++
		}
	}
	return n, fields
}

func requestHeadLibrarySources() (map[string]string, error) {
	sources := map[string]string{"dense-base.metal": metalDSparkDenseMSL, "attention-base.metal": metalDSparkAttentionMSL}
	for _, arm := range requestHeadArms() {
		if arm.From == "" {
			continue
		}
		if strings.Count(metalDSparkAttentionMSL, arm.From) != 1 {
			return nil, fmt.Errorf("request library: nonunique mutation")
		}
		sources[arm.Name+".metal"] = strings.Replace(metalDSparkAttentionMSL, arm.From, arm.To, 1)
	}
	return sources, nil
}

func requestHeadCompileLibraries(dir string, log func(string, ...any)) (err error) {
	sources, err := requestHeadLibrarySources()
	if err != nil {
		return err
	}
	for name, source := range sources {
		frozen, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || string(frozen) != source {
			return fmt.Errorf("request library %s: frozen source mismatch: %v", name, err)
		}
	}
	f := &requestHeadOwner{}
	defer func() {
		if closeErr := f.Close(); closeErr != nil {
			f.retain()
			err = errors.Join(err, closeErr)
		}
	}()
	f.dev, err = libmetal.NewDevice()
	if err != nil {
		return err
	}
	f.weights = &metalDSparkWeights{device: f.dev, model: &dsparkModel{block: 2, stages: 3, rank: 32}}
	f.dense, err = newMetalDSparkDense(f.dev, f.weights)
	if err != nil {
		errors.As(err, &f.pending)
		return err
	}
	log("REQUEST_LIBRARY name=dense-base sha256=%x pipelines=1", sha256.Sum256([]byte(metalDSparkDenseMSL)))
	f.attention, err = newMetalDSparkAttention(f.dev, f.weights)
	if err != nil {
		errors.As(err, &f.pending)
		return err
	}
	log("REQUEST_LIBRARY name=attention-base sha256=%x pipelines=7", sha256.Sum256([]byte(metalDSparkAttentionMSL)))
	seen := make(map[string]bool)
	for _, arm := range requestHeadArms() {
		if arm.From == "" || seen[arm.From] {
			continue
		}
		seen[arm.From] = true
		if strings.Count(metalDSparkAttentionMSL, arm.From) != 1 {
			return fmt.Errorf("request library: nonunique mutation")
		}
		f.mutantLibrary, err = f.dev.CompileLibrary(strings.Replace(metalDSparkAttentionMSL, arm.From, arm.To, 1))
		if err != nil {
			return err
		}
		f.mutantHead, err = f.dev.NewComputePipeline(f.mutantLibrary, "dspark_proposal_head")
		if err != nil {
			return err
		}
		log("REQUEST_LIBRARY name=%s sha256=%x pipelines=1", arm.Name, sha256.Sum256([]byte(strings.Replace(metalDSparkAttentionMSL, arm.From, arm.To, 1))))
		objectivec.ObjectFromID(f.mutantHead.GetID()).Release()
		f.mutantHead = nil
		objectivec.ObjectFromID(f.mutantLibrary.GetID()).Release()
		f.mutantLibrary = nil
	}
	return nil
}

func TestMetalDSparkRequestHeadGate(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_REQUEST_HEAD_GATE") != "1" {
		t.Skip("reviewed request-head physical gate only")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("canonical gpu-lock required")
	}
	path := filepath.Join(os.Getenv("DS4_METAL_DSPARK_FIXTURE_DIR"), "request-head.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest requestHeadManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	arms := requestHeadArms()
	if len(manifest.Oracle) != 8 || fmt.Sprint(manifest.Arms) != fmt.Sprint(arms) {
		t.Fatal("request head fixture: missing or changed arm inventory")
	}
	seen := make(map[string]bool)
	for i, arm := range arms {
		if seen[arm.Name] || manifest.Oracle[i].Name != arm.Name {
			t.Fatal("request head fixture: duplicate or reordered oracle")
		}
		seen[arm.Name] = true
		t.Run(arm.Name, func(t *testing.T) {
			seals := manifest.Finite
			if arm.Poison {
				seals = manifest.Poison
			}
			f, err := newRequestHeadOwner(arm, seals)
			defer func() {
				if err := f.Close(); err != nil {
					f.retain()
					t.Errorf("CLEANUP retained: %v", err)
				} else {
					t.Log("CLEANUP terminal released")
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			h := make([]float32, 8192)
			h[0], h[4096] = 1, 1
			hidden := metalmem.UploadIn(f.arena, "hidden", h)
			logits := metalmem.AllocIn[float32](f.arena, "logits", 2*Flash.Vocab)
			ids := metalmem.UploadIn(f.arena, "ids", []int32{-99, -99})
			p := metalmem.UploadIn(f.arena, "probabilities", []float32{-99, -99})
			raw := metalmem.UploadIn(f.arena, "raw-confidence", []float32{-99, -99})
			result := metalmem.UploadIn(f.arena, "result", []uint32{99, 99})
			if err := f.arena.Err(); err != nil {
				t.Fatal(err)
			}
			if err := f.dense.EncodeProposalHead(f.batch, f.attention, f.target.output.weight, hidden, logits, ids, p, raw, result, 2, 0, arm.Threshold); err != nil {
				t.Fatal(err)
			}
			if err := f.batch.Wait(t.Context()); err != nil {
				t.Fatal(err)
			}
			if f.batch.buffer.Status() != applemetal.MTLCommandBufferStatusCompleted {
				t.Fatal("request head fixture: command did not complete")
			}
			got := [8]float64{float64(result.HostSlice()[0]), float64(result.HostSlice()[1]), float64(ids.HostSlice()[0]), float64(ids.HostSlice()[1]), float64(p.HostSlice()[0]), float64(p.HostSlice()[1]), float64(raw.HostSlice()[0]), float64(raw.HostSlice()[1])}
			changed, fields := requestHeadMismatch(got, manifest.Oracle[i].Values)
			if arm.From == "" && changed != 0 {
				t.Fatalf("baseline changed=%d/8 got=%v want=%v", changed, got, manifest.Oracle[i].Values)
			}
			if arm.From != "" && (changed == 0 || !fields[arm.MismatchField]) {
				t.Fatalf("mutation assertion absent changed=%d/8 fields=%v", changed, fields)
			}
			_, publishErr := metalDSparkProposalResult(ids.HostSlice(), p.HostSlice(), result.HostSlice(), 2, arm.Threshold)
			if arm.From == "" && ((publishErr != nil) != (got[1] != 0)) {
				t.Fatalf("terminal publication: %v", publishErr)
			}
			t.Logf("REQUEST_HEAD name=%s readout=%v denominator=8 changed=%d mutation=%t terminal=completed", arm.Name, got, changed, arm.From != "")
		})
	}
}
