package cudacache_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCompiledArtifactCacheFixture(t *testing.T) {
	// Existing NVCC 13.4.59/ptxas output, target sm_90; see the fixture
	// README. Compiler/disassembler provenance does not imply GPU execution.
	encoded, err := os.ReadFile("../cudacapture/testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(encoded)); got != "825fc635f5ed57052e8f76d563862114735a0534bfc19c3698069391bae36f12" {
		t.Fatalf("compiled fixture changed: %s", got)
	}
	raw, err := hex.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "0d0f044503d4b56333c0263409098603a84a58ac36ed82af2cbd5354111b6a90" {
		t.Fatalf("decoded compiled fixture changed: %s", got)
	}
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "v", Type: "s32"},
	)
	if err != nil {
		t.Fatal(err)
	}
	art, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, "fill", schema, "sm_90")
	if err != nil {
		t.Fatalf("compiled fixture setup: %v", err)
	}
	// The adapter accepts caller-declared compiler metadata. This label
	// identifies retained NVCC bytes, not a new NVRTC call or probed tools.
	id := cudacache.CompilerIdentity{
		NVRTCVersion: "retained-nvcc-13.4.59",
		DriverSONAME: "fixture-no-driver",
		TargetArch:   "sm_90",
		Flags:        []string{"-arch=sm_90"},
		OutputKind:   nvrtcx.OutputCUBIN,
	}
	input := "cudacapture-kernels-nvcc-13.4.59"
	key := cudacache.CompilerCacheKey(input, id)
	artifactKey := key + "-" + art.Digest()
	ctx := t.Context()
	fresh := func(t *testing.T, root string) *cudacache.Cache {
		t.Helper()
		c, err := cudacache.New(cudacache.Config{RootDir: root})
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	store := func(t *testing.T, root string) {
		t.Helper()
		if err := cudacache.NewNVRTCAdapter(fresh(t, root)).Store(ctx, input, id, art); err != nil {
			t.Fatalf("Store compiled fixture: %v", err)
		}
	}
	lookup := func(t *testing.T, root string) {
		t.Helper()
		got, hit, err := cudacache.NewNVRTCAdapter(fresh(t, root)).Lookup(ctx, input, id)
		if err != nil || !hit || got == nil {
			t.Fatalf("fresh disk Lookup = %v, %v, %v", got, hit, err)
		}
		if !bytes.Equal(got.Bytes(), raw) || got.Digest() != art.Digest() ||
			got.Kind() != cudakernel.KindCUBIN || got.EntryPoint() != "fill" ||
			got.Target() != "sm_90" || !reflect.DeepEqual(got.Schema(), schema) {
			t.Fatal("compiled bytes or exact ABI/entry/target metadata did not roundtrip")
		}
	}

	t.Run("disk roundtrip and invalidation", func(t *testing.T) {
		root := t.TempDir()
		store(t, root)
		lookup(t, root)
		for _, change := range []struct {
			name string
			edit func(*cudacache.CompilerIdentity)
		}{
			{"compiler", func(id *cudacache.CompilerIdentity) { id.NVRTCVersion += "-changed" }},
			{"driver compatibility", func(id *cudacache.CompilerIdentity) { id.DriverSONAME += "-changed" }},
			{"target", func(id *cudacache.CompilerIdentity) { id.TargetArch = "sm_100" }},
			{"flags", func(id *cudacache.CompilerIdentity) { id.Flags = []string{"-arch=sm_90", "-O0"} }},
			{"output kind", func(id *cudacache.CompilerIdentity) { id.OutputKind = nvrtcx.OutputPTX }},
		} {
			t.Run(change.name, func(t *testing.T) {
				changed := id
				change.edit(&changed)
				if cudacache.CompilerCacheKey(input, changed) == key {
					t.Fatal("compatibility change did not change the key")
				}
				got, hit, err := cudacache.NewNVRTCAdapter(fresh(t, root)).Lookup(ctx, input, changed)
				if err != nil || hit || got != nil {
					t.Fatalf("changed identity Lookup = %v, %v, %v, want miss", got, hit, err)
				}
			})
		}
		// Identity misses must not retire the original compatible entry.
		lookup(t, root)
	})

	t.Run("separate receipts", func(t *testing.T) {
		root := t.TempDir()
		store(t, root)
		policies := []cudacache.SearchPolicyID{
			{Strategy: "beam", Seed: 1, ScoringEpoch: "fixture-epoch-1"},
			{Strategy: "mcts", Seed: 2, ScoringEpoch: "fixture-epoch-1"},
		}
		const workload = "fill-int32-128"
		for i, policy := range policies {
			// Synthetic evaluation metadata exercises persistence only. These
			// wall-channel values are not device measurements or qualification.
			receipt := &cudacache.SearchReceipt{
				PolicyID: policy, WorkloadKey: workload, CandidateID: fmt.Sprint("fill-", i),
				ArtifactKey: artifactKey, Complete: true,
				TimingKind: cudatrial.TimingWallElapsed,
				Timings:    map[string]int64{string(cudatrial.TimingWallElapsed): int64(100 + i)},
				RankScore:  float64(i + 1), Epoch: "fixture-epoch-1",
			}
			if err := fresh(t, root).PutReceipt(ctx, receipt); err != nil {
				t.Fatal(err)
			}
			got, err := fresh(t, root).GetReceipt(ctx, policy, workload)
			if err != nil || !reflect.DeepEqual(got, receipt) {
				t.Fatalf("fresh receipt roundtrip = %+v, %v, want %+v", got, err, receipt)
			}
			if !got.IsStale("fixture-epoch-2") || got.IsStale("fixture-epoch-1") {
				t.Fatal("receipt did not preserve scoring-epoch staleness")
			}
		}
		if cudacache.SearchReceiptKey(policies[0], workload) == cudacache.SearchReceiptKey(policies[1], workload) {
			t.Fatal("distinct policies share a receipt key")
		}
		original := policies[0]
		data, err := os.ReadFile(filepath.Join(root, "receipts", cudacache.SearchReceiptKey(original, workload)+".receipt"))
		if err != nil {
			t.Fatal(err)
		}
		for _, change := range []struct {
			name     string
			policy   cudacache.SearchPolicyID
			workload string
		}{
			{"workload", original, "fill-int32-256"},
			{"policy epoch", cudacache.SearchPolicyID{Strategy: "beam", Seed: 1, ScoringEpoch: "fixture-epoch-2"}, workload},
		} {
			t.Run(change.name, func(t *testing.T) {
				got, err := fresh(t, root).GetReceipt(ctx, change.policy, change.workload)
				if got != nil || !errors.Is(err, cudacache.ErrNotFound) {
					t.Fatalf("mismatched receipt query = %v, %v, want miss", got, err)
				}
				path := filepath.Join(root, "receipts", cudacache.SearchReceiptKey(change.policy, change.workload)+".receipt")
				if err := os.WriteFile(path, data, 0o600); err != nil {
					t.Fatal(err)
				}
				got, err = fresh(t, root).GetReceipt(ctx, change.policy, change.workload)
				if got != nil || !errors.Is(err, cudacache.ErrCorruptReceipt) {
					t.Fatalf("replayed receipt = %v, %v, want corrupt identity refusal", got, err)
				}
			})
		}
		lookup(t, root)
	})

	for _, damage := range []string{"truncated persisted write", "changed bytes old digest"} {
		t.Run(damage, func(t *testing.T) {
			root := t.TempDir()
			store(t, root)
			lookup(t, root) // Passing disk twin before the deliberate corruption.
			path := filepath.Join(root, "artifacts", artifactKey+".artifact")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			wantReason := "unmarshal failed"
			if damage == "truncated persisted write" {
				data = data[:len(data)/2]
			} else {
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(data, &fields); err != nil {
					t.Fatal(err)
				}
				changed := bytes.Clone(raw)
				changed[len(changed)/2] ^= 1
				fields["raw"], err = json.Marshal(changed)
				if err != nil {
					t.Fatal(err)
				}
				data, err = json.Marshal(fields)
				if err != nil {
					t.Fatal(err)
				}
				wantReason = "digest mismatch"
			}
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			got, hit, err := cudacache.NewNVRTCAdapter(fresh(t, root)).Lookup(ctx, input, id)
			if got != nil || hit || !errors.Is(err, cudacache.ErrCorruptArtifact) || !strings.Contains(err.Error(), wantReason) {
				t.Fatalf("damaged persisted artifact Lookup = %v, %v, %v; want data-admission refusal %q", got, hit, err, wantReason)
			}
		})
	}
}
