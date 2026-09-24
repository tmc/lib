package cudahf_test

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudahf"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

func Example() {
	// Lay out a hub cache the way huggingface_hub does.
	cache, err := os.MkdirTemp("", "hub")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(cache)
	repo := filepath.Join(cache, "models--org--tiny")
	snapshot := filepath.Join(repo, "snapshots", "0123abcd")
	if err := os.MkdirAll(snapshot, 0o755); err != nil {
		log.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "refs"), 0o755); err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "refs", "main"), []byte("0123abcd"), 0o644); err != nil {
		log.Fatal(err)
	}
	w := safetensors.Tensor{Name: "lm_head.weight", Dtype: safetensors.BF16, Shape: []int{2}, Data: make([]byte, 4)}
	if err := safetensors.WriteFile(filepath.Join(snapshot, "model.safetensors"), []safetensors.Tensor{w}, nil); err != nil {
		log.Fatal(err)
	}

	mf, err := cudahf.ResolveIn(cache, "org/tiny")
	if err != nil {
		log.Fatal(err)
	}
	ckpt, err := safetensors.OpenSharded(mf.SnapshotDir)
	if err != nil {
		log.Fatal(err)
	}
	defer ckpt.Close()
	info, _ := ckpt.Info("lm_head.weight")
	fmt.Println(filepath.Base(mf.SnapshotDir), info.Dtype, info.Shape)
	// Output: 0123abcd BF16 [2]
}
