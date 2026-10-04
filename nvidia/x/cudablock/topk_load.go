package cudablock

import (
	"crypto/sha256"
	"fmt"
	"reflect"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// LoadTopK loads a retained ordinary FP32 artifact without compiling it.
// The artifact must declare TopKSource as its tracked cudablock_topk.cu
// dependency, the exact TopKSchema, and the topk_fp32 entry point. These
// checks bind trusted compilation receipts; they do not authenticate a compiler
// or prove that arbitrary supplied machine code implements the source.
// The descriptor and its slices are copied. The caller must exclude concurrent
// mutation during the call. Execute and Close follow CompileTopK's ownership
// rules. A nonnil result returned with an error must be closed.
func LoadTopK(dev *cuda.Device, config TopKConfig, artifact *cudakernel.Artifact) (*TopK, error) {
	if _, err := config.Extents(); err != nil {
		return nil, err
	}
	owned, err := topkRetainedArtifact(artifact)
	if err != nil {
		return nil, err
	}
	return topkLoadArtifact(dev, config, owned)
}

func topkRetainedArtifact(artifact *cudakernel.Artifact) (*cudakernel.Artifact, error) {
	if err := artifact.Validate(); err != nil {
		return nil, fmt.Errorf("cudablock: retained topk artifact: %w", err)
	}
	if artifact.Kind != cudakernel.KindPTX && artifact.Kind != cudakernel.KindCUBIN {
		return nil, fmt.Errorf("cudablock: retained topk image is not loadable")
	}
	schema, err := TopKSchema()
	if err != nil {
		return nil, err
	}
	if artifact.EntryPoint != "topk_fp32" || !reflect.DeepEqual(artifact.Schema, schema) {
		return nil, fmt.Errorf("cudablock: retained topk entry or schema differs")
	}
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(TopKSource)))
	found := false
	for _, dep := range artifact.Dependencies {
		if dep.Path != "cudablock_topk.cu" {
			continue
		}
		if found || !dep.Tracked || dep.Digest != digest || dep.ErrorMsg != "" {
			return nil, fmt.Errorf("cudablock: retained topk source binding differs")
		}
		found = true
	}
	if !found {
		return nil, fmt.Errorf("cudablock: retained topk source binding missing")
	}
	owned := *artifact
	owned.Raw = append([]byte(nil), artifact.Raw...)
	owned.Schema.Params = append([]cudakernel.Param(nil), artifact.Schema.Params...)
	owned.CompilerOpts = append([]string(nil), artifact.CompilerOpts...)
	owned.Dependencies = append([]cudakernel.Dependency(nil), artifact.Dependencies...)
	return &owned, nil
}
