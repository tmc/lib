package tileir

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

// SemanticMemoryArtifact contains the assembled CUBIN descriptor and observed
// diagnostics. Toolchain version is unobserved unless reported by the tool;
// the current tileiras Result has no version field. No launch is performed.
type SemanticMemoryArtifact struct {
	Artifact       *cudakernel.Artifact
	BytecodeDigest string
	Remarks        []tileiras.Remark
	Log            string
}

// AssembleSemanticMemory serializes a bound pointer-memory lowering and invokes
// tileiras.Assemble for an unsuffixed target. A successful artifact still
// requires separately qualified compiler/device execution and caller buffers.
func AssembleSemanticMemory(ctx context.Context, r *LoweringResult, target string) (*SemanticMemoryArtifact, error) {
	return assembleSemanticMemory(ctx, r, target, func(ctx context.Context, data []byte, target string) (*tileiras.Result, error) {
		return tileiras.Assemble(ctx, data, tileiras.WithGPUName(target), tileiras.WithRemarks("all"))
	})
}

// The assembler seam is private and passed by value; no public bypass or
// mutable package hook can substitute a fake compiler in production.
func assembleSemanticMemory(ctx context.Context, r *LoweringResult, target string, assemble func(context.Context, []byte, string) (*tileiras.Result, error)) (*SemanticMemoryArtifact, error) {
	digits, ok := strings.CutPrefix(target, "sm_")
	if !ok || digits == "" {
		return nil, fmt.Errorf("tileir: invalid target %q", target)
	}
	for _, c := range digits {
		if c < '0' || c > '9' {
			return nil, fmt.Errorf("%w: %q", tileiras.ErrArchSuffix, target)
		}
	}
	if err := cudakernel.ValidateTargetArch(target); err != nil {
		return nil, err
	}
	if ctx == nil {
		return nil, fmt.Errorf("tileir: nil assembly context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r == nil || r.Memory == nil {
		return nil, fmt.Errorf("tileir: missing semantic memory lowering")
	}
	if err := r.Memory.Verify(r); err != nil {
		return nil, err
	}
	if err := checkEntryABI(r); err != nil {
		return nil, err
	}
	data, err := r.Module.EncodeBytecode()
	if err != nil {
		return nil, err
	}
	result, err := assemble(ctx, data, target)
	if err != nil {
		return nil, fmt.Errorf("tileir: assemble semantic memory: %w", err)
	}
	if result == nil || len(result.CUBIN) == 0 || len(result.Fatbin) != 0 {
		return nil, fmt.Errorf("tileir: assembler did not return one CUBIN image")
	}
	info, err := cudakernel.ParseCUBIN(result.CUBIN)
	if err != nil {
		return nil, err
	}
	if info.Arch != target {
		return nil, fmt.Errorf("tileir: CUBIN target %q differs from requested %q", info.Arch, target)
	}
	entry := r.Module.Functions[0].Name
	found := false
	for _, symbol := range info.Symbols {
		found = found || symbol == entry
	}
	if !found {
		return nil, fmt.Errorf("tileir: CUBIN missing exact entry %q", entry)
	}
	for _, remark := range result.Remarks {
		if (remark.Kind != "Passed" && remark.Kind != "Failed" && remark.Kind != "Missed" && remark.Kind != "Analysis") || remark.Pass == "" || remark.Name == "" ||
			(remark.Function != "" && remark.Function != entry) || remark.Line < 0 || remark.Column < 0 {
			return nil, fmt.Errorf("tileir: malformed or unrelated assembler diagnostic")
		}
	}
	schema := r.Schema
	schema.Params = append([]cudakernel.Param(nil), schema.Params...)
	artifact, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, result.CUBIN, entry, schema, target,
		cudakernel.WithToolchain("tileiras (version unobserved)"), cudakernel.WithCompilerOptions([]string{"--gpu-name=" + target, "--opt-level=3", "--remarks=all", "--remark-format=yaml"}))
	if err != nil {
		return nil, err
	}
	remarks := append([]tileiras.Remark(nil), result.Remarks...)
	for i := range remarks {
		args := make(map[string]string, len(remarks[i].Args))
		for key, value := range remarks[i].Args {
			args[key] = value
		}
		remarks[i].Args = args
	}
	return &SemanticMemoryArtifact{Artifact: artifact, BytecodeDigest: r.Memory.BytecodeDigest(), Remarks: remarks, Log: result.Log}, nil
}

// checkEntryABI checks that the artifact schema describes the parameters
// of the entry function that the CUBIN will contain. A binding digests both,
// so it does not imply that they agree: an ordered reduction whose outputs are
// all empty has a parameterless entry and a pointer schema for its C++ source.
func checkEntryABI(r *LoweringResult) error {
	if len(r.Module.Functions) != 1 || r.Module.Functions[0] == nil {
		return fmt.Errorf("tileir: semantic memory module must have one entry function")
	}
	params := r.Module.Functions[0].Params
	if len(params) != len(r.Schema.Params) {
		return fmt.Errorf("tileir: entry has %d parameters but schema has %d", len(params), len(r.Schema.Params))
	}
	for i, p := range params {
		tile, ok := p.TypeVal.(*TileType)
		pointer := ok && tile != nil && tile.ElementType != nil && tile.ElementType.Kind() == KindPointer
		if pointer != r.Schema.Params[i].IsPointer {
			return fmt.Errorf("tileir: entry parameter %d pointer kind differs from schema", i)
		}
	}
	return nil
}
