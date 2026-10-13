package cudainspect

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// LoweringReport retains the supplied PTX identity, CUBIN inspection, and static
// comparison. Equal opcode counts do not establish semantic equivalence or that
// the supplied PTX produced the CUBIN. Commands cover the explicit version and
// disassembly requests; the disassembler may also probe tool capabilities.
type LoweringReport struct {
	Artifact    *ReportSchemaV1
	PTXDigest   string
	SASSDigest  string
	PTXTarget   string
	Version     cudasass.Output
	Disassembly cudasass.Output
	Comparison  *cudalowering.Report
}

// ComparePTX disassembles cubin and compares it with ptx using cudalowering's
// default static rules. Divergences remain in Comparison, not the return error.
// Errors retain partial evidence. Target strings must match exactly; a feature
// suffix absent from the CUBIN metadata is not assumed. This does not load code
// or authorize a candidate. The caller must keep the input slices unchanged.
func (i *Inspector) ComparePTX(ctx context.Context, tool cudasass.Disassembler, cubin, ptx []byte, symbol string) (*LoweringReport, error) {
	if i == nil || i.WorkDir == "" {
		return nil, fmt.Errorf("cudarecipe: work directory required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	h := sha256.Sum256(ptx)
	r := &LoweringReport{PTXDigest: hex.EncodeToString(h[:])}
	artifact, err := i.Inspect(cubin, symbol)
	r.Artifact = artifact
	if err != nil {
		return r, err
	}
	if !artifact.ELFParsed || artifact.ELFMachine != "EM_CUDA" {
		return r, fmt.Errorf("cudarecipe: comparison requires CUDA ELF")
	}
	m, err := cudaptx.Parse(bytes.NewReader(ptx))
	if err != nil {
		return r, fmt.Errorf("cudarecipe: parse PTX: %w", err)
	}
	r.PTXTarget = m.Target
	if m.Target == "" || artifact.Architecture == "" || m.Target != artifact.Architecture {
		return r, fmt.Errorf("cudarecipe: PTX target %q does not match observable CUBIN target %q", m.Target, artifact.Architecture)
	}
	if m.Kernel(symbol) == nil {
		return r, fmt.Errorf("cudarecipe: symbol %q absent from PTX", symbol)
	}
	tool.TempDir = i.WorkDir
	r.Version, err = tool.Version(ctx)
	if err != nil {
		return r, fmt.Errorf("cudarecipe: disassembler version: %w", err)
	}
	r.Disassembly, err = tool.Disassemble(ctx, cubin, cudasass.Options{Mode: cudasass.Text})
	if err != nil {
		return r, fmt.Errorf("cudarecipe: disassemble: %w", err)
	}
	h = sha256.Sum256(r.Disassembly.Stdout)
	r.SASSDigest = hex.EncodeToString(h[:])
	d, err := cudasass.Parse(bytes.NewReader(r.Disassembly.Stdout))
	if err != nil {
		return r, fmt.Errorf("cudarecipe: parse SASS: %w", err)
	}
	r.Comparison, err = cudalowering.Compare(m, d, nil)
	if err != nil {
		return r, fmt.Errorf("cudarecipe: compare lowering: %w", err)
	}
	return r, nil
}
