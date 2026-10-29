package cudahostemu

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// MultimemHeader is the filename of the multi-rank NVLink symmetric memory header.
const MultimemHeader = "multimem.h"

//go:embed multimem.h
var multimemHeaderBytes []byte

// ErrNVLinkOrderingViolation is returned by RunMultiRank when a kernel performs
// a cross-rank multimem.ld_reduce or remote peer read before all ranks have
// synchronized at eg_emu_nvlink_barrier for an epoch with prior writes.
var ErrNVLinkOrderingViolation = errors.New("cudahostemu: NVLink ordering violation")

// MultimemSource returns the embedded multimem.h header bytes.
func MultimemSource() []byte {
	out := make([]byte, len(multimemHeaderBytes))
	copy(out, multimemHeaderBytes)
	return out
}

// RunMultiRank compiles and executes src across numRanks simulated GPUs
// simultaneously on the host CPU. Each rank r in 0..numRanks-1 executes its CTA
// grid on its own C++ thread(s) with eg_emu_rank() == r and
// eg_emu_num_ranks() == numRanks. Any Arg with a non-empty Data slice is
// updated in place with the buffer's final contents after the launch completes.
func RunMultiRank(ctx context.Context, src string, numRanks int, launch Launch, args ...Arg) error {
	if numRanks < 1 {
		return fmt.Errorf("cudahostemu: invalid numRanks %d (must be >= 1)", numRanks)
	}
	dir, err := os.MkdirTemp("", "cudahostemu-multimem-*")
	if err != nil {
		return fmt.Errorf("multimem temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, MultimemHeader), multimemHeaderBytes, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", MultimemHeader, err)
	}

	launchStmt, bufs, err := buildHazardLaunch(src, launch, args)
	if err != nil {
		return err
	}

	// Map static __shared__ arrays to per-CTA keyed shared memory so CTAs
	// executing simultaneously across multiple simulated GPUs never alias.
	rewritten := reStaticShared.ReplaceAllString(src, `EG_EMU_SHARED($1, $2, $3);`)

	var fullSrc strings.Builder
	fullSrc.WriteString("#include \"" + MultimemHeader + "\"\n")
	fullSrc.WriteString("#line 1 \"source\"\n")
	fullSrc.WriteString(rewritten)

	var fullLaunch strings.Builder
	fmt.Fprintf(&fullLaunch, "eg_emu_multimem_reset(%d);\n", numRanks)
	for _, b := range bufs {
		fmt.Fprintf(&fullLaunch, "    eg_emu_register_sym_buffer(%s, %d, %d);\n", b.Name, len(b.Data), numRanks)
	}
	fmt.Fprintf(&fullLaunch, "    %s", launchStmt)

	res, err := Run(ctx, dir, Program{
		Source:  fullSrc.String(),
		Launch:  fullLaunch.String(),
		Buffers: bufs,
		Flags:   launch.Flags,
		Mock:    launch.Mock,
	})
	if err != nil {
		if strings.Contains(err.Error(), "NVLink ordering violation") {
			return fmt.Errorf("%w: %v", ErrNVLinkOrderingViolation, err)
		}
		return err
	}

	bufIdx := 0
	for i := range args {
		a := &args[i]
		if a.Expr != "" && a.CType == "" && a.Data == nil && a.Size == 0 {
			continue
		}
		if bufIdx < len(bufs) {
			if out, ok := res[bufs[bufIdx].Name]; ok && len(a.Data) > 0 {
				copy(a.Data, out)
			}
			bufIdx++
		}
	}
	return nil
}
