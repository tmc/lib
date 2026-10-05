package cudaschedule

import (
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
)

func snapshotOutput(out *cudaoutputcoverage.RecordedOutput) (*cudaoutputcoverage.RecordedOutput, error) {
	if out == nil {
		return nil, nil
	}
	if len(out.Ordinary) > 1<<24 || len(out.TwoFill.Reference) > 1<<24 ||
		len(out.TwoFill.ReadbackA) > 1<<24 || len(out.TwoFill.ReadbackB) > 1<<24 ||
		len(out.TwoFill.Contract.Domain) > 1<<20 {
		return nil, fmt.Errorf("output contract: recorded evidence exceeds limits")
	}
	copy := *out
	copy.Ordinary = slices.Clone(out.Ordinary)
	copy.TwoFill.Reference = slices.Clone(out.TwoFill.Reference)
	copy.TwoFill.ReadbackA = slices.Clone(out.TwoFill.ReadbackA)
	copy.TwoFill.ReadbackB = slices.Clone(out.TwoFill.ReadbackB)
	copy.TwoFill.Contract.Domain = slices.Clone(out.TwoFill.Contract.Domain)
	return &copy, nil
}
