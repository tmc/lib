package cudalineinfo

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// WritePprof writes the PC samples of batches to w as a gzipped pprof profile
// whose stacks run from the sampled SASS instruction up through PTX, the
// generated CUDA line, and the program statement that produced it.
//
// It is [WriteProfile]'s counterpart: that one reports samples per line, this
// one keeps each line's stall reasons, which is the reading a text table
// cannot carry. The sample types are [cuptipcsampling.StallSampleTypes] plus
// the two values every batch writes, so the profile carries every column the
// bridge fills; a caller recording launches or SASS metrics alongside has
// types of its own to select and assembles the session itself.
//
// Host frames come from a.HostFrames, which reports none for a hand-written
// unit: there is no statement above forward.cu:10, so the CUDA line is the
// leaf and the profile is device frames only.
func WritePprof(ctx context.Context, w io.Writer, a *Attribution, inv cuptipcsampling.ReasonInventory, batches ...cuptipcsampling.Batch) error {
	if a == nil {
		return fmt.Errorf("cudalineinfo: nil attribution")
	}
	prof := cudaprof.NewSession(w)
	if err := prof.SetSampleTypes(append(cuptipcsampling.StallSampleTypes(), "pc_samples", "inst_executed")...); err != nil {
		return fmt.Errorf("cudalineinfo: %w", err)
	}
	// pc_samples is the only type every capture fills, so it is the view a
	// consumer should open on; a stall type can be legitimately zero.
	if err := prof.SetDefaultSampleType("pc_samples"); err != nil {
		return fmt.Errorf("cudalineinfo: %w", err)
	}
	prof.AddComment(profileComment(a))
	resolver, err := NewResolver(a, a.Module.CUBIN())
	if err != nil {
		return err
	}
	bridge := &cuptipcsampling.ProfileBridge{
		Session:    prof,
		Inventory:  inv,
		Resolver:   resolver,
		PTXOffsets: cudaprof.PTXLineOffsets(a.PTX),
		HostFrames: a.HostFrames,
	}
	for _, b := range batches {
		if !b.Valid {
			continue
		}
		if err := bridge.RecordBatch(ctx, b); err != nil {
			return err
		}
	}
	// The selection above is derived, so this holds only if the bridge grows a
	// value neither it nor this list knows about. It is a comment rather than
	// an error because the profile is already assembled and a reader who has
	// the artifact is the one who needs to know the breakdown is short.
	if dropped := bridge.DroppedSampleTypes(); len(dropped) > 0 {
		prof.AddComment("incomplete: the bridge counted into " + strings.Join(dropped, ", ") + ", which this profile does not carry")
	}
	if names := unmappedNames(bridge.UnmappedStalls); names != "" {
		prof.AddComment("stall_other holds reasons this build does not model: " + names)
	}
	return prof.Close()
}

// profileComment says what the profile is and what the join behind it reached,
// since a profile of an attribution that named no line is empty for a reason
// no consumer can see in the samples.
func profileComment(a *Attribution) string {
	c := fmt.Sprintf("cudalineinfo: PC samples of %s, %d of %d SASS instructions naming a program line",
		a.Unit.Name, a.Attributed, a.Instructions)
	if a.Unattributed() {
		c += "; nothing attributed, the unit is " + unitDescription(a.Unit)
	}
	return c
}

// unmappedNames lists the native reason names that reached no stall type,
// largest first. They are counted as stall_other in the profile, so without
// them the artifact says the breakdown is incomplete and not what is missing.
func unmappedNames(unmapped map[string]int64) string {
	if len(unmapped) == 0 {
		return ""
	}
	names := make([]string, 0, len(unmapped))
	for name := range unmapped {
		names = append(names, name)
	}
	slices.SortFunc(names, func(a, b string) int {
		if d := cmp.Compare(unmapped[b], unmapped[a]); d != 0 {
			return d
		}
		return cmp.Compare(a, b)
	})
	for i, name := range names {
		names[i] = fmt.Sprintf("%s (%d)", name, unmapped[name])
	}
	return strings.Join(names, ", ")
}
