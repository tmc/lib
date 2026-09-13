package cudaobjdump

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// TestOptionsAgainstRecordedToolkits asserts that every option this package can
// emit is accepted, with the same arity, by every cuobjdump the option corpus
// records.
//
// That is the measured state of affairs rather than an aspiration: across CUDA
// 12.6, 13.0 and 13.4, cuobjdump's option names and arities are identical, and
// the only difference is which architectures --gpu-architecture enumerates,
// which this package never passes. So there is nothing here to gate on a
// version, and this test is what says so: an option that starts to differ, or a
// new emitted option that some recorded toolkit lacks, fails here.
func TestOptionsAgainstRecordedToolkits(t *testing.T) {
	emitted := emittedFlags(t)
	versions := cudatool.KnownVersions(ToolName)
	if len(versions) < 2 {
		t.Fatalf("recorded toolkit versions for %s are %v; the comparison needs at least two", ToolName, versions)
	}
	for _, v := range versions {
		opts, source, ok := cudatool.KnownOptions(ToolName, v)
		if !ok {
			t.Fatalf("no recorded option set for %s %v", ToolName, v)
		}
		caps := cudatool.Caps{Tool: ToolName, Version: v, Source: source, Options: opts}
		if err := caps.Check(emitted...); err != nil {
			t.Errorf("%v: %v", v, err)
		}
	}
}

// emittedFlags returns every option this package puts on a cuobjdump command
// line, gathered from the accessors the package itself uses so that the list
// cannot drift from the code.
func emittedFlags(t *testing.T) []cudatool.Flag {
	t.Helper()
	flags := []cudatool.Flag{{Name: "--version"}}
	for _, kind := range []Kind{ELF, PTX} {
		list, err := kind.listOption()
		if err != nil {
			t.Fatalf("Kind(%d).listOption: %v", kind, err)
		}
		extract, err := kind.extractOption()
		if err != nil {
			t.Fatalf("Kind(%d).extractOption: %v", kind, err)
		}
		// --extract-elf takes the image to extract, "all" here.
		flags = append(flags, cudatool.Flag{Name: list}, cudatool.Flag{Name: extract, Value: "all"})
	}
	for _, section := range []Section{SASS, ELFSections, Resources} {
		option, err := section.option()
		if err != nil {
			t.Fatalf("Section(%d).option: %v", section, err)
		}
		flags = append(flags, cudatool.Flag{Name: option})
	}
	return flags
}

// TestUnknownKindAndSection covers the option accessors' rejection of values
// outside the enumerations, so a new Kind or Section cannot silently emit an
// empty option.
func TestUnknownKindAndSection(t *testing.T) {
	if _, err := Kind(99).listOption(); err == nil {
		t.Error("Kind(99).listOption returned an option")
	}
	if _, err := Kind(99).extractOption(); err == nil {
		t.Error("Kind(99).extractOption returned an option")
	}
	if _, err := Section(99).option(); err == nil {
		t.Error("Section(99).option returned an option")
	}
}
