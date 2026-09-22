package cudaobjdump

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// Output retains the command identity and both output streams, even on failure.
type Output = cudatool.Output

// ToolName is the executable [Tool] runs. It is also the name
// [cudatool.Capabilities] records option sets under.
const ToolName = "cuobjdump"

// Tool selects cuobjdump and a scratch parent. The zero value searches PATH
// and uses the operating system's temporary directory.
type Tool struct{ Path, TempDir string }

func (t Tool) path() string {
	if t.Path != "" {
		return t.Path
	}
	return "cuobjdump"
}

// Version reports the selected tool's version output.
func (t Tool) Version(ctx context.Context) (Output, error) {
	return cudatool.Run(ctx, t.path(), "", nil, "--version")
}

// Kind selects the embedded image format.
type Kind int

const (
	ELF Kind = iota
	PTX
)

// listOption and extractOption are the options that inventory and extract
// images of this kind. They exist as functions so that the option set this
// package can emit is reachable from a test without a toolkit installed; see
// TestOptionsAgainstRecordedToolkits.
func (k Kind) listOption() (string, error) {
	s, err := k.suffix()
	if err != nil {
		return "", err
	}
	return "--list-" + s, nil
}

func (k Kind) extractOption() (string, error) {
	s, err := k.suffix()
	if err != nil {
		return "", err
	}
	return "--extract-" + s, nil
}

// option returns the option that dumps this section.
func (s Section) option() (string, error) {
	switch s {
	case SASS:
		return "--dump-sass", nil
	case ELFSections:
		return "--dump-elf", nil
	case Resources:
		return "--dump-resource-usage", nil
	}
	return "", fmt.Errorf("unknown dump section %d", s)
}

func (k Kind) suffix() (string, error) {
	switch k {
	case ELF:
		return "elf", nil
	case PTX:
		return "ptx", nil
	}
	return "", fmt.Errorf("unknown image kind %d", k)
}

func inputPath(path string) (string, error) {
	p, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	s, err := os.Stat(p)
	if err != nil {
		return "", err
	}
	if !s.Mode().IsRegular() {
		return "", fmt.Errorf("input is not a regular file")
	}
	return p, nil
}

// Image identifies an embedded file as reported by cuobjdump.
type Image struct {
	Index int
	Name  string
}

// List inventories ELF or PTX images. Separate invocations are necessary because
// cuobjdump's list options override other operations.
func (t Tool) List(ctx context.Context, input string, kind Kind) ([]Image, Output, error) {
	option, err := kind.listOption()
	if err != nil {
		return nil, Output{}, err
	}
	p, err := inputPath(input)
	if err != nil {
		return nil, Output{}, err
	}
	out, err := cudatool.Run(ctx, t.path(), "", nil, option, p)
	if err != nil {
		return nil, out, err
	}
	images, err := ParseList(string(out.Stdout), kind)
	return images, out, err
}

// ParseList parses list output, rejecting nonempty unrecognized lines so a
// changed tool format cannot silently become an empty successful inventory.
func ParseList(text string, kind Kind) ([]Image, error) {
	s, err := kind.suffix()
	if err != nil {
		return nil, err
	}
	var images []Image
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		rest, ok := strings.CutPrefix(line, strings.ToUpper(s)+" file")
		if !ok {
			return nil, fmt.Errorf("unrecognized inventory line %q", line)
		}
		index, name, ok := strings.Cut(rest, ":")
		n, err := strconv.Atoi(strings.TrimSpace(index))
		if !ok || err != nil || n <= 0 || strings.TrimSpace(name) == "" {
			return nil, fmt.Errorf("invalid inventory line %q", line)
		}
		images = append(images, Image{n, strings.TrimSpace(name)})
	}
	return images, nil
}

// Section selects one textual inspection operation.
type Section int

const (
	SASS Section = iota
	ELFSections
	Resources
)

// Dump reports SASS, ELF sections, or resource usage. Resource output is kept
// verbatim because common and per-function fields vary with toolkit versions.
func (t Tool) Dump(ctx context.Context, input string, section Section) (Output, error) {
	option, err := section.option()
	if err != nil {
		return Output{}, err
	}
	p, err := inputPath(input)
	if err != nil {
		return Output{}, err
	}
	return cudatool.Run(ctx, t.path(), "", nil, option, p)
}

// Artifact owns one extracted image's bytes. Name is a basename, not a path.
type Artifact struct {
	Name string
	Data []byte
}

// Extract extracts all images of kind to a fresh directory, reads only regular
// files, then removes that directory. The original input is never an output.
// It trusts the selected executable, not arbitrary scripts or hostile tools.
func (t Tool) Extract(ctx context.Context, input string, kind Kind) (artifacts []Artifact, output Output, err error) {
	artifacts, receipts, err := t.ExtractWithReceipts(ctx, input, kind)
	if len(receipts) > 0 {
		output = receipts[len(receipts)-1]
	}
	return artifacts, output, err
}

// ExtractWithReceipts extracts images as Extract does, retaining both the
// inventory and extraction command outputs, including the failing command.
// The returned artifacts and receipts belong to the caller.
func (t Tool) ExtractWithReceipts(ctx context.Context, input string, kind Kind) (artifacts []Artifact, receipts []Output, err error) {
	return t.extract(ctx, input, kind, "")
}

// ExtractImage extracts one exactly named image and retains inventory and
// extraction receipts. cuobjdump matches substrings, so names matching other
// inventory entries are refused before extraction. Empty names, commas and the
// special selector "all" are not exact selectors.
func (t Tool) ExtractImage(ctx context.Context, input string, kind Kind, name string) (Artifact, []Output, error) {
	if name == "" || name == "all" || strings.Contains(name, ",") {
		return Artifact{}, nil, fmt.Errorf("invalid exact image selector %q", name)
	}
	images, receipts, err := t.extract(ctx, input, kind, name)
	if err != nil {
		return Artifact{}, receipts, err
	}
	return images[0], receipts, nil
}

func (t Tool) extract(ctx context.Context, input string, kind Kind, selected string) (artifacts []Artifact, receipts []Output, err error) {
	option, err := kind.extractOption()
	if err != nil {
		return nil, nil, err
	}
	p, err := inputPath(input)
	if err != nil {
		return nil, nil, err
	}
	images, inventory, err := t.List(ctx, p, kind)
	receipts = append(receipts, inventory)
	if err != nil {
		return nil, receipts, err
	}
	names := make(map[string]bool)
	for _, image := range images {
		name := image.Name
		if name == "." || name == ".." || strings.ContainsAny(name, "/\\\x00") || names[name] {
			return nil, receipts, fmt.Errorf("unsafe or duplicate extraction name %q", name)
		}
		names[name] = true
	}
	selector := "all"
	if selected != "" {
		if !names[selected] {
			return nil, receipts, fmt.Errorf("image %q not found", selected)
		}
		for name := range names {
			if name != selected && strings.Contains(name, selected) {
				return nil, receipts, fmt.Errorf("image selector %q also matches %q", selected, name)
			}
		}
		selector = selected
		names = map[string]bool{selected: true}
	}
	dir, err := os.MkdirTemp(t.TempDir, "cudaobjdump-")
	if err != nil {
		return nil, receipts, err
	}
	defer func() {
		if e := os.RemoveAll(dir); e != nil {
			err = errors.Join(err, fmt.Errorf("remove scratch %s: %w", dir, e))
		}
	}()
	out, err := cudatool.Run(ctx, t.path(), dir, nil, option, selector, p)
	receipts = append(receipts, out)
	if err != nil {
		return nil, receipts, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, receipts, err
	}
	var result []Artifact
	for _, e := range entries {
		if !names[e.Name()] {
			return nil, receipts, fmt.Errorf("unexpected extracted file %q", e.Name())
		}
		info, err := e.Info()
		if err != nil {
			return nil, receipts, err
		}
		if !info.Mode().IsRegular() {
			return nil, receipts, fmt.Errorf("extracted %q is not a regular file", e.Name())
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, receipts, err
		}
		result = append(result, Artifact{e.Name(), b})
	}
	if len(result) != len(names) {
		return nil, receipts, fmt.Errorf("extracted %d images, expected %d selected images", len(result), len(names))
	}
	return result, receipts, nil
}
