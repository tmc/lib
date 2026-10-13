package cudainspect

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"debug/elf"
	"github.com/tmc/lib/nvidia/x/cudaobjdump"
)

func imageTool(t *testing.T, extract string) cudaobjdump.Tool {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	dir := t.TempDir()
	for _, name := range []string{"first.cubin", "second.cubin"} {
		if err := os.WriteFile(filepath.Join(dir, name), metadataELF(true, uint64(len(name)), elf.EM_CUDA), 0600); err != nil {
			t.Fatal(err)
		}
	}
	body := `#!/bin/sh
case "$1" in
--version) printf 'synthetic cuobjdump fixture\n';;
--list-elf) test -f "$2" || exit 9; printf 'ELF file 1: first.cubin\nELF file 2: second.cubin\n';;
--extract-elf) test "$2" != all || exit 8; test -f "$3" || exit 9
` + extract + `
;;
*) exit 7;;
esac
`
	path := filepath.Join(dir, "cuobjdump")
	if err := os.WriteFile(path, []byte(body), 0700); err != nil {
		t.Fatal(err)
	}
	return cudaobjdump.Tool{Path: path}
}

const extractImages = `cp "$(dirname "$0")/$2" "$2"`

func ExampleInspector_InspectImage() {
	i, _ := NewInspector(".")
	// Selection must be explicit; an empty name never picks the first image.
	_, err := i.InspectImage(context.Background(), cudaobjdump.Tool{}, []byte("container"), "", "kernel")
	fmt.Println(err != nil)
	// Output: true
}

func TestFatbinaryImageSelection(t *testing.T) {
	i, _ := NewInspector(t.TempDir())
	tool := imageTool(t, extractImages)
	container := []byte("synthetic container bytes")
	r, err := i.InspectImage(context.Background(), tool, container, "second.cubin", "kernel")
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(container)
	if r.ContainerDigest != hex.EncodeToString(digest[:]) || r.ContainerBytes != len(container) || r.ImageName != "second.cubin" {
		t.Fatalf("container identity: %+v", r)
	}
	image := metadataELF(true, uint64(len("second.cubin")), elf.EM_CUDA)
	selected := sha256.Sum256(image)
	if r.Report == nil || r.Report.Digest != hex.EncodeToString(selected[:]) || r.Report.SharedMemory != uint32(len("second.cubin")) {
		t.Fatalf("wrong selected image: %+v", r.Report)
	}
	if len(r.Commands) != 3 {
		t.Fatalf("receipts: %+v", r.Commands)
	}
	for j, option := range []string{"--version", "--list-elf", "--extract-elf"} {
		if r.Commands[j].Path != tool.Path || r.Commands[j].Args[0] != option {
			t.Fatalf("command %d: %+v", j, r.Commands[j])
		}
	}
	if len(r.Commands[0].Stdout) == 0 || len(r.Commands[1].Stdout) == 0 {
		t.Fatal("lost version or inventory output")
	}
	entries, err := os.ReadDir(i.WorkDir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("scratch cleanup: %v %v", entries, err)
	}
	if string(container) != "synthetic container bytes" {
		t.Fatal("mutated caller bytes")
	}
}

func TestImageInspectionFailuresRetainEvidence(t *testing.T) {
	for _, tt := range []struct {
		name, body, selection string
		commands              int
	}{
		{"missing_image", extractImages, "missing.cubin", 2},
		{"extract_failure", "printf failure >&2; exit 3", "first.cubin", 3},
		{"invalid_image", "printf bad > first.cubin", "first.cubin", 3},
	} {
		t.Run(tt.name, func(t *testing.T) {
			i, _ := NewInspector(t.TempDir())
			r, err := i.InspectImage(context.Background(), imageTool(t, tt.body), []byte("container"), tt.selection, "kernel")
			if err == nil || r == nil || len(r.Commands) != tt.commands || r.ContainerDigest == "" {
				t.Fatalf("result=%+v err=%v", r, err)
			}
			if tt.name == "extract_failure" && string(r.Commands[2].Stderr) != "failure" {
				t.Fatal("lost failed command stderr")
			}
			entries, err := os.ReadDir(i.WorkDir)
			if err != nil || len(entries) != 0 {
				t.Fatal("failed inspection leaked scratch")
			}
		})
	}
	i, _ := NewInspector(t.TempDir())
	r, err := i.InspectImage(context.Background(), cudaobjdump.Tool{Path: filepath.Join(i.WorkDir, "missing")}, []byte("container"), "x", "k")
	if err == nil || r == nil || len(r.Commands) != 1 {
		t.Fatal("missing tool lost receipt")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := i.InspectImage(ctx, cudaobjdump.Tool{}, []byte("container"), "x", "k"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
}
