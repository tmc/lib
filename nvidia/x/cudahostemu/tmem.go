package cudahostemu

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// TMEMHeaderFile is the filename of the Blackwell TMEM and tcgen05 host emulation C++ header.
const TMEMHeaderFile = "tmem.h"

//go:embed tmem.h
var tmemHeaderBytes []byte

// TMEMHeader returns the C++ header implementing Blackwell sm_100a Tensor Memory
// (TMEM) and tcgen05 host emulation.
func TMEMHeader() string {
	return string(tmemHeaderBytes)
}

// RunWithTMEM compiles and runs src on the host CPU with Blackwell sm_100a TMEM
// and tcgen05 emulation enabled. Any BufferArg with non-empty Data is updated in
// place with the buffer's post-launch contents. After the kernel finishes,
// RunWithTMEM verifies eg_emu_tmem_error_count() == 0 and returns a descriptive
// error if any TMEM synchronization hazard (such as tcgen05.ld before
// tcgen05.wait) or TMEM lifecycle error (such as leaked columns on CTA exit)
// occurred.
func RunWithTMEM(ctx context.Context, src string, launch Launch, args ...Arg) error {
	dir, err := os.MkdirTemp("", "cudahostemu-tmem-*")
	if err != nil {
		return fmt.Errorf("tmem temp dir: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, TMEMHeaderFile), tmemHeaderBytes, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", TMEMHeaderFile, err)
	}

	errFile := filepath.Join(dir, "tmem_errors.txt")
	launchStmt, bufs, err := buildHazardLaunch(src, launch, args)
	if err != nil {
		return err
	}
	launchStmt = launchStmt + "; eg_emu_tmem_flush_errors()"

	var fullSrc strings.Builder
	fullSrc.WriteString("#define EG_EMU 1\n")
	fmt.Fprintf(&fullSrc, "#define EG_EMU_TMEM_ERROR_FILE %q\n", errFile)
	fullSrc.WriteString("#include \"" + TMEMHeaderFile + "\"\n")
	fullSrc.WriteString("#line 1 \"source\"\n")
	fullSrc.WriteString(src)

	flags := append([]string{"-DEG_EMU=1"}, launch.Flags...)
	res, err := Run(ctx, dir, Program{
		Source:  fullSrc.String(),
		Launch:  launchStmt,
		Buffers: bufs,
		Flags:   flags,
		Mock:    launch.Mock,
	})
	if err != nil {
		return err
	}

	// Copy output buffer contents back into any caller-supplied Data slices.
	for _, b := range bufs {
		if out, ok := res[b.Name]; ok && len(b.Data) == len(out) {
			copy(b.Data, out)
		}
	}

	count, details, err := readTMEMErrorFile(errFile)
	if err != nil {
		return err
	}
	if count > 0 {
		return fmt.Errorf("cudahostemu: %d TMEM error(s): %s", count, strings.Join(details, "; "))
	}
	return nil
}

func readTMEMErrorFile(path string) (int, []string, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return 0, nil, nil
	}
	if err != nil {
		return 0, nil, fmt.Errorf("read tmem error report: %w", err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return 0, nil, nil
	}
	count, err := strconv.Atoi(strings.TrimSpace(lines[0]))
	if err != nil {
		return 0, nil, fmt.Errorf("parse tmem error count %q: %w", lines[0], err)
	}
	var details []string
	for _, line := range lines[1:] {
		line = strings.TrimSpace(line)
		if line != "" {
			details = append(details, line)
		}
	}
	return count, details, nil
}
