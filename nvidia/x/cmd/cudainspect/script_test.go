package main

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudainspect"
	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// TestScript runs the scripts in testdata/script. The cudainspect command
// calls run in this process; no program is built or executed. The CUBIN and
// SASS fixtures are the recorded nvcc 13.4 sm_121a pair documented in
// ../../cudasass/testdata/roundtrip/README.md, plus synthetic ELF headers.
func TestScript(t *testing.T) {
	roundtrip, err := filepath.Abs("../../cudasass/testdata/roundtrip")
	if err != nil {
		t.Fatal(err)
	}
	cmds := scripttest.DefaultCmds()
	cmds["cudainspect"] = script.Command(
		script.CmdUsage{Summary: "run cudainspect in process", Args: "args..."},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			var stdout, stderr bytes.Buffer
			code := run(args, &stdout, &stderr)
			return func(*script.State) (string, string, error) {
				if code != 0 {
					return stdout.String(), stderr.String(), fmt.Errorf("exit status %d", code)
				}
				return stdout.String(), stderr.String(), nil
			}, nil
		})
	cmds["unhex"] = script.Command(
		script.CmdUsage{Summary: "decode a hex fixture", Args: "src dst"},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) != 2 {
				return nil, script.ErrUsage
			}
			text, err := os.ReadFile(s.Path(args[0]))
			if err != nil {
				return nil, err
			}
			data, err := hex.DecodeString(strings.Join(strings.Fields(string(text)), ""))
			if err != nil {
				return nil, err
			}
			return nil, os.WriteFile(s.Path(args[1]), data, 0o600)
		})
	cmds["mkelf"] = script.Command(
		script.CmdUsage{Summary: "write a synthetic ELF64 image with one named section; cuda-v2 has e_version 2", Args: "cuda|cuda-v2|x86 section dst"},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			machines := map[string]elf.Machine{"cuda": elf.EM_CUDA, "cuda-v2": elf.EM_CUDA, "x86": elf.EM_X86_64}
			if len(args) != 3 || machines[args[0]] == 0 {
				return nil, script.ErrUsage
			}
			b := minimalELF(machines[args[0]], args[1])
			if args[0] == "cuda-v2" {
				binary.LittleEndian.PutUint32(b[20:], 2)
			}
			return nil, os.WriteFile(s.Path(args[2]), b, 0o600)
		})
	cmds["readreport"] = script.Command(
		script.CmdUsage{Summary: "decode and validate a report", Args: "file"},
		func(s *script.State, args ...string) (script.WaitFunc, error) {
			if len(args) != 1 {
				return nil, script.ErrUsage
			}
			f, err := os.Open(s.Path(args[0]))
			if err != nil {
				return nil, err
			}
			defer f.Close()
			_, err = cudainspect.ReadReport(f)
			return nil, err
		})
	engine := &script.Engine{Cmds: cmds, Conds: scripttest.DefaultConds()}
	env := []string{"ROUNDTRIP=" + roundtrip}
	scripttest.Test(t, context.Background(), engine, env, "testdata/script/*.txt")
}

// minimalELF returns an ELF64 header for machine with a section table holding
// .shstrtab and one empty section, using CUDA ABI 7 flags for sm_121. It has
// no code, symbols or attributes.
func minimalELF(machine elf.Machine, section string) []byte {
	names := "\x00.shstrtab\x00" + section + "\x00"
	const count = 3
	b := make([]byte, 64+count*64+len(names))
	copy(b, "\x7fELF")
	b[elf.EI_CLASS], b[elf.EI_DATA], b[elf.EI_VERSION] = byte(elf.ELFCLASS64), byte(elf.ELFDATA2LSB), byte(elf.EV_CURRENT)
	b[elf.EI_OSABI], b[elf.EI_ABIVERSION] = 0x33, 7
	le := binary.LittleEndian
	le.PutUint16(b[16:], uint16(elf.ET_EXEC))
	le.PutUint16(b[18:], uint16(machine))
	le.PutUint32(b[20:], uint32(elf.EV_CURRENT))
	le.PutUint64(b[40:], 64)  // e_shoff
	le.PutUint32(b[48:], 121) // e_flags
	le.PutUint16(b[52:], 64)  // e_ehsize
	le.PutUint16(b[58:], 64)  // e_shentsize
	le.PutUint16(b[60:], count)
	le.PutUint16(b[62:], 1) // e_shstrndx
	strtab := b[64+64:]
	le.PutUint32(strtab, 1)
	le.PutUint32(strtab[4:], uint32(elf.SHT_STRTAB))
	le.PutUint64(strtab[24:], 64+count*64)
	le.PutUint64(strtab[32:], uint64(len(names)))
	sec := b[64+2*64:]
	le.PutUint32(sec, uint32(len("\x00.shstrtab\x00")))
	le.PutUint32(sec[4:], uint32(elf.SHT_PROGBITS))
	copy(b[64+count*64:], names)
	return b
}

func TestRunExitStatus(t *testing.T) {
	dir := t.TempDir()
	raw, err := os.ReadFile("../../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	image := filepath.Join(dir, "k.cubin")
	if err := os.WriteFile(image, cubin, 0o600); err != nil {
		t.Fatal(err)
	}
	sass := "../../cudasass/testdata/roundtrip/sm121a-roundtrip.sass"
	for _, tt := range []struct {
		name   string
		args   []string
		status int
		stdout bool
	}{
		{"complete", []string{"report", "-symbol", "addk", "-require", "target,opcode_histogram", "-sass", sass, image}, exitOK, true},
		{"partial", []string{"report", "-symbol", "addk", "-optional", "opcode_histogram", image}, exitOK, true},
		{"spaced_list", []string{"report", "-symbol", "addk", "-require", "target, toolchain_stamp ", image}, exitOK, true},
		{"required_missing", []string{"report", "-symbol", "addk", "-require", "opcode_histogram", image}, exitRequired, true},
		{"unknown_symbol", []string{"report", "-symbol", "add", image}, exitRefused, false},
		{"missing_symbol", []string{"report", image}, exitRefused, false},
		{"oversize", []string{"report", "-symbol", "addk", "-max-bytes", "100", image}, exitRefused, false},
		{"bad_bound", []string{"report", "-symbol", "addk", "-max-bytes", "-1", image}, exitRefused, false},
		{"bad_flag", []string{"report", "-bogus", image}, exitRefused, false},
		{"no_subcommand", []string{image}, exitRefused, false},
		{"two_files", []string{"report", "-symbol", "addk", image, image}, exitRefused, false},
		{"missing_file", []string{"report", "-symbol", "addk", filepath.Join(dir, "absent")}, exitRefused, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if got := run(tt.args, &stdout, &stderr); got != tt.status {
				t.Fatalf("status %d, want %d; stderr: %s", got, tt.status, stderr.String())
			}
			if (stdout.Len() > 0) != tt.stdout {
				t.Fatalf("stdout %q", stdout.String())
			}
			if tt.stdout {
				if _, err := cudainspect.ReadReport(&stdout); err != nil {
					t.Fatal(err)
				}
			}
			if (stderr.Len() > 0) != (tt.status != exitOK) {
				t.Fatalf("stderr %q", stderr.String())
			}
		})
	}
}
