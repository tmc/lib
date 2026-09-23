package cudabpfdev

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
)

func FuzzPTXTranslation(f *testing.F) {
	f.Add([]byte{byte(Exit), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 || len(data)%12 != 0 || len(data) > 64*12 {
			return
		}
		p := program()
		for off := 0; off < len(data); off += 12 {
			x := data[off : off+12]
			p.Code = append(p.Code, Instruction{Op: Op(x[0]), Dst: x[1], A: x[2], B: x[3], Imm: binary.LittleEndian.Uint32(x[4:]), Target: int(int32(binary.LittleEndian.Uint32(x[8:])))})
		}
		x, err := Translate(p, budget(), "sm_121a")
		if err != nil {
			return
		}
		if _, err := cudaptx.ParseBytes(x.Artifact.Raw); err != nil {
			t.Fatalf("unparseable PTX: %v\n%s", err, x.Artifact.Raw)
		}
		if err := x.Artifact.Validate(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestTranslationDeterministicAndParsed(t *testing.T) {
	p := program(Instruction{Op: Lane, Dst: 1}, Instruction{Op: Leader, Dst: 2, A: 1}, Instruction{Op: Ballot, Dst: 3, A: 1}, Instruction{Op: MapAdd, A: 2, B: 3}, Instruction{Op: Repeat, Target: 3, Imm: 2}, Instruction{Op: Exit, A: 2})
	for _, target := range []string{"sm_100a", "sm_121a"} {
		a, err := Translate(p, budget(), target)
		if err != nil {
			t.Fatal(err)
		}
		b, err := Translate(p, budget(), target)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a.Artifact.Raw, b.Artifact.Raw) || a.Artifact.Digest != b.Artifact.Digest {
			t.Fatal("nondeterministic translation")
		}
		if _, err := cudaptx.ParseBytes(a.Artifact.Raw); err != nil {
			t.Fatalf("PTX parser: %v\n%s", err, a.Artifact.Raw)
		}
		if len(a.Artifact.Schema.Params) != 4 || a.Artifact.Schema.TotalParamBytes != 32 || a.DeclaredRegisters != 28 {
			t.Fatalf("translation=%+v schema=%+v", a, a.Artifact.Schema)
		}
		text := string(a.Artifact.Raw)
		for _, want := range []string{".reqntid 32, 1, 1", "vote.sync.ballot.b32", "shfl.sync.idx.b32", "@write_map atom.global.add.u32", "@write_map atom.global.add.u64", "setp.lt.u32 in_bounds, r2, 4", "mov.u32 loop4, 0"} {
			if !strings.Contains(text, want) {
				t.Errorf("missing %q", want)
			}
		}
	}
}

func TestTranslationIdentityCannotBeRebound(t *testing.T) {
	for _, change := range []func(*Translation){
		func(t *Translation) { t.Artifact.Target = "sm_100a" },
		func(t *Translation) { t.Artifact.EntryPoint = "other" },
		func(t *Translation) { t.Artifact.Schema.Params[0].Name = "different" },
		func(t *Translation) { *t = Translation{Artifact: t.Artifact} },
	} {
		x, err := Translate(program(Instruction{Op: Exit}), budget(), "sm_121a")
		if err != nil {
			t.Fatal(err)
		}
		change(&x)
		called := false
		_, err = assemble(x, func([]nvjitlinkx.Input, ...string) (*nvjitlinkx.Result, error) { called = true; return nil, nil })
		if err == nil || called {
			t.Fatal("changed translation reached linker")
		}
	}
}

// This compiles only; it never loads a module or executes a device kernel.
func TestPTXAssembler(t *testing.T) {
	tool, err := cudatool.LookTool("ptxas", "")
	if err != nil {
		t.Skipf("assembler qualification unavailable: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	v, err := cudatool.Run(ctx, tool, "", nil, "--version")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tool=%s version=%s%s", v.Path, v.Stdout, v.Stderr)
	p := program(Instruction{Op: Const, Dst: 1, Imm: 1}, Instruction{Op: Lane, Dst: 2}, Instruction{Op: Ballot, Dst: 3, A: 2}, Instruction{Op: Leader, Dst: 2, A: 2}, Instruction{Op: Move, Dst: 3, A: 1}, Instruction{Op: And, Dst: 3, A: 3, B: 1}, Instruction{Op: Add, Dst: 3, A: 3, B: 1}, Instruction{Op: JumpNonzero, A: 1, Target: 8}, Instruction{Op: MapAdd, B: 3}, Instruction{Op: Repeat, Target: 8, Imm: 2}, Instruction{Op: Exit, A: 2})
	for _, target := range []string{"sm_100a", "sm_121a"} {
		t.Run(target, func(t *testing.T) {
			x, err := Translate(p, budget(), target)
			if err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			src := filepath.Join(dir, "program.ptx")
			dst := filepath.Join(dir, "program.cubin")
			if err := os.WriteFile(src, x.Artifact.Raw, 0600); err != nil {
				t.Fatal(err)
			}
			out, err := cudatool.Run(ctx, tool, "", nil, "-arch="+target, "-v", src, "-o", dst)
			t.Logf("args=%v stdout=%s stderr=%s", out.Args, out.Stdout, out.Stderr)
			if err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(dst)
			if err != nil || len(data) == 0 {
				t.Fatalf("missing CUBIN: %v", err)
			}
		})
	}
}

func TestPTXAssembledThroughLinker(t *testing.T) {
	x, err := Translate(program(Instruction{Op: Exit}), budget(), "sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	wantErr := errors.New("fixture link failure")
	want := &nvjitlinkx.Result{ErrorLog: "fixture diagnostic"}
	called := false
	got, err := assemble(x, func(inputs []nvjitlinkx.Input, options ...string) (*nvjitlinkx.Result, error) {
		called = true
		if len(inputs) != 1 || inputs[0].Kind != nvjitlink.InputPtx || !bytes.Equal(inputs[0].Data, x.Artifact.Raw) {
			t.Fatalf("inputs=%+v", inputs)
		}
		if len(options) != 1 || options[0] != "-arch=sm_121a" {
			t.Fatalf("options=%v", options)
		}
		return want, wantErr
	})
	if !called || got != want || !errors.Is(err, wantErr) {
		t.Fatalf("result=%v err=%v", got, err)
	}
	x.Artifact.Raw[0] = '!'
	called = false
	_, err = assemble(x, func([]nvjitlinkx.Input, ...string) (*nvjitlinkx.Result, error) { called = true; return nil, nil })
	if err == nil || called {
		t.Fatal("corrupted artifact reached linker")
	}
}

func TestTranslationRefusesUnsupportedResources(t *testing.T) {
	p := program(Instruction{Op: Exit})
	b := budget()
	b.MaxRegisters = 4
	if _, err := Translate(p, b, "sm_121a"); err == nil {
		t.Fatal("scratch registers omitted from budget")
	}
	if _, err := Translate(p, budget(), "sm_999a"); err == nil {
		t.Fatal("accepted unknown target")
	}
	p.StagingBytes = 128
	if _, err := Translate(p, budget(), "sm_121a"); err == nil {
		t.Fatal("silently ignored shared staging")
	}
	p.StagingBytes = 110000
	b = budget()
	b.MaxSharedBytes = 240000
	if _, err := Translate(p, b, "sm_121a"); err == nil || !strings.Contains(err.Error(), "target limit") {
		t.Fatalf("sm121 staging gate=%v", err)
	}
	if _, err := Translate(p, b, "sm_100a"); err == nil || !strings.Contains(err.Error(), "not emitted") {
		t.Fatalf("sm100 staging gate=%v", err)
	}
}
