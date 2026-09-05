package cudasass

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// This is only an ELF header fixture, not executable GPU code.
func testCUBIN() []byte {
	b := make([]byte, 64)
	copy(b, "\x7fELF\x02\x01\x01")
	binary.LittleEndian.PutUint16(b[16:], 2)
	binary.LittleEndian.PutUint16(b[18:], 190)
	binary.LittleEndian.PutUint32(b[20:], 1)
	binary.LittleEndian.PutUint16(b[52:], 64)
	return b
}

func fakeTool(t *testing.T, body string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX subprocess fixture")
	}
	p := filepath.Join(t.TempDir(), "fake tool")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+body+"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestDisassembler(t *testing.T) {
	dir := t.TempDir()
	p := fakeTool(t, `for arg do last="$arg"; done
test -f "$last" || exit 7
printf 'Function : kernel\n/*0000*/ EXIT;\n'
printf 'diagnostic' >&2`)
	d := Disassembler{Path: p, TempDir: dir}
	r, err := d.Disassemble(context.Background(), testCUBIN(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Path != p || string(r.Stderr) != "diagnostic" {
		t.Fatalf("output: %+v", r)
	}
	if len(r.Args) != 1 || r.Args[0] == "-" {
		t.Fatalf("args: %q", r.Args)
	}
	if _, err := os.Stat(r.Args[0]); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("scratch input survived: %v", err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatalf("scratch leaked: %v", entries)
	}
	parsed, err := ParseString(string(r.Stdout))
	if err != nil || parsed.Function("kernel") == nil {
		t.Fatalf("parse: %v, %v", parsed, err)
	}
}

func TestDisassemblerErrors(t *testing.T) {
	for _, b := range [][]byte{nil, []byte("not elf"), append([]byte(nil), testCUBIN()...)} {
		if len(b) == 64 {
			b[18] = 62
		}
		if _, err := (Disassembler{}).Disassemble(context.Background(), b, Options{}); err == nil {
			t.Fatal("accepted non-CUDA ELF")
		}
	}
	p := fakeTool(t, `printf partial; printf broken >&2; exit 4`)
	d := Disassembler{Path: p, TempDir: t.TempDir()}
	r, err := d.Disassemble(context.Background(), testCUBIN(), Options{})
	if err == nil || string(r.Stdout) != "partial" || string(r.Stderr) != "broken" {
		t.Fatalf("%+v, %v", r, err)
	}
	entries, _ := os.ReadDir(d.TempDir)
	if len(entries) != 0 {
		t.Fatal("failed command leaked input")
	}
	d.Path = filepath.Join(t.TempDir(), "missing")
	if _, err = d.Version(context.Background()); err == nil {
		t.Fatal("missing tool succeeded")
	}
}

func TestCancellation(t *testing.T) {
	p := fakeTool(t, `exec sleep 30`)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := (Disassembler{Path: p}).Version(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline: %v", err)
	}
}

func TestModes(t *testing.T) {
	for _, tt := range []struct {
		o    Options
		want []string
	}{
		{Options{}, nil},
		{Options{Mode: JSON}, []string{"--emit-json"}},
		{Options{Mode: ControlFlow}, []string{"--output-control-flow-graph"}},
		{Options{Mode: BasicBlockFlow}, []string{"--output-control-flow-graph-with-basic-blocks"}},
		{Options{Mode: RegisterLife}, []string{"--print-life-ranges"}},
		{Options{SourceLines: true, PTXLines: true}, []string{"--print-line-info", "--print-line-info-ptx"}},
		{Options{InlineLines: true}, []string{"--print-line-info-inline"}},
	} {
		got, err := tt.o.args()
		if err != nil || !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%+v = %v, %v", tt.o, got, err)
		}
	}
	for _, o := range []Options{{Mode: 99}, {Mode: JSON, SourceLines: true}, {Mode: RegisterLife, PTXLines: true}} {
		if _, err := o.args(); err == nil {
			t.Errorf("accepted %+v", o)
		}
	}
}

func TestDemangle(t *testing.T) {
	p := fakeTool(t, `test "$#" = 0 || exit 2
exec cat`)
	r, err := (Demangler{Path: p}).Demangle(context.Background(), []string{"-not-an-option", "_Z3foov"})
	if err != nil || string(r.Stdout) != "-not-an-option\n_Z3foov\n" {
		t.Fatalf("%+v, %v", r, err)
	}
	if _, err := (Demangler{Path: p}).Demangle(context.Background(), []string{"bad\nname"}); err == nil {
		t.Fatal("accepted separator")
	}
}

const jsonFixture = `[{"ELF":{},"SM":{"version":{"major":12,"minor":1}},"SchemaVersion":{"major":13,"minor":0,"revision":0},"Producer":"test fixture","Description":""},[{"function-name":"kernel","start":32,"length":48,"sass-instructions":[{"opcode":"MOV","operands":"R0,R1"},{"opcode":""},{"opcode":"EXIT","other-attributes":{"control-flow":"True"}}]}]]`

func TestParseJSON(t *testing.T) {
	d, err := ParseJSON(strings.NewReader(jsonFixture))
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Functions) != 1 || len(d.Functions[0].Instructions) != 3 || d.Functions[0].Instructions[1].Opcode != "" {
		t.Fatalf("lost gap: %+v", d)
	}
	for _, s := range []string{"null", "[]", "[{},[]]", jsonFixture + " {}", strings.Replace(jsonFixture, `"opcode":"MOV"`, `"oops":"MOV"`, 1), strings.Replace(jsonFixture, `"start":32`, `"start":-1`, 1)} {
		if _, err := ParseJSON(strings.NewReader(s)); err == nil {
			t.Errorf("accepted %s", s)
		}
	}
}

func TestNVDisasmText(t *testing.T) {
	d, err := ParseString(".section .text.kernel,\"ax\",@progbits\n/*0000*/ MOV R0, R1; /* 0x123 */\n/*0010*/ EXIT;\n")
	if err != nil {
		t.Fatal(err)
	}
	f := d.Function("kernel")
	if f == nil || f.BaseAddress != 0 || len(f.Instructions) != 2 || f.Instructions[0].Operands != "R0, R1" {
		t.Fatalf("parse: %+v", f)
	}
}

func ExampleParseJSON() {
	d, err := ParseJSON(strings.NewReader(jsonFixture))
	if err != nil {
		panic(err)
	}
	fmt.Println(d.Functions[0].Name, len(d.Functions[0].Instructions))
	// Output: kernel 3
}

// Real-tool gates are explicit opt-in: no synthesized ELF is a real cubin.
func TestInstalledTools(t *testing.T) {
	p := os.Getenv("CUDASASS_TEST_CUBIN")
	if p == "" {
		t.Skip("set CUDASASS_TEST_CUBIN to a real compiled cubin")
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	d := Disassembler{Path: os.Getenv("NVDISASM"), TempDir: t.TempDir()}
	v, err := d.Version(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("version %s: %s%s", v.Path, v.Stdout, v.Stderr)
	for _, mode := range []Mode{Text, JSON, ControlFlow, BasicBlockFlow, RegisterLife} {
		r, err := d.Disassemble(ctx, b, Options{Mode: mode})
		t.Logf("args %q stderr %s", r.Args, r.Stderr)
		if err != nil {
			t.Fatal(err)
		}
		if len(r.Stdout) == 0 {
			t.Fatal("empty disassembly")
		}
		if mode == Text {
			parsed, err := ParseString(string(r.Stdout))
			if err != nil || len(parsed.Functions) == 0 {
				t.Fatalf("text: %v %v", parsed, err)
			}
		}
		if mode == JSON {
			j, err := ParseJSON(strings.NewReader(string(r.Stdout)))
			if err != nil || len(j.Functions) == 0 {
				t.Fatalf("JSON: %v, %v", j, err)
			}
		}
	}
	demangler := Demangler{Path: os.Getenv("CUFILT")}
	v, err = demangler.Version(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("demangler %s: %s%s", v.Path, v.Stdout, v.Stderr)
	r, err := demangler.Demangle(ctx, []string{"_Z3foov"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(r.Stdout)) != "foo()" {
		t.Fatalf("demangle: %s", r.Stdout)
	}
}
