//go:build linux && amd64

package bpfprobe

import (
	"context"
	"debug/elf"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestNativeNoFramePointerStack(t *testing.T) {
	if os.Getenv("EBPFHOST_NATIVE") != "1" {
		t.Skip("native no-frame-pointer control is opt-in")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "nofp.c")
	bin := filepath.Join(dir, "nofp")
	code := `#include <stdio.h>
__attribute__((noinline)) void observed(unsigned long value){__asm__ volatile(""::"r"(value):"memory");}
__attribute__((noinline)) void caller(void){volatile unsigned long value=1;observed(value);value++;}
int main(void){if(getchar()==EOF)return 2;caller();return 0;}
`
	if err := os.WriteFile(src, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O2", "-fcf-protection=none", "-fomit-frame-pointer", "-fno-pie", "-no-pie", "-o", bin, src).CombinedOutput(); err != nil {
		t.Fatalf("fixture: %v %s", err, out)
	}
	f, err := elf.Open(bin)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	syms, err := f.Symbols()
	if err != nil {
		t.Fatal(err)
	}
	var offset uint64
	var caller, observed elf.Symbol
	for _, s := range syms {
		if s.Name == "observed" {
			observed = s
			sec := f.Sections[s.Section]
			offset = sec.Offset + s.Value - sec.Addr
		}
		if s.Name == "caller" {
			caller = s
		}
	}
	if caller.Size == 0 || observed.Size == 0 {
		t.Fatal("fixture symbols absent")
	}
	cmd := exec.Command(bin)
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { input.Close(); cmd.Process.Kill(); cmd.Wait() }()
	p, err := Attach(Config{Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 1, UserStack: true, Capacity: 65536})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	input.Write([]byte{'x'})
	input.Close()
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if err := p.Disable(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	e, err := p.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if !e.UserStackRequested {
		t.Fatal("missing stack request provenance")
	}
	if len(e.Stack) == 0 && e.StackUnavailable == "" {
		t.Fatal("empty stack lacks unavailable reason")
	}
	for _, address := range e.Stack {
		if address >= caller.Value && address < caller.Value+caller.Size {
			t.Fatalf("no-FP control unexpectedly recovered immediate caller; update qualification scope: %x", e.Stack)
		}
	}
	t.Logf("no-frame-pointer control returns %d bounded observed addresses, unavailable=%q; immediate caller unavailable and full-stack attribution is not claimed", len(e.Stack), e.StackUnavailable)
}
