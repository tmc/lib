//go:build linux && amd64

package bpfprobe

import (
	"context"
	"debug/elf"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNativeUserStacks(t *testing.T) {
	if os.Getenv("EBPFHOST_NATIVE") != "1" {
		t.Skip("native user-stack check is opt-in")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "stack.c")
	bin := filepath.Join(dir, "stack")
	var code strings.Builder
	code.WriteString("#include <stdio.h>\n__attribute__((noinline)) void observed(unsigned long n){__asm__ volatile(\"\"::\"r\"(n):\"memory\");}\n")
	for n := 0; n < 1200; n++ {
		fmt.Fprintf(&code, "__attribute__((noinline)) void caller%d(void){observed(%d);__asm__ volatile(\"\"::: \"memory\");}\n", n, n)
	}
	code.WriteString("int main(void){if(getchar()==EOF)return 2;")
	for n := 0; n < 1200; n++ {
		fmt.Fprintf(&code, "caller%d();", n)
	}
	code.WriteString("return 0;}\n")
	if err := os.WriteFile(src, []byte(code.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-fcf-protection=none", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie", "-o", bin, src).CombinedOutput(); err != nil {
		t.Fatalf("stack fixture: %v %s", err, out)
	}
	f, err := elf.Open(bin)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	symbols, err := f.Symbols()
	if err != nil {
		t.Fatal(err)
	}
	var offset uint64
	var observed elf.Symbol
	callers := make(map[uint64]elf.Symbol)
	for _, s := range symbols {
		if s.Name == "observed" {
			observed = s
			section := f.Sections[s.Section]
			offset = section.Offset + s.Value - section.Addr
		}
		if strings.HasPrefix(s.Name, "caller") {
			var n uint64
			if _, err := fmt.Sscanf(s.Name, "caller%d", &n); err == nil {
				callers[n] = s
			}
		}
	}
	if observed.Size == 0 || len(callers) != 1200 {
		t.Fatal("fixture symbols missing")
	}
	// Attach after a verified frame-pointer prologue. At the symbol entry the
	// callee frame has not been established and the immediate caller may be
	// absent from a frame-pointer walk; we do not advertise full stack recovery.
	image, err := os.Open(bin)
	if err != nil {
		t.Fatal(err)
	}
	defer image.Close()
	bytes := make([]byte, 4)
	if _, err := image.ReadAt(bytes, int64(offset)); err != nil {
		t.Fatal(err)
	}
	if string(bytes) != string([]byte{0x55, 0x48, 0x89, 0xe5}) {
		t.Fatalf("unsupported compiled prologue %x", bytes)
	}
	offset += 4
	cmd := exec.Command(bin)
	input, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { input.Close(); cmd.Process.Kill(); cmd.Wait() }()
	p, err := Attach(Config{Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 1, UserStack: true, Capacity: 1 << 20})
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
	submitted, dropped, err := p.Counts()
	if err != nil || submitted != 1200 || dropped != 0 {
		t.Fatalf("counts %d/%d %v", submitted, dropped, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	captured, refused := 0, 0
	for n := uint64(0); n < submitted; n++ {
		e, err := p.Read(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !e.UserStackRequested || e.Arguments[0] != n {
			t.Fatalf("stack identity: %+v", e)
		}
		if e.StackID < 0 {
			refused++
			if e.StackUnavailable == "" || len(e.Stack) != 0 {
				t.Fatalf("stack refusal hidden: %+v", e)
			}
			continue
		}
		if e.StackUnavailable != "" || len(e.Stack) == 0 || len(e.Stack) > 8 {
			t.Fatalf("stack lookup: %+v", e)
		}
		foundCaller := false
		target := callers[n]
		for _, address := range e.Stack {
			if address >= target.Value && address < target.Value+target.Size {
				foundCaller = true
			}
		}
		if !foundCaller {
			t.Fatalf("callsite %d absent in stack %x; expected %#x..%#x", n, e.Stack, target.Value, target.Value+target.Size)
		}
		captured++
	}
	failures, err := p.OutputFailures()
	if err != nil || captured == 0 || refused == 0 || uint64(refused) != failures.StackFailure {
		t.Fatalf("capture%d refusal%d failures%+v: %v", captured, refused, failures, err)
	}
	t.Logf("actual user stacks captured%d refused%d; independently compiled caller bounds match; stable map IDs never reused and kernel refusals remain explicit", captured, refused)
}
