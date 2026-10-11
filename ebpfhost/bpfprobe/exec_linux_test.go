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

func TestNativeExecIdentity(t *testing.T) {
	if os.Getenv("EBPFHOST_NATIVE") != "1" {
		t.Skip("native exec identity check is opt-in")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "exec.c")
	bin := filepath.Join(dir, "exec")
	code := `#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
static const char *binary;
__attribute__((noinline)) int observed(uint64_t *out,int restart) { if(restart){execl(binary,binary,"again",NULL);return 2;} *out=0xfedcba9876543210ULL; return 0; }
int main(int argc,char **argv) { binary=argv[0]; if(argc==1 && getchar()==EOF) return 3; volatile uint64_t out=0; int status=observed((uint64_t *)&out,argc==1); out=0; return status; }
`
	if err := os.WriteFile(src, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-o", bin, src).CombinedOutput(); err != nil {
		t.Fatalf("fixture compile: %v %s", err, out)
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
	found := false
	for _, s := range symbols {
		if s.Name == "observed" && int(s.Section) < len(f.Sections) {
			section := f.Sections[s.Section]
			offset = section.Offset + s.Value - section.Addr
			found = true
			break
		}
	}
	if !found {
		t.Fatal("fixture symbol missing")
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
	g, err := AttachGroup([]Config{{Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 1, OutputPointer: true, CorrelationTag: 7, Capacity: 65536}, {Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 2, Return: true, OutputPointer: true, CorrelationTag: 7, Capacity: 65536}})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	input.Write([]byte{'x'})
	input.Close()
	if err := cmd.Wait(); err != nil {
		t.Fatal(err)
	}
	if err := g.Disable(); err != nil {
		t.Fatal(err)
	}
	submitted, dropped, err := g.Counts()
	if err != nil || submitted != 3 || dropped != 0 {
		t.Fatalf("counts %d/%d %v", submitted, dropped, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var events [3]Event
	for i := range events {
		events[i], err = g.Read(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if !events[i].ExecIDKnown || !events[i].ThreadGenerationKnown {
			t.Fatalf("unknown identity: %+v", events[i])
		}
	}
	if events[0].Tag != 1 || events[1].Tag != 1 || events[2].Tag != 2 || events[0].PIDTGID != events[1].PIDTGID || events[0].ThreadGeneration != events[1].ThreadGeneration || events[1].ExecID != events[0].ExecID+1 || events[2].ExecID != events[1].ExecID || !events[2].OutputKnown || events[2].Output != 0xfedcba9876543210 {
		t.Fatalf("exec identities/output: %+v", events)
	}
	failures, err := g.OutputFailures()
	if err != nil || failures != (OutputFailures{}) {
		t.Fatalf("state failures: %+v %v", failures, err)
	}
	t.Logf("same binary/process/birth across exec: exec ID %d -> %d; stale allocation frame remains isolated, new output exact", events[0].ExecID, events[1].ExecID)
}
