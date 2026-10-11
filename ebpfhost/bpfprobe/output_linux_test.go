//go:build linux && amd64

package bpfprobe

import (
	"context"
	"debug/elf"
	"errors"
	"github.com/tmc/lib/ebpfhost/bpfload"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

func TestNativeOutputPointer(t *testing.T) {
	if os.Getenv("EBPFHOST_NATIVE") != "1" {
		t.Skip("native allocation output capture is opt-in")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "alloc.c")
	bin := filepath.Join(dir, "alloc")
	code := `#include <stdio.h>
#include <stdint.h>
__attribute__((noinline)) int alloc(uint64_t *p,int mode,int depth,uint64_t value) {
 if (mode==1) return 0;
 if (mode==2) return 7;
 if (depth>0) { uint64_t inner=0; alloc(&inner,mode,depth-1,value); }
 *p=value; return 0;
}
int main(void) { int mode=getchar()-'0'; if(mode==0) { for(uint64_t n=0;n<100;n++){volatile uint64_t value=99; uint64_t want=n ? 0xfedcba9876540000ULL+n : 0; if(alloc((uint64_t *)&value,0,0,want)!=0 || value!=want) return 2; value=0xffffffffffffffffULL; } } else if(mode==1) alloc((uint64_t *)1,1,0,0); else if(mode==2) alloc((uint64_t *)1,2,0,0); else if(mode==3) {uint64_t out; alloc(&out,3,9,42);} else if(mode==4) {uint64_t out; alloc(&out,0,0,42);} return 0; }
`
	if err := os.WriteFile(src, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-fno-omit-frame-pointer", "-o", bin, src).CombinedOutput(); err != nil {
		t.Fatalf("fixture: %v %s", err, out)
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
		if s.Name == "alloc" && int(s.Section) < len(f.Sections) {
			section := f.Sections[s.Section]
			offset = section.Offset + s.Value - section.Addr
			found = true
			break
		}
	}
	if !found {
		t.Fatal("alloc symbol missing")
	}
	for mode := 0; mode <= 4; mode++ {
		t.Run(string(rune('0'+mode)), func(t *testing.T) {
			cmd := exec.Command(bin)
			input, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			defer func() { input.Close(); cmd.Process.Kill(); cmd.Wait() }()
			configs := []Config{{Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 1, OutputPointer: true, CorrelationTag: 123, Capacity: 65536}, {Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 2, Return: true, OutputPointer: true, CorrelationTag: 123, Capacity: 65536}}
			if mode == 4 {
				configs = configs[1:]
			}
			g, err := AttachGroup(configs)
			if err != nil {
				var log string
				var le *bpfload.LoadError
				if errors.As(err, &le) {
					log = le.Log
				}
				t.Fatalf("attach: %v %s", err, log)
			}
			defer g.Close()
			input.Write([]byte{byte('0' + mode)})
			input.Close()
			if err := cmd.Wait(); err != nil {
				t.Fatal(err)
			}
			if err := g.Disable(); err != nil {
				t.Fatal(err)
			}
			submitted, dropped, err := g.Counts()
			if err != nil || dropped != 0 {
				t.Fatalf("counts %d/%d %v", submitted, dropped, err)
			}
			wantCount := uint64(2)
			if mode == 0 {
				wantCount = 200
			}
			if mode == 3 {
				wantCount = 20
			}
			if mode == 4 {
				wantCount = 1
			}
			if submitted != wantCount {
				t.Fatalf("emitted %d want%d", submitted, wantCount)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			known := 0
			for n := uint64(0); n < submitted; n++ {
				e, err := g.Read(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !e.ThreadGenerationKnown || e.ThreadGeneration == 0 {
					t.Fatalf("generation %+v", e)
				}
				if e.Tag == 2 {
					if !e.ReturnKnown || e.ArgumentsMask != 0 {
						t.Fatalf("return metadata %+v", e)
					}
					switch mode {
					case 0:
						want := uint64(0)
						if n/2 != 0 {
							want = 0xfedcba9876540000 + n/2
						}
						if !e.OutputKnown || e.Output != want || e.OutputError != 0 {
							t.Fatalf("return %d expectedpreoverwrite%x: %+v", n, want, e)
						}
					case 1:
						if e.OutputKnown || e.OutputError != 4 {
							t.Fatalf("memory refusal: %+v", e)
						}
					case 2:
						if e.OutputKnown || e.OutputError != 0 || uint32(e.ReturnValue) != 7 {
							t.Fatalf("failure status %+v", e)
						}
					case 3:
						if e.OutputKnown {
							known++
							if e.Output != 42 {
								t.Fatalf("nested output: %+v", e)
							}
						} else if e.OutputError != 2 {
							t.Fatalf("depth overflow: %+v", e)
						}
					case 4:
						if e.OutputKnown || e.OutputError != 1 {
							t.Fatalf("missing state: %+v", e)
						}
					}
				}
			}
			failures, err := g.OutputFailures()
			if err != nil {
				t.Fatal(err)
			}
			if mode == 0 && failures != (OutputFailures{}) || mode == 1 && failures.Unreadable != 1 || mode == 2 && failures != (OutputFailures{}) || mode == 3 && (known != 8 || failures.Overflow != 4) || mode == 4 && failures.Missing != 1 {
				t.Fatalf("mode%d known%d failures%+v", mode, known, failures)
			}
			t.Logf("mode%d known-output%d failures%+v; deferred reader after target overwrote output slots", mode, known, failures)
		})
	}
}
