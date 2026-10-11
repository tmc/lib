//go:build linux && amd64

package bpfprobe

import (
	"context"
	"debug/elf"
	"fmt"
	"github.com/tmc/lib/ebpfhost/bpfload"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestNativeUprobe(t *testing.T) {
	if os.Getenv("EBPFHOST_NATIVE") != "1" {
		t.Skip("set EBPFHOST_NATIVE=1 for actual kernel uprobe attachment")
	}
	array, err := bpfload.CreateMap(bpfload.MapSpec{Type: 2, KeySize: 4, ValueSize: 8, MaxEntries: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer array.Close()
	arrayFile, err := array.File()
	if err != nil {
		t.Fatal(err)
	}
	defer arrayFile.Close()
	if r, err := OpenRing(int(arrayFile.Fd()), 4096); err == nil {
		r.Close()
		t.Fatal("ring accepted array map descriptor")
	}
	ringMap, err := bpfload.CreateRingMap(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer ringMap.Close()
	ringFile, err := ringMap.File()
	if err != nil {
		t.Fatal(err)
	}
	defer ringFile.Close()
	if r, err := OpenRing(int(ringFile.Fd()), 8192); err == nil {
		r.Close()
		t.Fatal("ring accepted capacity contradiction")
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "probe.c")
	bin := filepath.Join(dir, "probe")
	source := `#include <stdio.h>
#include <stddef.h>
#include <asm/ptrace.h>
_Static_assert(offsetof(struct pt_regs,rdi)==112,"rdi offset");
_Static_assert(offsetof(struct pt_regs,rsi)==104,"rsi offset");
_Static_assert(offsetof(struct pt_regs,rdx)==96,"rdx offset");
_Static_assert(offsetof(struct pt_regs,rcx)==88,"rcx offset");
_Static_assert(offsetof(struct pt_regs,r8)==72,"r8 offset");
_Static_assert(offsetof(struct pt_regs,r9)==64,"r9 offset");
_Static_assert(offsetof(struct pt_regs,rax)==80,"rax offset");
_Static_assert(offsetof(struct pt_regs,rsp)==152,"rsp offset");
_Static_assert(offsetof(struct pt_regs,rip)==128,"rip offset");
__attribute__((noinline)) unsigned long observed(unsigned long a,unsigned long b,unsigned long c,unsigned long d,unsigned long e,unsigned long f,unsigned long g,unsigned long h,unsigned long i,unsigned long j,unsigned long k) { return a+b+c+d+e+f+g+h+i+j+k; }
int main(void) { if (getchar()==EOF) return 2; for(unsigned long n=0;n<100;n++) if(observed(n,2,3,4,5,6,7,8,9,10,11)!=n+65) return 3; return 0; }
`
	if err := os.WriteFile(src, []byte(source), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-O0", "-fno-omit-frame-pointer", "-o", bin, src).CombinedOutput(); err != nil {
		t.Fatalf("compile fixture: %v: %s", err, out)
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
	for _, returning := range []bool{false, true} {
		name := "entry"
		if returning {
			name = "return"
		}
		t.Run(name, func(t *testing.T) {
			cmd := exec.Command(bin)
			input, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			stat, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", cmd.Process.Pid))
			if err != nil {
				t.Fatal(err)
			}
			end := strings.LastIndexByte(string(stat), ')')
			fields := strings.Fields(string(stat)[end+1:])
			if end < 0 || len(fields) < 20 {
				t.Fatal("malformed task stat")
			}
			ticks, err := strconv.ParseUint(fields[19], 10, 64)
			if err != nil {
				t.Fatal(err)
			}
			hzRaw, err := exec.Command("getconf", "CLK_TCK").Output()
			if err != nil {
				t.Fatal(err)
			}
			hz, err := strconv.ParseUint(strings.TrimSpace(string(hzRaw)), 10, 64)
			if err != nil || hz == 0 {
				t.Fatalf("clock tick frequency %q: %v", hzRaw, err)
			}
			defer func() { input.Close(); cmd.Process.Kill(); cmd.Wait() }()
			p, err := Attach(Config{Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 0xfedcba9876543210, Return: returning, Capacity: 65536})
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			if _, err := input.Write([]byte{'x'}); err != nil {
				t.Fatal(err)
			}
			input.Close()
			if err := cmd.Wait(); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var prior uint64
			for n := 0; n < 100; n++ {
				e, err := p.Read(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !e.ThreadGenerationKnown || e.ThreadGeneration == 0 {
					t.Fatalf("unknown task generation: %+v", e)
				}
				if e.ThreadGeneration/(1000000000/hz) != ticks {
					t.Fatalf("kernel birth time %d disagrees with independent proc ticks %d at %dHz", e.ThreadGeneration, ticks, hz)
				}
				if e.PIDTGID>>32 != uint64(cmd.Process.Pid) || uint32(e.PIDTGID) != uint32(cmd.Process.Pid) || e.Tag != 0xfedcba9876543210 || e.Timestamp == 0 || e.Timestamp < prior {
					t.Fatalf("identity/clock: %+v", e)
				}
				prior = e.Timestamp
				if returning {
					if e.ArgumentsMask != 0 || e.ReturnValue != uint64(n+65) {
						t.Fatalf("return %d: %+v", n, e)
					}
				} else {
					if e.ArgumentsMask != 0x7ff {
						t.Fatalf("mask %x", e.ArgumentsMask)
					}
					for i, v := range e.Arguments {
						want := uint64(i + 1)
						if i == 0 {
							want = uint64(n)
						}
						if v != want {
							t.Fatalf("event %d argument %d=%d want %d", n, i, v, want)
						}
					}
				}
			}
			submitted, dropped, err := p.Counts()
			if err != nil || submitted != 100 || dropped != 0 {
				t.Fatalf("counts %d/%d: %v", submitted, dropped, err)
			}
			t.Logf("actual kernel %s uprobe: 100/100 known-answer observations, 0 drops; eleven SysV slots/return and PID/tag/monotonic checks passed", name)
		})
	}
	t.Run("group-order-and-overflow", func(t *testing.T) {
		for _, capacity := range []int{65536, 4096} {
			cmd := exec.Command(bin)
			input, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			g, err := AttachGroup([]Config{{Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 1, Capacity: capacity}, {Path: bin, Offset: offset, PID: cmd.Process.Pid, Tag: 2, Return: true, Capacity: capacity}})
			if err != nil {
				input.Close()
				cmd.Process.Kill()
				cmd.Wait()
				t.Fatal(err)
			}
			input.Write([]byte{'x'})
			input.Close()
			err = cmd.Wait()
			if err != nil {
				g.Close()
				t.Fatal(err)
			}
			if err := g.Disable(); err != nil {
				g.Close()
				t.Fatal(err)
			}
			submitted, dropped, err := g.Counts()
			if err != nil || submitted+dropped != 200 {
				g.Close()
				t.Fatalf("group counts %d/%d: %v", submitted, dropped, err)
			}
			if capacity == 65536 && (submitted != 200 || dropped != 0) {
				g.Close()
				t.Fatalf("group no-pressure counts %d/%d", submitted, dropped)
			}
			if capacity == 4096 && (submitted == 0 || dropped == 0) {
				g.Close()
				t.Fatalf("overflow control did not lose records: %d/%d", submitted, dropped)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			for n := uint64(0); n < submitted; n++ {
				e, err := g.Read(ctx)
				if err != nil {
					cancel()
					g.Close()
					t.Fatal(err)
				}
				if capacity == 65536 {
					if e.Tag != 1+n%2 {
						cancel()
						g.Close()
						t.Fatalf("group event %d tag %d", n, e.Tag)
					}
				}
			}
			cancel()
			if err := g.Close(); err != nil {
				t.Fatal(err)
			}
			t.Logf("shared ring capacity=%d emitted=%d refused=%d", capacity, submitted, dropped)
		}
	})

}
