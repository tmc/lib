package cudatmem

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

func TestTMEMAllocationUnits(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	x, err := a.Allocate(p, 32, 32, 1)
	if err != nil || x.Columns() != 32 || x.Offset() != 0 {
		t.Fatalf("allocation: %v %+v", err, x)
	}
	_ = a.Release(p, x, 32)
	_ = a.Relinquish(p)
	_ = a.Close()
}

func TestTMEMForgedHandleAndOverflow(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	x, _ := a.Allocate(p, 32, 512, 1)
	forged := *x
	if !errors.Is(a.Release(p, &forged, 32), ErrClosed) {
		t.Fatal("forged allocation handle accepted")
	}
	if _, err := a.Allocate(p, 32, 32, 1); !errors.Is(err, ErrInvalidUnit) {
		t.Fatalf("column exhaustion: %v", err)
	}
	_ = a.Release(p, x, 32)
	_ = a.Relinquish(p)
	_ = a.Close()
}

func TestTMEMFixtureInstructions(t *testing.T) {
	f, err := GenerateFixture(SM100A)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateFixture(f); err != nil {
		t.Fatal(err)
	}
	if _, err := GenerateFixture(SM121A); !errors.Is(err, ErrUnsupportedTarget) {
		t.Fatalf("unsupported fixture: %v", err)
	}
}

func TestTMEMFixtureMutants(t *testing.T) {
	for _, kind := range []string{"omit-store", "perturb-load"} {
		s, err := GenerateMutantCUDA(kind)
		if err != nil {
			t.Fatal(err)
		}
		if kind == "omit-store" && strings.Contains(s, "tcgen05.st.sync.aligned") {
			t.Fatal("omit-store mutant retained store")
		}
		if kind == "perturb-load" && !strings.Contains(s, "a += 1") {
			t.Fatal("perturb-load mutant did not perturb output")
		}
	}
}
func TestTMEMSliceLifetime(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	x, _ := a.Allocate(p, 32, 32, 1)
	if err := x.BeginUser(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(a.Release(p, x, 32), ErrBusy) {
		t.Fatal("premature release accepted")
	}
	if err := x.CompleteUser(); err != nil {
		t.Fatal(err)
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatal(err)
	}
	_ = a.Relinquish(p)
	_ = a.Close()
}
func TestTMEMRejectSM121(t *testing.T) {
	if _, err := New(SM121A); !errors.Is(err, ErrUnsupportedTarget) {
		t.Fatalf("sm121: %v", err)
	}
}

func TestTMEMGB300AndFamilyTargets(t *testing.T) {
	for _, target := range []Target{SM100A, SM103A, SM107A, SM100F, SM103F, SM107F} {
		parsed, err := ParseTarget(string(target))
		if err != nil || parsed != target {
			t.Fatalf("ParseTarget(%q) = %q, %v; want %q, nil", target, parsed, err, target)
		}
		a, err := New(target)
		if err != nil {
			t.Fatalf("New(%q): %v", target, err)
		}
		cap := a.Capability()
		if cap.Target != target || cap.MaxColumns != 512 || cap.MaxRows != 128 || cap.ColumnUnit != 32 || cap.RowUnit != 1 {
			t.Fatalf("Capability(%q) = %+v", target, cap)
		}
		p, err := a.AcquirePermit(32)
		if err != nil {
			t.Fatalf("AcquirePermit(%q): %v", target, err)
		}
		x, err := a.Allocate(p, 32, 128, 128)
		if err != nil || x.Columns() != 128 || x.Offset() != 0 {
			t.Fatalf("Allocate(%q) = %+v, %v", target, x, err)
		}
		if err := a.Release(p, x, 32); err != nil {
			t.Fatalf("Release(%q): %v", target, err)
		}
		if err := a.Relinquish(p); err != nil {
			t.Fatalf("Relinquish(%q): %v", target, err)
		}
		if err := a.Close(); err != nil {
			t.Fatalf("Close(%q): %v", target, err)
		}
	}
}
func TestTMEMCollectiveParticipation(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	if _, err := a.Allocate(p, 31, 32, 1); !errors.Is(err, ErrParticipation) {
		t.Fatalf("participation: %v", err)
	}
}
func TestTMEMPermitSequence(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	if err := a.Relinquish(p); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Allocate(p, 32, 32, 1); !errors.Is(err, ErrPermission) {
		t.Fatalf("released permit: %v", err)
	}
}
func TestTMEMAccessLayout(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	if _, err := a.Allocate(p, 32, 31, 1); !errors.Is(err, ErrInvalidUnit) {
		t.Fatalf("unit: %v", err)
	}
}

func TestTMEMPermitOwnershipAndReuse(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	cp := *p
	if a.Relinquish(&cp) == nil {
		t.Fatal("accepted copied permit")
	}
	x, _ := a.Allocate(p, 32, 512, 1)
	if err := a.Release(p, x, 32); err != nil {
		t.Fatal(err)
	}
	y, e := a.Allocate(p, 32, 512, 1)
	if e != nil || y.Offset() != 0 {
		t.Fatalf("free-list reuse=%v %v", y, e)
	}
}
func TestTMEMUserIdentity(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	x, _ := a.Allocate(p, 32, 32, 1)
	cp := *x
	if cp.BeginUser() == nil {
		t.Fatal("forged allocation accepted")
	}
	_ = a.Release(p, x, 32)
	_ = a.Relinquish(p)
}

func TestTMEMExclusive576Columns(t *testing.T) {
	if _, err := NewExclusiveMap(SM90A, CTAGroup2); !errors.Is(err, ErrUnsupportedTarget) {
		t.Fatalf("NewExclusiveMap(SM90A, CTAGroup2) = %v, want ErrUnsupportedTarget", err)
	}
	if _, err := NewExclusiveMap(SM107F, CTAGroup(3)); !errors.Is(err, ErrParticipation) {
		t.Fatalf("NewExclusiveMap(SM107F, CTAGroup(3)) = %v, want ErrParticipation", err)
	}

	// 1. Allocate 512 standard columns + 64 exclusive tail columns on SM107F CTAGroup2.
	m, err := NewExclusiveMap(SM107F, CTAGroup2)
	if err != nil {
		t.Fatalf("NewExclusiveMap(SM107F, CTAGroup2): %v", err)
	}
	if !m.Exclusive() || m.CTAGroup() != CTAGroup2 || m.Capability().MaxColumns != MaxColumnsExclusive {
		t.Fatalf("unexpected exclusive map capability: exclusive=%v group=%d cap=%+v",
			m.Exclusive(), m.CTAGroup(), m.Capability())
	}
	p, err := m.AcquirePermit(32)
	if err != nil {
		t.Fatalf("AcquirePermit: %v", err)
	}
	x512, err := m.Allocate(p, 32, 512, 128)
	if err != nil || x512.Columns() != 512 || x512.Offset() != 0 {
		t.Fatalf("Allocate(512) = %+v, %v", x512, err)
	}
	x64, err := m.Allocate(p, 32, 64, 128)
	if err != nil || x64.Columns() != 64 || x64.Offset() != 512 {
		t.Fatalf("Allocate(64) = %+v, %v", x64, err)
	}
	if _, err := m.Allocate(p, 32, 32, 128); !errors.Is(err, ErrInvalidUnit) {
		t.Fatalf("expected exhaustion after 576 columns, got %v", err)
	}
	if err := m.Release(p, x64, 32); err != nil {
		t.Fatalf("Release(64): %v", err)
	}
	if err := m.Release(p, x512, 32); err != nil {
		t.Fatalf("Release(512): %v", err)
	}
	if err := m.Relinquish(p); err != nil {
		t.Fatalf("Relinquish: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// 2. Exclusive capacity is 576 columns on sm_107 and 512 on sm_100/sm_103,
	// with either CTA group, and need not be a power of two (PTX ISA 9.4 Table 58).
	tests := []struct {
		target Target
		group  CTAGroup
		max    uint32
	}{
		{SM100A, CTAGroup1, 512},
		{SM103A, CTAGroup2, 512},
		{SM107A, CTAGroup1, MaxColumnsExclusive},
		{SM107F, CTAGroup2, MaxColumnsExclusive},
	}
	for _, tt := range tests {
		m2, err := NewExclusiveMap(tt.target, tt.group)
		if err != nil {
			t.Fatalf("NewExclusiveMap(%s, %d): %v", tt.target, tt.group, err)
		}
		if got := m2.Capability().MaxColumns; got != tt.max {
			t.Fatalf("NewExclusiveMap(%s).MaxColumns = %d, want %d", tt.target, got, tt.max)
		}
		p2, err := m2.AcquirePermit(32)
		if err != nil {
			t.Fatalf("AcquirePermit(%s): %v", tt.target, err)
		}
		if _, err := m2.Allocate(p2, 32, tt.max+32, 128); !errors.Is(err, ErrInvalidUnit) {
			t.Fatalf("Allocate(%s, %d) = %v, want ErrInvalidUnit", tt.target, tt.max+32, err)
		}
		x, err := m2.Allocate(p2, 32, tt.max, 128)
		if err != nil || x.Columns() != tt.max || x.Offset() != 0 {
			t.Fatalf("Allocate(%s, %d) = %+v, %v", tt.target, tt.max, x, err)
		}
		if err := m2.Release(p2, x, 32); err != nil {
			t.Fatalf("Release(%s): %v", tt.target, err)
		}
		y, err := m2.Allocate(p2, 32, 96, 128) // multiple of 32, not a power of two
		if err != nil || y.Columns() != 96 {
			t.Fatalf("Allocate(%s, 96) = %+v, %v", tt.target, y, err)
		}
		if err := m2.Release(p2, y, 32); err != nil {
			t.Fatalf("Release(%s, 96): %v", tt.target, err)
		}
		if err := m2.Relinquish(p2); err != nil {
			t.Fatalf("Relinquish(%s): %v", tt.target, err)
		}
		if err := m2.Close(); err != nil {
			t.Fatalf("Close(%s): %v", tt.target, err)
		}
	}
}

func FuzzTMEMAllocationBounds(f *testing.F) {
	f.Add(uint32(32), uint32(1), uint32(1))
	f.Add(uint32(0), uint32(0), uint32(0))
	f.Fuzz(func(t *testing.T, columns, rows, participants uint32) {
		a, err := New(SM100A)
		if err != nil {
			t.Fatal(err)
		}
		p, err := a.AcquirePermit(participants)
		if err != nil {
			return
		}
		_, _ = a.Allocate(p, participants, columns, rows)
	})
}
func TestNativeTMEMReference(t *testing.T) {
	f, err := GenerateFixture(SM100A)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateFixture(f); err != nil {
		t.Fatal(err)
	}
	if os.Getenv("CUDA_TMEM_TEST") != "1" && os.Getenv("CUDA_TMEM_COMPILE") != "1" {
		t.Skip("native fixture requires CUDA_TMEM_COMPILE=1 or CUDA_TMEM_TEST=1")
	}
	// See cudashared/native_test.go: the lock exports the holder's pid, not "1".
	if os.Getenv("CUDA_TMEM_TEST") == "1" && os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDA_TMEM_TEST=1 requires running under the GPU lock")
	}
	nvcc, _ := cudatest.NVCC(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "fixture.cu")
	if err := os.WriteFile(src, []byte(f.CUDA), 0600); err != nil {
		t.Fatal(err)
	}
	outPath := filepath.Join(dir, "fixture.out")
	args := []string{"-arch=sm_100a", src, "-o", outPath}
	if os.Getenv("CUDA_TMEM_TEST") != "1" {
		args = []string{"-arch=sm_100a", "-cubin", src, "-o", outPath}
	}
	if out, err := exec.Command(nvcc, args...).CombinedOutput(); err != nil {
		t.Fatalf("tcgen05 fixture compile: %v\n%s", err, out)
	}
	if os.Getenv("CUDA_TMEM_TEST") != "1" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, outPath).CombinedOutput(); err != nil {
		t.Fatalf("tcgen05 fixture run: %v\n%s", err, out)
	}
}
