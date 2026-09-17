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
	if os.Getenv("CUDA_TMEM_TEST") == "1" && os.Getenv("GPU_LOCK_HELD") != "1" {
		t.Fatal("CUDA_TMEM_TEST=1 requires GPU_LOCK_HELD=1")
	}
	if _, err := exec.LookPath("nvcc"); err != nil {
		t.Fatal("CUDA_TMEM_* requires nvcc: " + err.Error())
	}
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
	if out, err := exec.Command("nvcc", args...).CombinedOutput(); err != nil {
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
