package cudahotswap

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type generationOwner struct {
	mu     sync.Mutex
	closed int
	fail   bool
}

func (o *generationOwner) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.fail {
		return errors.New("injected unload failure")
	}
	o.closed++
	return nil
}

func (o *generationOwner) count() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.closed
}

func testGeneration(t *testing.T, raw driver.Function, param string) (*generation, *generationOwner) {
	t.Helper()
	ptx := fmt.Sprintf(".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry(%s) { ret; }", param)
	sig, err := cudalaunch.SignatureFromPTX(ptx, "entry")
	if err != nil {
		t.Fatal(err)
	}
	o := new(generationOwner)
	return &generation{digest: fmt.Sprintf("source-%d", raw), function: raw, signature: sig, owner: o}, o
}

func TestGenerationVersionsAndRevert(t *testing.T) {
	var table generationTable
	a, aOwner := testGeneration(t, 1, ".param .u64 arg")
	b, bOwner := testGeneration(t, 2, ".param .u64 renamed")
	one, err := table.publish("kernel", a, 0)
	if err != nil {
		t.Fatal(err)
	}
	two, err := table.publish("kernel", b, one.version)
	if err != nil {
		t.Fatal(err)
	}
	three, err := table.selectVersion("kernel", one.version)
	if err != nil || three.version != one.version || !(one.revision < two.revision && two.revision < three.revision) {
		t.Fatalf("select = %+v/%v", three, err)
	}
	if _, err := table.selectVersion("kernel", two.version); err != nil {
		t.Fatal(err)
	}
	reverted, err := table.revert("kernel")
	if err != nil || reverted.version != one.version || reverted.revision <= three.revision {
		t.Fatalf("revert = %+v/%v", reverted, err)
	}
	pin, _, err := table.pin("kernel")
	if err != nil || pin.gen.function != 1 {
		t.Fatalf("revert did not restore original function: %v", err)
	}
	if err := pin.release(); err != nil {
		t.Fatal(err)
	}
	if aOwner.count()+bOwner.count() != 0 {
		t.Fatal("retained versions unloaded")
	}
	if err := table.close(); err != nil || aOwner.count() != 1 || bOwner.count() != 1 {
		t.Fatalf("close = %v, counts=%d/%d", err, aOwner.count(), bOwner.count())
	}
	if err := table.close(); err != nil || aOwner.count() != 1 || bOwner.count() != 1 {
		t.Fatal("close repeated an unload")
	}
}

func TestGenerationPinnedRetirementAndRetry(t *testing.T) {
	var table generationTable
	a, _ := testGeneration(t, 1, "")
	first, err := table.publish("kernel", a, 0)
	if err != nil {
		t.Fatal(err)
	}
	b, owner := testGeneration(t, 2, "")
	second, err := table.publish("kernel", b, first.version)
	if err != nil {
		t.Fatal(err)
	}
	pin, _, err := table.pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := table.revert("kernel"); err != nil {
		t.Fatal(err)
	}
	if err := table.drop("kernel", second.version); err != nil || owner.count() != 0 {
		t.Fatal("drop unloaded a pinned version")
	}
	owner.fail = true
	if err := pin.release(); err == nil || b.owner == nil {
		t.Fatal("failed unload lost retry owner")
	}
	owner.fail = false
	if err := table.drop("kernel", second.version); err != nil || owner.count() != 1 {
		t.Fatalf("retry = %v, unloads=%d", err, owner.count())
	}
	if _, err := table.selectVersion("kernel", second.version); err == nil {
		t.Fatal("selected an unloaded version")
	}
	if err := pin.release(); err != nil || owner.count() != 1 {
		t.Fatal("repeat release unloaded twice")
	}
	if err := table.close(); err != nil {
		t.Fatal(err)
	}
}

func TestGenerationABIAndStalePublish(t *testing.T) {
	for _, param := range []string{".param .u32 arg", ".param .b8 bytes[9]", ".param .u64 a, .param .u64 b"} {
		var table generationTable
		a, _ := testGeneration(t, 1, ".param .u64 arg")
		first, err := table.publish("kernel", a, 0)
		if err != nil {
			t.Fatal(err)
		}
		bad, owner := testGeneration(t, 2, param)
		if _, err := table.publish("kernel", bad, first.version); err == nil || bad.version != 0 || owner.count() != 0 {
			t.Fatal("mismatched ABI accepted or failed candidate ownership consumed")
		}
		good, _ := testGeneration(t, 3, ".param .u64 arg")
		if _, err := table.publish("kernel", good, first.version); err != nil {
			t.Fatal(err)
		}
		stale, staleOwner := testGeneration(t, 4, ".param .u64 arg")
		if _, err := table.publish("kernel", stale, first.version); err == nil || stale.version != 0 || staleOwner.count() != 0 {
			t.Fatal("stale candidate accepted or retry owner consumed")
		}
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		if err := staleOwner.Close(); err != nil {
			t.Fatal(err)
		}
		if err := table.close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestGenerationCloseWhilePinned(t *testing.T) {
	var table generationTable
	a, owner := testGeneration(t, 1, "")
	if _, err := table.publish("kernel", a, 0); err != nil {
		t.Fatal(err)
	}
	pin, _, err := table.pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	if err := table.close(); !errors.Is(err, errGenerationPinned) || owner.count() != 0 {
		t.Fatal("close abandoned a pinned generation")
	}
	if _, _, err := table.pin("kernel"); err == nil {
		t.Fatal("pin after close")
	}
	if err := pin.release(); err != nil || owner.count() != 1 {
		t.Fatalf("completion release = %v", err)
	}
	if err := table.close(); err != nil || owner.count() != 1 {
		t.Fatal("repeat close changed module lifetime")
	}
}

func TestGenerationConcurrentSelection(t *testing.T) {
	var table generationTable
	a, aOwner := testGeneration(t, 1, "")
	one, err := table.publish("kernel", a, 0)
	if err != nil {
		t.Fatal(err)
	}
	b, bOwner := testGeneration(t, 2, "")
	two, err := table.publish("kernel", b, one.version)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Go(func() {
			for j := 0; j < 200; j++ {
				pin, _, err := table.pin("kernel")
				if err != nil {
					t.Error(err)
					return
				}
				if pin.gen.function != 1 && pin.gen.function != 2 || aOwner.count()+bOwner.count() != 0 {
					t.Error("pin observed invalid or unloaded module")
				}
				if err := pin.release(); err != nil {
					t.Error(err)
				}
			}
		})
	}
	for i := 0; i < 200; i++ {
		version := one.version
		if i%2 != 0 {
			version = two.version
		}
		if _, err := table.selectVersion("kernel", version); err != nil {
			t.Fatal(err)
		}
	}
	wg.Wait()
	if err := table.close(); err != nil {
		t.Fatal(err)
	}
}

func TestGenerationCopiesCompilerMetadata(t *testing.T) {
	var table generationTable
	g, _ := testGeneration(t, 1, ".param .u64 arg")
	params := g.signature.Params
	flags := []string{"--std=c++17"}
	g.flags = flags
	if _, err := table.publish("kernel", g, 0); err != nil {
		t.Fatal(err)
	}
	params[0].Size = 4
	flags[0] = "--use_fast_math"
	pin, _, err := table.pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	if pin.gen.signature.Params[0].Size != 8 || pin.gen.flags[0] != "--std=c++17" {
		t.Fatal("caller mutated published compiler metadata")
	}
	if err := pin.release(); err != nil {
		t.Fatal(err)
	}
	if err := table.close(); err != nil {
		t.Fatal(err)
	}
}

func FuzzGenerationSelection(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3, 0, 1})
	f.Fuzz(func(t *testing.T, operations []byte) {
		if len(operations) > 32 {
			t.Skip()
		}
		var table generationTable
		first, owner := testGeneration(t, 1, ".param .u64 arg")
		stamp, err := table.publish("kernel", first, 0)
		if err != nil {
			t.Fatal(err)
		}
		versions := []uint64{stamp.version}
		owners := []*generationOwner{owner}
		for i, op := range operations {
			before := stamp
			switch op % 4 {
			case 0:
				o := new(generationOwner)
				g := &generation{digest: fmt.Sprintf("source-%d", i), signature: first.signature, function: driver.Function(i + 2), owner: o}
				stamp, err = table.publish("kernel", g, stamp.version)
				versions = append(versions, stamp.version)
				owners = append(owners, o)
			case 1:
				stamp, err = table.selectVersion("kernel", versions[int(op)%len(versions)])
			case 2:
				stamp, err = table.revert("kernel")
			case 3:
				pin, current, pinErr := table.pin("kernel")
				if pinErr != nil || current != stamp {
					t.Fatalf("pin = %+v/%v, want %+v", current, pinErr, stamp)
				}
				err = pin.release()
			}
			if err != nil {
				t.Fatal(err)
			}
			if op%4 != 3 && stamp.revision != before.revision+1 || op%4 == 3 && stamp != before {
				t.Fatalf("mutation counter changed incorrectly: %+v -> %+v", before, stamp)
			}
		}
		if err := table.close(); err != nil {
			t.Fatal(err)
		}
		for _, o := range owners {
			if o.count() != 1 {
				t.Fatalf("unload count = %d, want 1", o.count())
			}
		}
	})
}
