package cudaregress

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func TestCommandOutputFixture(t *testing.T) {
	mode := os.Args[len(os.Args)-1]
	switch mode {
	case "fixture-small":
		fmt.Print("hello")
		os.Exit(0)
	case "fixture-large":
		for i := 0; i < 17; i++ {
			fmt.Print(strings.Repeat("x", 1<<20))
		}
		os.Exit(0)
	case "fixture-wait":
		time.Sleep(3 * time.Second)
		os.Exit(0)
	}
}

func TestSafeCommandOutputBounds(t *testing.T) {
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	digest, size, err := ComputeFileDigest(binary)
	if err != nil {
		t.Fatal(err)
	}
	p, _, _, _ := historyFixture(t)
	p.BinaryArtifacts = []ArtifactRef{{Name: "fixture", Path: binary, Digest: digest, SizeBytes: size}}
	p.PlanDigest, err = ComputePlanDigest(p)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"fixture-small", "fixture-large", "fixture-wait"} {
		t.Run(mode, func(t *testing.T) {
			arm, err := SafeCommandArm(p, "fixture", binary, "-test.run=^TestCommandOutputFixture$", "--", mode)
			if err != nil {
				t.Fatal(err)
			}
			ctx := context.Background()
			if mode == "fixture-wait" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, 20*time.Millisecond)
				defer cancel()
			}
			r, err := arm.Run(ctx)
			if mode == "fixture-small" {
				if err != nil {
					t.Fatal(err)
				}
				want := fmt.Sprintf("%x", sha256.Sum256([]byte("hello")))
				if r.Digest != want {
					t.Fatalf("digest %s want %s", r.Digest, want)
				}
			} else if err == nil || r.Complete {
				t.Fatalf("unbounded command accepted: %+v %v", r, err)
			}
		})
	}
	arm, err := SafeCommandArm(p, "fixture", binary)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = arm.Run(nil); err == nil {
		t.Fatal("nil context accepted")
	}
}
