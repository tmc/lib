//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
)

func TestMetalDSparkHeadIdentifier(t *testing.T) {
	forbidden := regexp.MustCompile(`\buint\s+thread\b|\bif\s*\(\s*thread\b`)
	if forbidden.MatchString(metalDSparkAttentionMSL) {
		t.Fatal("reserved Metal thread identifier in proposal head")
	}
	if strings.Count(metalDSparkAttentionMSL, "uint tid [[thread_position_in_grid]]") != 1 ||
		strings.Count(metalDSparkAttentionMSL, "if(tid!=0)return;") != 1 {
		t.Fatal("proposal head thread selector missing or duplicated")
	}
	mutant := strings.ReplaceAll(metalDSparkAttentionMSL, "uint tid [[thread_position_in_grid]]", "uint thread [[thread_position_in_grid]]")
	mutant = strings.ReplaceAll(mutant, "if(tid!=0)", "if(thread!=0)")
	if !forbidden.MatchString(mutant) {
		t.Fatal("reserved identifier regression not detected")
	}
}

// Source-reviewed opt-in compilation only: six libraries, zero dispatches.
// This cannot qualify an operator or the support transformer numerically.
func TestMetalDSparkCompilePacket(t *testing.T) {
	if os.Getenv("DS4_METAL_DSPARK_COMPILE_GATE") != "1" {
		t.Skip("opt-in native compile packet")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	dir := os.Getenv("DS4_METAL_DSPARK_PACKET_DIR")
	if dir == "" {
		t.Fatal("sealed library directory required before device creation")
	}
	var sources []struct{ name, text string }
	for _, tc := range []struct{ name, source, from, to string }{
		{"dense", metalDSparkDenseMSL, "output[index]=sum;", "output[index]=sum+1.0f;"},
		{"features", metalDSparkFeatureMSL, "features[slot*4096+d]=mean;", "features[slot*4096+d]=mean+1.0f;"},
		{"attention-head", metalDSparkAttentionMSL, "previous=best;", "previous=args[3];"},
	} {
		if strings.Count(tc.source, tc.from) != 1 {
			t.Fatalf("%s mutation site not unique", tc.name)
		}
		for _, mutant := range []bool{false, true} {
			source := tc.source
			if mutant {
				source = strings.Replace(source, tc.from, tc.to, 1)
			}
			name := tc.name + "-base.metal"
			if mutant {
				name = tc.name + "-mutant.metal"
			}
			frozen, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil || string(frozen) != source {
				t.Fatalf("frozen source %s mismatch: %v", name, err)
			}
			sources = append(sources, struct{ name, text string }{name, string(frozen)})
		}
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Errorf("device close: %v", err)
		}
	}()
	for i, source := range sources {
		library, err := dev.CompileLibrary(source.text)
		if err != nil {
			t.Fatalf("library %d: %v", i, err)
		}
		if library == nil || library.GetID() == 0 {
			t.Fatalf("library %d missing", i)
		}
		objectivec.ObjectFromID(library.GetID()).Release()
		t.Logf("COMPILE_LIBRARY name=%s sha256=%x released=true", source.name, sha256.Sum256([]byte(source.text)))
	}
}
