package ncclinspector

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"regexp"
	"testing"
)

// reviewedProducerPatch is the SHA256 of the independently reviewed producer
// patch from which the pinned aarch64 Inspector plugin was built.
const reviewedProducerPatch = "c825c908d737876e370fbdbc0edef6ee6270370388ec34b6dc0aa3016233563e"

func TestProducerPatchPin(t *testing.T) {
	patch, err := os.ReadFile("testdata/inspector-full-comm-hash.patch")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(patch)
	got := hex.EncodeToString(sum[:])
	readme, err := os.ReadFile("testdata/README.md")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile("(?m)^Patch SHA256: `([0-9a-f]{64})`").FindAllSubmatch(readme, -1)
	if len(m) != 1 {
		t.Fatalf("README states %d patch SHA256 lines, want 1", len(m))
	}
	if stated := string(m[0][1]); got != stated || got != reviewedProducerPatch {
		t.Fatalf("patch SHA256 %s, README %s, reviewed %s", got, stated, reviewedProducerPatch)
	}
}
