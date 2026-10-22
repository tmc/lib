//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMetalIQ2CodebooksRegistered pins the IQ2_XXS tables by content.
//
// While the two shader sources each carried their own transcription, a typo in
// one of them showed up as a disagreement between the batched kernel and the
// per-token reference it is compared against. Splicing one copy into both took
// that detector away with the duplication: a corrupted entry now corrupts the
// kernel and its reference identically and they still agree bit for bit. The
// registered digest is what replaces it, and it is the stronger check, because
// it also catches a corruption both copies would have shared.
//
// The digest covers the constant's whole text, comment included. To change the
// tables deliberately, take the new digest from the failure message and say in
// the commit which upstream revision it was copied from.
func TestMetalIQ2CodebooksRegistered(t *testing.T) {
	const registered = "a70db0f599948b67163a77232b1b0738af3ccb184f6241b646d7d9afe411758c"
	if got := sha256.Sum256([]byte(metalIQ2CodebooksMSL)); registered != hex.EncodeToString(got[:]) {
		t.Errorf("IQ2_XXS codebooks are sha256 %s, registered %s", hex.EncodeToString(got[:]), registered)
	}

	// The digest only means something if both shader sources actually take the
	// tables from this constant rather than from a copy of their own.
	for _, s := range []struct {
		name string
		src  string
	}{
		{"metalPinnedMSL", metalPinnedMSL},
		{"metalMatmulExpertPreservedMSL", metalMatmulExpertPreservedMSL},
	} {
		if !strings.Contains(s.src, metalIQ2CodebooksMSL) {
			t.Errorf("%s does not contain the shared IQ2_XXS codebooks; it has grown its own copy", s.name)
		}
		if n := strings.Count(s.src, "ds4_metal_iq2xxs_grid"+"[256]"); n != 1 {
			t.Errorf("%s declares ds4_metal_iq2xxs_grid %d times, want 1", s.name, n)
		}
	}
}

// TestMetalIQ2CodebooksAreNotDuplicated is the source half of the gate, in the
// same spirit as TestCommandBufferCreationIsCentralized: the value test above
// proves both shaders contain the table, but a second literal copy of the same
// text would satisfy it too and be free to drift on the next edit. Only a scan
// of the sources can say the tables are declared once.
func TestMetalIQ2CodebooksAreNotDuplicated(t *testing.T) {
	const home = "metal_iq2_codebook_darwin.go"

	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range []string{
		// Split so this file does not itself count as a declaration site.
		"ds4_metal_ksigns_iq2xs" + "[128] = {",
		"ds4_metal_iq2xxs_grid" + "[256] = {",
	} {
		var sites []string
		for _, f := range files {
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			if n := strings.Count(string(src), decl); n > 0 {
				sites = append(sites, f)
				if n > 1 {
					t.Errorf("%s declares %s %d times", f, decl, n)
				}
			}
		}
		if len(sites) != 1 || sites[0] != home {
			t.Errorf("%s is declared in %v, want only %s", decl, sites, home)
		}
	}
}
