//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

const (
	wantIQ2GridSHA256  = "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54"
	wantIQ2SignsSHA256 = "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31"
)

// TestMetalExpertTableDigests pins the byte-level representation of the Metal
// IQ2_XXS codebook tables against the registered SHA-256 digests shared across
// both Metal and CUDA backends.
func TestMetalExpertTableDigests(t *testing.T) {
	reSigns := regexp.MustCompile(`ksigns_iq2xs\[128\]\s*=\s*\{([^}]+)\};`)
	reGrid := regexp.MustCompile(`iq2xxs_grid\[256\]\s*=\s*\{([^}]+)\};`)

	// 1. Signs table: 128 bytes
	mSigns := reSigns.FindStringSubmatch(metalIQ2CodebooksMSL)
	if mSigns == nil {
		t.Fatal("ds4_metal_ksigns_iq2xs table literal not found in metalIQ2CodebooksMSL")
	}
	signTokens := strings.Split(mSigns[1], ",")
	signs := make([]byte, 0, 128)
	for _, tok := range signTokens {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		val, err := strconv.ParseUint(tok, 0, 8)
		if err != nil {
			t.Fatalf("invalid sign byte %q: %v", tok, err)
		}
		signs = append(signs, byte(val))
	}
	if len(signs) != 128 {
		t.Fatalf("expected 128 signs entries, got %d", len(signs))
	}
	gotSignsHash := sha256.Sum256(signs)
	if hex.EncodeToString(gotSignsHash[:]) != wantIQ2SignsSHA256 {
		t.Fatalf("Metal signs digest mismatch: got %s, want %s", hex.EncodeToString(gotSignsHash[:]), wantIQ2SignsSHA256)
	}

	// 2. Grid table: 256 uint64 entries (2048 bytes little-endian)
	mGrid := reGrid.FindStringSubmatch(metalIQ2CodebooksMSL)
	if mGrid == nil {
		t.Fatal("ds4_metal_iq2xxs_grid table literal not found in metalIQ2CodebooksMSL")
	}
	gridTokens := strings.Split(mGrid[1], ",")
	gridBytes := make([]byte, 0, 256*8)
	buf := make([]byte, 8)
	var count int
	for _, tok := range gridTokens {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		val, err := strconv.ParseUint(tok, 0, 64)
		if err != nil {
			t.Fatalf("invalid grid entry %q: %v", tok, err)
		}
		binary.LittleEndian.PutUint64(buf, val)
		gridBytes = append(gridBytes, buf...)
		count++
	}
	if count != 256 {
		t.Fatalf("expected 256 grid entries, got %d", count)
	}
	gotGridHash := sha256.Sum256(gridBytes)
	if hex.EncodeToString(gotGridHash[:]) != wantIQ2GridSHA256 {
		t.Fatalf("Metal grid digest mismatch: got %s, want %s", hex.EncodeToString(gotGridHash[:]), wantIQ2GridSHA256)
	}
}
