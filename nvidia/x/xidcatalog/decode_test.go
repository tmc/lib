package xidcatalog

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func literalDecodeRows(t *testing.T) []struct {
	Row   int
	Cells map[string]string
} {
	t.Helper()
	b, err := os.ReadFile("testdata/decode-cells.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		SourceSHA256 string `json:"source_sha256"`
		Rows         []struct {
			Row   int
			Cells map[string]string
		}
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.SourceSHA256 != SourceSHA256 {
		t.Fatal("source identity differs")
	}
	return doc.Rows[1:]
}

func TestDecodeLiteralCells(t *testing.T) {
	rows := literalDecodeRows(t)
	if len(rows) != 95 {
		t.Fatal(len(rows))
	}
	counts := make(map[uint64]int)
	status, secondary := 0, 0
	for i, want := range rows {
		got := decodeRules[i]
		code, _ := strconv.ParseUint(want.Cells[fmt.Sprint("A", want.Row)], 10, 64)
		if got.Code != code || got.Row != want.Row {
			t.Fatalf("row identity %d", want.Row)
		}
		for column := 0; column < 13; column++ {
			key := fmt.Sprintf("%c%d", 'A'+column, want.Row)
			if got.Cells[column] != want.Cells[key] {
				t.Fatalf("cell %s differs", key)
			}
		}
		counts[code]++
		if got.Cells[4] != "" {
			status++
		}
		if got.Cells[6] != "" {
			secondary++
			if len(got.Cells[6]) != 32 || len(got.Cells[7]) != 32 {
				t.Fatal("secondary width")
			}
		}
	}
	for i, want := range []int{6, 33, 20, 2, 1, 31, 2} {
		if counts[uint64(144+i)] != want {
			t.Fatal(counts)
		}
	}
	if status != 73 || secondary != 3 {
		t.Fatalf("populations %d/%d", status, secondary)
	}
	// These cells are literal source oracles, independent of generated matches.
	if decodeRules[21].Cells[2] != "------000000---------00011000010" || decodeRules[21].Cells[6] != "------000000---------10011000010" || decodeRules[21].Cells[8] != "RESET_GPU" {
		t.Fatal("row23 primary/secondary source")
	}
	if len(DecodeSources()) != 106 {
		t.Fatal("source cells lost")
	}
	b, err := os.ReadFile("testdata/decode-sources.json")
	if err != nil {
		t.Fatal(err)
	}
	var source []SourceCell
	if err := json.Unmarshal(b, &source); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(DecodeSources(), source) {
		t.Fatal("literal supporting cells differ")
	}
}

func TestDecodeEverySourceRow(t *testing.T) {
	for _, raw := range literalDecodeRows(t) {
		id, _ := strconv.ParseUint(raw.Cells[fmt.Sprint("A", raw.Row)], 10, 64)
		status := uint32(0)
		text := raw.Cells[fmt.Sprint("E", raw.Row)]
		if text != "" {
			n, err := strconv.ParseUint(strings.Split(text, "/")[0][2:], 16, 32)
			if err != nil {
				t.Fatal(err)
			}
			status = uint32(n)
		}
		for _, version := range []struct{ driver, column string }{{"574.64.01", "C"}, {"575.0", "D"}} {
			for _, variant := range []struct {
				offset byte
				kind   DecodeVariant
				bucket string
			}{{0, PrimaryDecode, "F"}, {4, SecondaryDecode, "I"}} {
				column := string(version.column[0] + variant.offset)
				mask := raw.Cells[column+strconv.Itoa(raw.Row)]
				if mask == "" {
					continue
				}
				bits, err := strconv.ParseUint(strings.ReplaceAll(mask, "-", "0"), 2, 32)
				if err != nil {
					t.Fatal(err)
				}
				for _, value := range []uint32{uint32(bits), uint32(bits) | wildcardBits(mask)} {
					result := Decode(id, B100, version.driver, value, status)
					found := false
					for _, match := range result.Matches {
						if match.Rule.Row == raw.Row && match.Variant == variant.kind {
							found = true
							if match.Bucket != raw.Cells[variant.bucket+strconv.Itoa(raw.Row)] || match.StatusConstrained != (text != "") {
								t.Fatal("source interpretation differs")
							}
						}
					}
					if !found {
						t.Fatalf("source row %d variant%d driver%s unmatched", raw.Row, variant.kind, version.driver)
					}
					if result.Outcome != OneDecodeMatch && result.Outcome != AmbiguousDecode {
						t.Fatal("match outcome missing")
					}
				}
			}
		}
	}
}

func wildcardBits(mask string) uint32 {
	var bits uint32
	for _, c := range mask {
		bits <<= 1
		if c == '-' {
			bits |= 1
		}
	}
	return bits
}

func TestDecodeBoundaries(t *testing.T) {
	for _, test := range []struct {
		name, driver string
		info, status uint32
		row          int
		variant      DecodeVariant
		outcome      DecodeOutcome
	}{
		{"older", "574.64.0", 0x21, 1, 2, PrimaryDecode, OneDecodeMatch},
		{"boundary", "575.0", 1, 1, 2, PrimaryDecode, OneDecodeMatch},
		{"unknown", "", 1, 1, 0, PrimaryDecode, UnknownDriver},
		{"malformed", "R575", 1, 1, 0, PrimaryDecode, UnknownDriver},
		{"secondary_v1", "574.64.0", 0x4c2, 1, 23, SecondaryDecode, OneDecodeMatch},
		{"secondary_v2", "575.0", 0x86, 1, 23, SecondaryDecode, OneDecodeMatch},
		{"wrong_status", "575.0", 1, 0xffffffff, 0, PrimaryDecode, NoDecodeMatch},
	} {
		t.Run(test.name, func(t *testing.T) {
			r := Decode(144, B100, test.driver, test.info, test.status)
			if test.variant == SecondaryDecode {
				r = Decode(145, B100, test.driver, test.info, test.status)
			}
			if r.Outcome != test.outcome {
				t.Fatalf("outcome %+v", r)
			}
			if test.row != 0 && (len(r.Matches) != 1 || r.Matches[0].Rule.Row != test.row || r.Matches[0].Variant != test.variant) {
				t.Fatalf("match %+v", r)
			}
		})
	}
	if got := Decode(146, B100, "574.0", 0x143, 1); got.Outcome != AmbiguousDecode || len(got.Matches) < 2 {
		t.Fatalf("ambiguity lost: %+v", got)
	}
	if got := Decode(144, H100, "575.0", 1, 1); got.Outcome != UnsupportedProduct || got.Applicability != NotApplicable || len(got.Matches) != 0 {
		t.Fatal("covered but inapplicable product decoded", got)
	}
	for _, driver := range []string{" ", "0", "575..1", "575.1.1.1", "-575", "4294967296", "575.a", "575.1\n", strings.Repeat("1", 129)} {
		if got := Decode(144, B100, driver, 1, 1); got.Outcome != UnknownDriver || len(got.Matches) != 0 {
			t.Fatalf("invalid driver %q", driver)
		}
	}
	for _, product := range []Architecture{"L4", "GB10", "Hopper", ""} {
		if got := Decode(144, product, "575.0", 1, 1); got.Outcome != UnsupportedProduct || got.Applicability != UncoveredProduct {
			t.Fatal(got)
		}
	}
	if !matchesStatus("0x00000000/0xFFFFFFFF", 0) || !matchesStatus("0x00000000/0xFFFFFFFF", 0xffffffff) || matchesStatus("0x00000000/0xFFFFFFFF", 1) || !matchesStatus("", 123) {
		t.Fatal("status alternatives/blank")
	}
}

func TestProductCoverage(t *testing.T) {
	for _, test := range []struct {
		id      uint64
		product Architecture
		want    Applicability
	}{{154, H100, Applicable}, {1, H100, NotApplicable}, {154, "L4", UncoveredProduct}, {154, "GB10", UncoveredProduct}, {999, H100, UnknownID}, {999, "L4", UnknownID}} {
		entry, state := LookupProduct(test.id, test.product)
		if state != test.want {
			t.Fatalf("%+v: %v", test, state)
		}
		_, old := Lookup(test.id, test.product)
		if old != (state == Applicable) {
			t.Fatal("old behavior changed")
		}
		_, known := LookupRaw(test.id)
		if known != (state != UnknownID) || known && entry.Code != test.id {
			t.Fatal("raw lookup differs")
		}
	}
	if len(catalog) != 172 {
		t.Fatal("original population changed")
	}
	for id := range catalog {
		for _, p := range []Architecture{A100, H100, B100, GB200} {
			_, state := LookupProduct(id, p)
			_, old := Lookup(id, p)
			if old != (state == Applicable) {
				t.Fatal("old row changed")
			}
		}
	}
}

func TestDecodeDetachedConcurrent(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rows := DecodeRules(145)
			rows[0].Cells[5] = "changed"
			cells := DecodeSources()
			cells[0].Value = "changed"
			_ = Decode(145, B100, "575.0", 0x86, 1)
		}()
	}
	wg.Wait()
	if DecodeRules(145)[0].Cells[5] == "changed" || DecodeSources()[0].Value == "changed" {
		t.Fatal("data aliases")
	}
}
