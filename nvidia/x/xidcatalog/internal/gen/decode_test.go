package main

import (
	"bytes"
	"os"
	"testing"
)

func TestDecodePinnedRoundTrip(t *testing.T) {
	rules, cells, err := parseDecode(workbook(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(rules) != 95 || len(cells) != 106 {
		t.Fatalf("populations %d/%d", len(rules), len(cells))
	}
	got, err := renderDecode(rules, cells)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("../../decode.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("decode regeneration differs")
	}
	again, err := renderDecode(rules, cells)
	if err != nil || !bytes.Equal(got, again) {
		t.Fatal("nondeterministic decode")
	}
}

func TestDecodeMalformedSource(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(map[string][]byte)
	}{
		{"missing_sheet", func(f map[string][]byte) { delete(f, "xl/worksheets/sheet2.xml") }},
		{"coordinate", func(f map[string][]byte) {
			f["xl/worksheets/sheet2.xml"] = bytes.Replace(f["xl/worksheets/sheet2.xml"], []byte(`r="C23"`), []byte(`r="D23"`), 1)
		}},
		{"row_number", func(f map[string][]byte) {
			f["xl/worksheets/sheet2.xml"] = bytes.Replace(f["xl/worksheets/sheet2.xml"], []byte(`r="23"`), []byte(`r="230"`), 1)
		}},
		{"sheet_identity", func(f map[string][]byte) {
			f["xl/workbook.xml"] = bytes.Replace(f["xl/workbook.xml"], []byte(`name="Xid 144-150 Decode"`), []byte(`name="Other Decode"`), 1)
		}},
		{"guide_schema", func(f map[string][]byte) { delete(f, "xl/worksheets/sheet4.xml") }},
		{"primary_mask", func(f map[string][]byte) {
			f["xl/sharedStrings.xml"] = bytes.Replace(f["xl/sharedStrings.xml"], []byte("------000000----------0000100001"), []byte("-----000000----------0000100001"), 1)
		}},
		{"secondary_mask", func(f map[string][]byte) {
			f["xl/sharedStrings.xml"] = bytes.Replace(f["xl/sharedStrings.xml"], []byte("------000000---------10011000010"), []byte("-----000000---------10011000010"), 1)
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := parseDecode(workbook(t, test.change)); err == nil {
				t.Fatal("malformed decode source accepted")
			}
		})
	}
}

func FuzzDecodeWorkbook(f *testing.F) {
	f.Add([]byte("not zip"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > maxBytes {
			return
		}
		_, _, _ = parseDecode(b)
	})
}
