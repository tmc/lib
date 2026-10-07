package main

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func workbook(t *testing.T, change func(map[string][]byte)) []byte {
	t.Helper()
	files := make(map[string][]byte)
	for _, name := range []string{"workbook.xml", "sharedStrings.xml", "sheet1.xml"} {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		path := "xl/" + name
		if name == "sheet1.xml" {
			path = "xl/worksheets/" + name
		}
		files[path] = b
	}
	if change != nil {
		change(files)
	}
	var b bytes.Buffer
	z := zip.NewWriter(&b)
	for _, name := range []string{"xl/workbook.xml", "xl/sharedStrings.xml", "xl/worksheets/sheet1.xml"} {
		if data, ok := files[name]; ok {
			w, err := z.Create(name)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = w.Write(data); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestPinnedCatalogRoundTrip(t *testing.T) {
	rows, err := parse(workbook(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 172 {
		t.Fatalf("rows = %d", len(rows))
	}
	got, err := render(rows, "https://docs.nvidia.com/deploy/xid-errors/_downloads/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx", "7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5", "2026-10-06")
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("../../catalog.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("every-row regeneration differs from embedded catalog")
	}
	second, err := render(rows, "https://docs.nvidia.com/deploy/xid-errors/_downloads/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx", "7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5", "2026-10-06")
	if err != nil || !bytes.Equal(got, second) {
		t.Fatal("nondeterministic output")
	}
}

func TestRefusesMalformedWorkbook(t *testing.T) {
	for name, change := range map[string]func(map[string][]byte){
		"missing member": func(f map[string][]byte) { delete(f, "xl/sharedStrings.xml") },
		"wrong sheet": func(f map[string][]byte) {
			f["xl/workbook.xml"] = bytes.Replace(f["xl/workbook.xml"], []byte(`name="Xids"`), []byte(`name="Other"`), 1)
		},
		"wrong shared index": func(f map[string][]byte) {
			f["xl/worksheets/sheet1.xml"] = bytes.Replace(f["xl/worksheets/sheet1.xml"], []byte(`<v>0</v>`), []byte(`<v>999999</v>`), 1)
		},
		"unknown applicability": func(f map[string][]byte) {
			f["xl/sharedStrings.xml"] = bytes.Replace(f["xl/sharedStrings.xml"], []byte(`<t>YES</t>`), []byte(`<t>MAYBE</t>`), 1)
		},
		"duplicate ID": func(f map[string][]byte) {
			f["xl/worksheets/sheet1.xml"] = bytes.Replace(f["xl/worksheets/sheet1.xml"], []byte(`<c r="B3" s="5"><v>2</v></c>`), []byte(`<c r="B3" s="5"><v>1</v></c>`), 1)
		},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := parse(workbook(t, change)); err == nil {
				t.Fatal("accepted malformed catalog")
			}
		})
	}
	for _, data := range [][]byte{nil, []byte("not zip"), workbook(t, nil)[:100]} {
		if _, err := parse(data); err == nil {
			t.Fatal("accepted truncated/non-zip input")
		}
	}
}

func TestDigestRefusalPreservesOutput(t *testing.T) {
	dir := t.TempDir()
	source, out := filepath.Join(dir, "source.xlsx"), filepath.Join(dir, "catalog.go")
	if err := os.WriteFile(source, workbook(t, nil), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, []byte("existing output"), 0600); err != nil {
		t.Fatal(err)
	}
	err := run(source, "https://docs.nvidia.com/catalog.xlsx", "0000000000000000000000000000000000000000000000000000000000000000", "2026-10-06", out)
	if err == nil {
		t.Fatal("accepted wrong digest")
	}
	got, e := os.ReadFile(out)
	if e != nil || string(got) != "existing output" {
		t.Fatal("changed output on digest refusal")
	}
}
