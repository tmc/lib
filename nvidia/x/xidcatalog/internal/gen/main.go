// Command gen imports the pinned NVIDIA XID workbook. It verifies its digest
// before parsing or replacing output. Use -source to regenerate offline.
package main

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/xml"
	"flag"
	"fmt"
	"go/format"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const maxBytes = 16 << 20

type xidRow struct {
	id                                          uint64
	name, description, immediate, investigatory string
	products                                    uint8
}

func main() {
	source := flag.String("source", "", "local workbook instead of downloading")
	url := flag.String("url", "", "official source URL")
	digest := flag.String("sha256", "", "required workbook SHA256")
	date := flag.String("retrieved", "", "retrieval date, YYYY-MM-DD")
	out := flag.String("out", "catalog.gen.go", "output Go file")
	flag.Parse()
	if err := run(*source, *url, *digest, *date, *out); err != nil {
		fmt.Fprintln(os.Stderr, "xidcatalog:", err)
		os.Exit(1)
	}
}

func run(source, url, digest, date, out string) error {
	if len(digest) != 64 || !strings.HasPrefix(url, "https://docs.nvidia.com/") {
		return fmt.Errorf("official URL and SHA256 required")
	}
	if _, err := time.Parse("2006-01-02", date); err != nil {
		return fmt.Errorf("retrieval date: %w", err)
	}
	var r io.ReadCloser
	var err error
	if source != "" {
		r, err = os.Open(source)
	} else {
		client := http.Client{Timeout: 30 * time.Second}
		resp, e := client.Get(url)
		if e != nil {
			return fmt.Errorf("download catalog: %w", e)
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return fmt.Errorf("download catalog: HTTP %d", resp.StatusCode)
		}
		r = resp.Body
	}
	if err != nil {
		return err
	}
	defer r.Close()
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return err
	}
	if len(data) > maxBytes {
		return fmt.Errorf("workbook exceeds byte limit")
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != digest {
		return fmt.Errorf("workbook SHA256 %s differs from pin %s", got, digest)
	}
	rows, err := parse(data)
	if err != nil {
		return err
	}
	generated, err := render(rows, url, digest, date)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(out), ".xidcatalog-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err = f.Write(generated); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	if err = os.Chmod(name, 0644); err != nil {
		return err
	}
	return os.Rename(name, out)
}

func parse(data []byte) ([]xidRow, error) {
	z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("open workbook: %w", err)
	}
	files := make(map[string]*zip.File)
	for _, f := range z.File {
		if files[f.Name] != nil {
			return nil, fmt.Errorf("duplicate zip member %q", f.Name)
		}
		files[f.Name] = f
	}
	read := func(name string, v any) error {
		f := files[name]
		if f == nil {
			return fmt.Errorf("missing workbook member %q", name)
		}
		if f.UncompressedSize64 > maxBytes {
			return fmt.Errorf("member exceeds limit")
		}
		r, err := f.Open()
		if err != nil {
			return err
		}
		defer r.Close()
		b, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
		if err != nil {
			return err
		}
		if len(b) > maxBytes {
			return fmt.Errorf("member exceeds limit")
		}
		return xml.Unmarshal(b, v)
	}
	var book struct {
		Sheets []struct {
			Name string `xml:"name,attr"`
			ID   string `xml:"sheetId,attr"`
		} `xml:"sheets>sheet"`
	}
	if err := read("xl/workbook.xml", &book); err != nil {
		return nil, err
	}
	if len(book.Sheets) == 0 || book.Sheets[0].Name != "Xids" || book.Sheets[0].ID != "1" {
		return nil, fmt.Errorf("unexpected Xids sheet identity")
	}
	var shared struct {
		Strings []struct {
			Text string `xml:"t"`
			Runs []struct {
				Text string `xml:"t"`
			} `xml:"r"`
		} `xml:"si"`
	}
	if err := read("xl/sharedStrings.xml", &shared); err != nil {
		return nil, err
	}
	stringsTable := make([]string, len(shared.Strings))
	for i, s := range shared.Strings {
		stringsTable[i] = s.Text
		for _, run := range s.Runs {
			stringsTable[i] += run.Text
		}
	}
	var sheet struct {
		Rows []struct {
			Cells []struct {
				Ref   string `xml:"r,attr"`
				Type  string `xml:"t,attr"`
				Value string `xml:"v"`
			} `xml:"c"`
		} `xml:"sheetData>row"`
	}
	if err := read("xl/worksheets/sheet1.xml", &sheet); err != nil {
		return nil, err
	}
	if len(sheet.Rows) < 2 || len(sheet.Rows) > 4096 {
		return nil, fmt.Errorf("invalid row count")
	}
	var rows []xidRow
	seen := make(map[uint64]bool)
	for i, r := range sheet.Rows {
		values := make(map[string]string)
		for _, c := range r.Cells {
			col := strings.TrimRight(c.Ref, "0123456789")
			value := c.Value
			if c.Type == "s" {
				n, e := strconv.Atoi(value)
				if e != nil || n < 0 || n >= len(stringsTable) {
					return nil, fmt.Errorf("invalid shared string")
				}
				value = stringsTable[n]
			} else if c.Type != "" {
				return nil, fmt.Errorf("unsupported cell type %q", c.Type)
			}
			if _, ok := values[col]; ok {
				return nil, fmt.Errorf("duplicate column %q", col)
			}
			values[col] = strings.TrimSpace(value)
		}
		if i == 0 {
			for col, want := range map[string]string{"B": "Code", "C": "Mnemonic", "D": "Description", "E": "Applies to A100", "F": "Applies to H100", "G": "Applies to B100", "H": "Applies to GB200", "I": "Resolution Bucket (Immediate Action)", "J": "Resolution Bucket (Investigatory Action)"} {
				if strings.Join(strings.Fields(values[col]), " ") != want {
					return nil, fmt.Errorf("unexpected header %s", col)
				}
			}
			continue
		}
		if values["A"] != "XID" {
			return nil, fmt.Errorf("unexpected row %d type %q", i+1, values["A"])
		}
		id, err := strconv.ParseUint(values["B"], 10, 64)
		if err != nil || id == 0 || seen[id] || values["C"] == "" {
			return nil, fmt.Errorf("invalid or duplicate XID row %d", i+1)
		}
		seen[id] = true
		var mask uint8
		for j, col := range []string{"E", "F", "G", "H"} {
			switch values[col] {
			case "YES":
				mask |= 1 << j
			case "NO":
			default:
				return nil, fmt.Errorf("unknown applicability %q", values[col])
			}
		}
		rows = append(rows, xidRow{id, values["C"], values["D"], values["I"], values["J"], mask})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].id < rows[j].id })
	return rows, nil
}

func render(rows []xidRow, url, digest, date string) ([]byte, error) {
	var b bytes.Buffer
	fmt.Fprintln(&b, "// Code generated from the pinned NVIDIA XID Catalog. DO NOT EDIT.\n\npackage xidcatalog")
	fmt.Fprintf(&b, "\n// SourceURL identifies the imported workbook.\nconst SourceURL = %q\n", url)
	fmt.Fprintf(&b, "\n// SourceSHA256 is the version identity of the imported workbook.\nconst SourceSHA256 = %q\n", digest)
	fmt.Fprintf(&b, "\n// Retrieved records when the workbook was retrieved.\nconst Retrieved = %q\n", date)
	fmt.Fprintln(&b, "\nvar catalog = map[uint64]row{")
	for _, r := range rows {
		fmt.Fprintf(&b, "%d: {Entry{%d, %q, %q, %q, %q, %t}, %d},\n", r.id, r.id, r.name, r.description, r.immediate, r.investigatory, r.id == 154, r.products)
	}
	fmt.Fprintln(&b, "}")
	return format.Source(b.Bytes())
}
