package main

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"go/format"
	"io"
	"strconv"
	"strings"
)

type decodeRow struct {
	code  uint64
	row   int
	cells [13]string
}

type sourceCell struct{ sheet, coordinate, value string }

type sourceRow struct {
	Number int `xml:"r,attr"`
	Cells  []struct {
		Ref   string `xml:"r,attr"`
		Type  string `xml:"t,attr"`
		Value string `xml:"v"`
	} `xml:"c"`
}

func parseDecode(data []byte) ([]decodeRow, []sourceCell, error) {
	z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, nil, err
	}
	files := make(map[string]*zip.File)
	for _, f := range z.File {
		if files[f.Name] != nil {
			return nil, nil, fmt.Errorf("duplicate member %q", f.Name)
		}
		files[f.Name] = f
	}
	read := func(name string, value any) error {
		f := files[name]
		if f == nil || f.UncompressedSize64 > maxBytes {
			return fmt.Errorf("missing or oversized member %q", name)
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
		return xml.Unmarshal(b, value)
	}
	var book struct {
		Sheets []struct {
			Name string `xml:"name,attr"`
			ID   string `xml:"sheetId,attr"`
		} `xml:"sheets>sheet"`
	}
	if err := read("xl/workbook.xml", &book); err != nil {
		return nil, nil, err
	}
	names := []string{"Xids", "Xid 144-150 Decode", "Resolution Buckets", "Customer User Guide"}
	if len(book.Sheets) != len(names) {
		return nil, nil, fmt.Errorf("unexpected workbook sheets")
	}
	for i, s := range book.Sheets {
		if s.Name != names[i] || s.ID != strconv.Itoa(i+1) {
			return nil, nil, fmt.Errorf("unexpected sheet identity")
		}
	}
	var shared struct {
		Items []struct {
			Text string `xml:"t"`
			Runs []struct {
				Text string `xml:"t"`
			} `xml:"r"`
		} `xml:"si"`
	}
	if err := read("xl/sharedStrings.xml", &shared); err != nil {
		return nil, nil, err
	}
	table := make([]string, len(shared.Items))
	for i, item := range shared.Items {
		table[i] = item.Text
		for _, run := range item.Runs {
			table[i] += run.Text
		}
	}
	rows := func(sheet int, columns int) ([][13]string, error) {
		var doc struct {
			Rows []sourceRow `xml:"sheetData>row"`
		}
		if err := read(fmt.Sprintf("xl/worksheets/sheet%d.xml", sheet), &doc); err != nil {
			return nil, err
		}
		if len(doc.Rows) == 0 || len(doc.Rows) > 4096 {
			return nil, fmt.Errorf("invalid sheet population")
		}
		var result [][13]string
		for i, row := range doc.Rows {
			if row.Number != i+1 {
				return nil, fmt.Errorf("noncontiguous source row")
			}
			var cells [13]string
			seen := make(map[int]bool)
			for _, c := range row.Cells {
				if len(c.Ref) < 2 {
					return nil, fmt.Errorf("invalid cell coordinate")
				}
				col := int(c.Ref[0] - 'A')
				if col < 0 || col >= columns || c.Ref != string(rune('A'+col))+strconv.Itoa(row.Number) || seen[col] {
					return nil, fmt.Errorf("invalid or duplicate coordinate %q", c.Ref)
				}
				seen[col] = true
				v := c.Value
				switch c.Type {
				case "s":
					n, e := strconv.Atoi(v)
					if e != nil || n < 0 || n >= len(table) {
						return nil, fmt.Errorf("invalid shared string")
					}
					v = table[n]
				case "":
				default:
					return nil, fmt.Errorf("unsupported cell type")
				}
				cells[col] = v
			}
			result = append(result, cells)
		}
		return result, nil
	}
	d, err := rows(2, 13)
	if err != nil {
		return nil, nil, err
	}
	if len(d) != 96 {
		return nil, nil, fmt.Errorf("decode population differs")
	}
	header := []string{"Xid", "Subcode V1(<R575)/V2(>=R575) V1(<R575): IntrInfo[9:5] V2(>=R575):IntrInfo[6:0]", "(V1(<R575)) IntrInfo decode for Data Center Recovery Action IntrInfo (binary; \"-\" user decode)", "(V2(>=R575)) IntrInfo decode for Data Center Recovery Action IntrInfo (binary; \"-\" user decode)", "Error Status (hex)", "Resolution Bucket (Data Center Recovery Action)", "(V1(<R575)) Decode for action 2", "(V2(>=R575)) Decode for action 2", "Action 2", "Resolution Bucket (Investigatory Action)", "Severity (for items with '*' please see Customer User Guide tab)", "HW/SW", "Local/Remote (for items with '*' please see Customer User Guide tab)"}
	for i, want := range header {
		if strings.Join(strings.Fields(d[0][i]), " ") != want {
			return nil, nil, fmt.Errorf("decode header %c differs", 'A'+i)
		}
	}
	var rules []decodeRow
	var evidence []sourceCell
	counts := make(map[uint64]int)
	secondary, status := 0, 0
	for i, v := range d {
		if i == 0 {
			for j, text := range v {
				evidence = append(evidence, sourceCell{names[1], fmt.Sprintf("%c1", 'A'+j), text})
			}
			continue
		}
		code, e := strconv.ParseUint(v[0], 10, 64)
		if e != nil || code < 144 || code > 150 || v[1] == "" || v[5] == "" {
			return nil, nil, fmt.Errorf("invalid decode row %d", i+1)
		}
		for _, column := range []int{2, 3, 6, 7} {
			pattern := v[column]
			if pattern == "" && column >= 6 {
				continue
			}
			if len(pattern) != 32 || strings.Trim(pattern, "01-") != "" {
				return nil, nil, fmt.Errorf("invalid mask at row %d column %c", i+1, 'A'+column)
			}
		}
		if v[4] != "" {
			status++
			for _, part := range strings.Split(v[4], "/") {
				if !strings.HasPrefix(part, "0x") {
					return nil, nil, fmt.Errorf("invalid status")
				}
				if _, e := strconv.ParseUint(part[2:], 16, 32); e != nil {
					return nil, nil, e
				}
			}
		}
		if v[6] != "" || v[7] != "" || v[8] != "" {
			if v[6] == "" || v[7] == "" || v[8] == "" {
				return nil, nil, fmt.Errorf("partial secondary action")
			}
			secondary++
		}
		counts[code]++
		rules = append(rules, decodeRow{code, i + 1, v})
	}
	for i, want := range []int{6, 33, 20, 2, 1, 31, 2} {
		if counts[uint64(144+i)] != want {
			return nil, nil, fmt.Errorf("XID population differs")
		}
	}
	if secondary != 3 || status != 73 {
		return nil, nil, fmt.Errorf("status/secondary population differs")
	}
	guide, err := rows(4, 3)
	if err != nil {
		return nil, nil, err
	}
	if len(guide) != 25 || guide[0][0] != "Sheet Name" || guide[0][1] != "Column Name" || guide[0][2] != "Description" {
		return nil, nil, fmt.Errorf("guide schema differs")
	}
	for i, row := range guide {
		for j := 0; j < 3; j++ {
			evidence = append(evidence, sourceCell{names[3], fmt.Sprintf("%c%d", 'A'+j, i+1), row[j]})
		}
	}
	xids, err := rows(1, 12)
	if err != nil {
		return nil, nil, err
	}
	for i, row := range xids {
		if i == 0 {
			continue
		}
		id, _ := strconv.ParseUint(row[1], 10, 64)
		if id == 95 || id >= 144 && id <= 150 || id == 154 {
			for j := 10; j < 12; j++ {
				evidence = append(evidence, sourceCell{names[0], fmt.Sprintf("%c%d", 'A'+j, i+1), row[j]})
			}
		}
	}
	return rules, evidence, nil
}

func renderDecode(rows []decodeRow, evidence []sourceCell) ([]byte, error) {
	var b bytes.Buffer
	fmt.Fprintln(&b, "// Code generated from the pinned NVIDIA XID Catalog. DO NOT EDIT.\n\npackage xidcatalog\n\nvar decodeRules = []DecodeRule{")
	for _, r := range rows {
		fmt.Fprintf(&b, "{Code:%d, Row:%d, Cells:[13]string{", r.code, r.row)
		for _, v := range r.cells {
			fmt.Fprintf(&b, "%q,", v)
		}
		fmt.Fprintln(&b, "}},")
	}
	fmt.Fprintln(&b, "}\n\nvar decodeSources = []SourceCell{")
	for _, c := range evidence {
		fmt.Fprintf(&b, "{%q,%q,%q},\n", c.sheet, c.coordinate, c.value)
	}
	fmt.Fprintln(&b, "}")
	return format.Source(b.Bytes())
}
