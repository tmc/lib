package xidcatalog

import (
	"strconv"
	"strings"
)

//go:generate go run ./internal/gen -decode -url https://docs.nvidia.com/deploy/xid-errors/_downloads/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx -sha256 7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5 -retrieved 2026-10-06 -out decode.gen.go

// DecodeRule preserves one row of the pinned Xid 144-150 Decode sheet.
// Cells are the unmodified A through M values, including blank cells. Row is
// the actual workbook row number. SourceSHA256 identifies the workbook.
type DecodeRule struct {
	Code  uint64
	Row   int
	Cells [13]string
}

// SourceCell retains supporting interpretation or linkage text separately
// from a rule. It is not an instruction to perform recovery.
type SourceCell struct{ Sheet, Coordinate, Value string }

// DecodeRules returns detached source rows for id, in workbook order.
func DecodeRules(id uint64) []DecodeRule {
	var out []DecodeRule
	for _, r := range decodeRules {
		if r.Code == id {
			out = append(out, r)
		}
	}
	return out
}

// DecodeSources returns detached header, guide and XID linkage cells.
func DecodeSources() []SourceCell { return append([]SourceCell(nil), decodeSources...) }

// DecodeOutcome describes matching evidence, never a health or recovery policy.
type DecodeOutcome uint8

const (
	UnknownDriver DecodeOutcome = iota
	UnsupportedProduct
	NoDecodeMatch
	OneDecodeMatch
	AmbiguousDecode
)

// DecodeVariant identifies the independent primary or secondary IntrInfo mask.
type DecodeVariant uint8

const (
	PrimaryDecode DecodeVariant = iota
	SecondaryDecode
)

// DecodeMatch retains the actual matching source row and raw bucket. A blank
// status cell is unconstrained by this source, not an implicit zero value.
type DecodeMatch struct {
	Rule              DecodeRule
	Variant           DecodeVariant
	Bucket            string
	StatusConstrained bool
}

// DecodeResult preserves caller driver identity and every matching variant.
// More than one match is ambiguous; no first or preferred action is selected.
type DecodeResult struct {
	Driver        string
	DriverMajor   uint32
	Outcome       DecodeOutcome
	Applicability Applicability
	Matches       []DecodeMatch
}

func driverMajor(driver string) (uint32, bool) {
	if driver == "" || len(driver) > 128 {
		return 0, false
	}
	parts := strings.Split(driver, ".")
	if len(parts) > 3 {
		return 0, false
	}
	var major uint32
	for i, part := range parts {
		if part == "" {
			return 0, false
		}
		for _, c := range part {
			if c < '0' || c > '9' {
				return 0, false
			}
		}
		n, err := strconv.ParseUint(part, 10, 32)
		if err != nil {
			return 0, false
		}
		if i == 0 {
			major = uint32(n)
		}
	}
	return major, major != 0
}

func matchesMask(pattern string, value uint32) bool {
	if len(pattern) != 32 {
		return false
	}
	for i, c := range pattern {
		switch c {
		case '-':
		case '0', '1':
			if (value>>uint(31-i))&1 != uint32(c-'0') {
				return false
			}
		default:
			return false
		}
	}
	return true
}

func matchesStatus(raw string, value uint32) bool {
	if raw == "" {
		return true
	}
	for _, s := range strings.Split(raw, "/") {
		if !strings.HasPrefix(s, "0x") {
			return false
		}
		n, err := strconv.ParseUint(s[2:], 16, 32)
		if err == nil && uint32(n) == value {
			return true
		}
	}
	return false
}

// Decode matches the caller's IntrInfo and errorStatus against the pinned
// source for a documented product and a numeric driver version (one to three
// decimal components). Unknown or malformed drivers do not select V1. Drivers
// below R575 select V1; R575 and later select V2. Secondary masks are tested
// independently of primary masks. No other registers or remediation are inferred.
func Decode(id uint64, product Architecture, driver string, intrInfo, errorStatus uint32) DecodeResult {
	_, applicability := LookupProduct(id, product)
	result := DecodeResult{Driver: driver, Applicability: applicability}
	major, known := driverMajor(driver)
	if !known {
		result.Outcome = UnknownDriver
		return result
	}
	result.DriverMajor = major
	if applicability != Applicable {
		result.Outcome = UnsupportedProduct
		return result
	}
	column := 2
	if major >= 575 {
		column = 3
	}
	for _, r := range decodeRules {
		if r.Code != id || !matchesStatus(r.Cells[4], errorStatus) {
			continue
		}
		if matchesMask(r.Cells[column], intrInfo) {
			result.Matches = append(result.Matches, DecodeMatch{r, PrimaryDecode, r.Cells[5], r.Cells[4] != ""})
		}
		if r.Cells[column+4] != "" && matchesMask(r.Cells[column+4], intrInfo) {
			result.Matches = append(result.Matches, DecodeMatch{r, SecondaryDecode, r.Cells[8], r.Cells[4] != ""})
		}
	}
	switch len(result.Matches) {
	case 0:
		result.Outcome = NoDecodeMatch
	case 1:
		result.Outcome = OneDecodeMatch
	default:
		result.Outcome = AmbiguousDecode
	}
	return result
}
