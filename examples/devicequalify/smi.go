package main

import (
	"fmt"
	"strconv"
	"strings"
)

// smiFields are the nvidia-smi query fields, in the order the CSV returns
// them. Keep queryArgs and smiFields in step.
var smiFields = []string{
	"name", "driver_version", "pci.bus_id",
	"pcie.link.gen.current", "pcie.link.gen.max",
	"pcie.link.width.current", "pcie.link.width.max",
	"ecc.mode.current", "ecc.errors.uncorrected.volatile.total",
}

// requiredFields must hold a real value. A device that cannot name itself
// or its driver is not a device a measurement should trust. The remaining
// fields are recorded but may legitimately be unsupported: an integrated
// part reports no ECC, and its link fields describe a bus the weights do
// not necessarily travel.
var requiredFields = map[string]bool{"name": true, "driver_version": true, "pci.bus_id": true}

// unusable are the strings nvidia-smi substitutes for a value it could not
// read. "ERR!" is the one that matters: it is what a wedged device reports
// where a number belongs.
var unusable = []string{"unknown", "err!", "n/a", "[n/a]", "[not supported]", "[unknown error]", ""}

// isUnusable reports whether nvidia-smi declined to answer.
func isUnusable(value string) bool {
	v := strings.ToLower(strings.TrimSpace(strings.Trim(value, "[]")))
	for _, bad := range unusable {
		if v == strings.Trim(bad, "[]") {
			return true
		}
	}
	return false
}

// parseSMI decodes one CSV row per GPU from nvidia-smi --format=csv,noheader,nounits.
func parseSMI(out string) ([]map[string]string, error) {
	var rows []map[string]string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != len(smiFields) {
			return nil, fmt.Errorf("nvidia-smi: got %d fields, want %d", len(parts), len(smiFields))
		}
		row := make(map[string]string, len(smiFields))
		for i, name := range smiFields {
			row[name] = strings.TrimSpace(parts[i])
		}
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("nvidia-smi: no GPU rows")
	}
	return rows, nil
}

// integer reads a field that must be a non-negative count, reporting
// whether it held one at all.
func integer(row map[string]string, field string) (int, bool) {
	if isUnusable(row[field]) {
		return 0, false
	}
	n, err := strconv.Atoi(strings.Fields(row[field])[0])
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// eccSupport distinguishes the two ways a device declines to report an
// error count. Not supported is a fact about the part, and an integrated
// GB10 reports it on a healthy day; anything else in that field is the
// device failing to answer a question it is meant to answer.
type eccSupport int

const (
	eccUnsupported eccSupport = iota // the part has no ECC to report
	eccCount                         // a real count was read
	eccMalformed                     // ERR!, Unknown, or text that is not a count
)

// unsupported are the strings that mean "this part does not have this
// feature", as opposed to "this device could not answer".
var unsupported = []string{"n/a", "not supported", "disabled"}

// eccErrors classifies the uncorrected volatile error field.
func eccErrors(row map[string]string) (eccSupport, int) {
	value := strings.ToLower(strings.TrimSpace(strings.Trim(row["ecc.errors.uncorrected.volatile.total"], "[]")))
	for _, s := range unsupported {
		if value == s {
			return eccUnsupported, 0
		}
	}
	if n, ok := integer(row, "ecc.errors.uncorrected.volatile.total"); ok {
		return eccCount, n
	}
	return eccMalformed, 0
}
