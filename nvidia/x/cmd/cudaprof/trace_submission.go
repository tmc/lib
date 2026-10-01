package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Validate imported native-match envelopes before presenting their status
// columns. Supplied export provenance is not independent collector evidence.
func validateSubmissionQuery(rows [][]string, domain string) error {
	if len(rows) == 0 {
		return errors.New("trace submission query: missing schema")
	}
	columns := map[string]int{}
	for i, name := range rows[0] {
		if _, duplicate := columns[name]; duplicate {
			return errors.New("trace submission query: duplicate column")
		}
		columns[name] = i
	}
	for _, name := range []string{"state", "row_kind", "capture_sha256", "arm", "journal", "raw_activity_sequence", "raw_activity_index", "domain", "api_status", "range_status", "metadata_status", "provenance", "native_match_hex", "reason"} {
		if _, ok := columns[name]; !ok {
			return fmt.Errorf("trace submission query: missing %s column", name)
		}
	}
	for _, row := range rows[1:] {
		if len(row) != len(rows[0]) {
			return errors.New("trace submission query: malformed row")
		}
		at := func(name string) string { return row[columns[name]] }
		if len(at("capture_sha256")) != 64 {
			return errors.New("trace submission query: missing capture identity")
		}
		if _, err := hex.DecodeString(at("capture_sha256")); err != nil {
			return errors.New("trace submission query: invalid capture identity")
		}
		if at("domain") != domain {
			return errors.New("trace submission query: domain differs from request")
		}
		if at("state") != "available" && at("state") != "unavailable" {
			return errors.New("trace submission query: unsupported state")
		}
		if at("row_kind") == "coverage" {
			continue
		}
		if at("row_kind") != "activity" || at("state") != "available" {
			return errors.New("trace submission query: invalid activity disposition")
		}
		if at("journal") == "" {
			return errors.New("trace submission query: missing journal")
		}
		sequence, err := strconv.ParseUint(at("raw_activity_sequence"), 10, 64)
		if err != nil || sequence == 0 {
			return errors.New("trace submission query: invalid raw source sequence")
		}
		index, err := strconv.Atoi(at("raw_activity_index"))
		if err != nil || index < 0 {
			return errors.New("trace submission query: invalid activity index")
		}
		data, err := hex.DecodeString(at("native_match_hex"))
		if err != nil {
			return errors.New("trace submission query: invalid native receipt encoding")
		}
		var match timeline.Correlation
		if err := json.Unmarshal(data, &match); err != nil {
			return fmt.Errorf("trace submission query: native receipt: %w", err)
		}
		if match.Activity != index || match.APIStatus != at("api_status") || match.RangeStatus != at("range_status") || match.MetadataStatus != at("metadata_status") || match.Provenance != at("provenance") {
			return errors.New("trace submission query: columns disagree with native receipt")
		}
	}
	return nil
}
