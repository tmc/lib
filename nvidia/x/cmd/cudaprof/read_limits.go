package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Reading a capture is bounded twice: by the size of the artifact's files,
// which cudacapture checks as it reads, and by the records and decompressed
// bytes of each journal, which the decoders check. Every bound a reader does
// not set explicitly is the larger of its built-in default and what the
// capture records it was written under, so a capture this tool wrote reads
// back without restating the flags it was written with. A reader that sets a
// bound uses it as set, even below what the capture records.
//
// The recorded bounds come from the capture itself, which is right for the
// captures a user points this tool at and wrong for a service decoding
// artifacts it does not trust; such a service uses cudacapture with its
// defaults.

// trustedLimits returns limits with the capture's recorded bounds admitted and,
// unless limits sets it, the input bound raised to size, the size of the file
// about to be read.
func trustedLimits(limits cudacapture.Limits, size int64) cudacapture.Limits {
	limits.Recorded = true
	if limits.InputBytes <= 0 && size > cudacapture.MaxCaptureInputBytes {
		limits.InputBytes = size
	}
	return limits
}

// storedLimits returns the limits recorded in cap's capture-status.json, or
// the zero captureLimits for a capture that records none.
func storedLimits(cap *cudacapture.Capture) captureLimits {
	data, ok := cap.Files["capture-status.json"]
	if !ok {
		return captureLimits{}
	}
	var envelope struct {
		Status struct {
			Limits *captureLimits `json:"limits"`
		} `json:"status"`
	}
	if json.Unmarshal(data, &envelope) != nil || envelope.Status.Limits == nil {
		return captureLimits{}
	}
	return *envelope.Status.Limits
}

// decodeLimits returns the bounds for decoding cap's journals: each of
// maxRecords and journalBytes as given when positive, otherwise the larger of
// the built-in default and what cap records.
func decodeLimits(cap *cudacapture.Capture, maxRecords, journalBytes int64) (int64, int64) {
	stored := storedLimits(cap)
	if maxRecords <= 0 {
		maxRecords = max(controllerMaxJournalRecords, stored.MaxRecords, stored.JournalRecords)
	}
	if journalBytes <= 0 {
		journalBytes = max(cudacapture.MaxCaptureInputBytes, stored.JournalBytes)
	}
	return maxRecords, journalBytes
}

// decodeRemedy adds to a journal refused under maxRecords or journalBytes the
// flags that would decode it, measured from the journals themselves. Any other
// error is returned as is.
func decodeRemedy(cap *cudacapture.Capture, err error, maxRecords, journalBytes int64) error {
	if !errors.Is(err, wire.ErrRecordLimit) && !errors.Is(err, timeline.ErrJournalBytes) {
		return err
	}
	journals, jerr := timeline.Journals(cap)
	if jerr != nil {
		return err
	}
	records, size, merr := timeline.Measure(journals)
	if merr != nil {
		return err
	}
	var flags []string
	if records > maxRecords {
		flags = append(flags, fmt.Sprintf("--max-records=%d", records))
	}
	if size > journalBytes {
		flags = append(flags, fmt.Sprintf("--journal-bytes=%d", size))
	}
	if len(flags) == 0 {
		return err
	}
	return fmt.Errorf("%w; rerun with %s", err, strings.Join(flags, " "))
}

// readLimitRemedy adds to a refusal of a capture's size the read flags that
// would admit it. Any other error is returned as is.
func readLimitRemedy(err error) error {
	var e *cudacapture.LimitError
	if !errors.As(err, &e) {
		return err
	}
	var flags []string
	if e.Need.FileBytes > 0 {
		flags = append(flags, fmt.Sprintf("--capture-file-bytes=%d", e.Need.FileBytes))
	}
	if e.Need.TotalBytes > 0 {
		flags = append(flags, fmt.Sprintf("--capture-total-bytes=%d", e.Need.TotalBytes))
	}
	return fmt.Errorf("%w; rerun with %s", err, strings.Join(flags, " "))
}

// fileSize returns the size of the file at path, or zero for a directory or a
// path that cannot be read.
func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil || fi.IsDir() {
		return 0
	}
	return fi.Size()
}
