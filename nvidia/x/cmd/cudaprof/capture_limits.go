package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// The bounds a capture is written under unless the caller sets them. They are
// above the cudacapture defaults, which exist for reading artifacts of unknown
// origin; a capture records the bounds it used, and cudaprof's readers admit
// it from there.
//
// Measured on GB10 with a capture whose one journal is 86 MB compressed and
// 678 MB, 2-4 million records, decompressed: inspect peaks at 2.2 GB resident
// and compare against a second capture at 3.0 GB. The bounds allocate nothing
// by themselves, so a smaller capture costs what it did before.
const (
	captureFileBytes  = 1 << 30
	captureTotalBytes = 1 << 30
	captureMaxRecords = 1 << 23
)

// captureLimits records the bounds a capture was written under, and the sizes
// that decide what a reader needs to decode it. It is published as the
// "limits" object of capture-status.json.
type captureLimits struct {
	// MaxRecords is the most records the capture admitted from one journal.
	MaxRecords int64 `json:"max_records,omitempty"`
	// CaptureFileBytes and CaptureTotalBytes are the artifact budget.
	CaptureFileBytes  int64 `json:"capture_file_bytes,omitempty"`
	CaptureTotalBytes int64 `json:"capture_total_bytes,omitempty"`
	// JournalRecords is the most records any one journal held, including a
	// journal refused for holding more than MaxRecords.
	JournalRecords int64 `json:"journal_records,omitempty"`
	// JournalBytes is the largest journal, decompressed.
	JournalBytes int64 `json:"journal_bytes,omitempty"`
	// AttachmentBytes is the total of every attachment offered to the
	// artifact, uncompressed, whether or not it was kept.
	AttachmentBytes int64 `json:"attachment_bytes,omitempty"`
}

// journalRecordLimitError reports a journal that holds more records than the
// capture admits.
type journalRecordLimitError struct {
	Limit, Records int64
	Err            error
}

func (e *journalRecordLimitError) Error() string {
	return fmt.Sprintf("capture journal holds %d records, over the limit of %d (rerun with --max-records=%d): %v", e.Records, e.Limit, roundLimit(e.Records), e.Err)
}

func (e *journalRecordLimitError) Unwrap() error { return e.Err }

// countJournalRecords counts the frames of the journal at path, keeping none
// of them.
func countJournalRecords(path string) (int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	d := wire.NewDecoder(bufio.NewReader(f), wire.Limits{MaxRecords: math.MaxUint32})
	for {
		if _, err := d.Next(); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return int64(d.Stats().Records), nil
			}
			return 0, err
		}
	}
}

// noteJournal records in r.Limits the record count of one observed journal.
func (r *captureResult) noteJournal(state journalState, err error) {
	if r.Limits == nil {
		r.Limits = &captureLimits{}
	}
	n := int64(state.records)
	if e := (*journalRecordLimitError)(nil); errors.As(err, &e) {
		n = e.Records
	}
	r.Limits.JournalRecords = max(r.Limits.JournalRecords, n)
}

// recordLimits sets r.Limits to the bounds the capture is written under and
// the sizes of the attachments offered to it, keeping the journal record count
// observeSession noted. A maxRecords of zero means captureMaxRecords.
func (r *captureResult) recordLimits(maxRecords int64, budget captureArtifactBudget) {
	var l captureLimits
	if r.Limits != nil {
		l.JournalRecords = r.Limits.JournalRecords
	}
	if maxRecords <= 0 {
		maxRecords = captureMaxRecords
	}
	budget = budget.effective()
	l.MaxRecords, l.CaptureFileBytes, l.CaptureTotalBytes = maxRecords, budget.FileBytes, budget.TotalBytes
	if r.ScratchDir != "" {
		for _, a := range scratchAttachments(r.ScratchDir) {
			l.AttachmentBytes += a.size
			if cudacapture.IsJournalName(a.name) {
				l.JournalBytes = max(l.JournalBytes, a.size)
			}
		}
	}
	r.Limits = &l
}

// roundLimit returns the power of two at or above n: a bound to rerun a
// capture with, which must admit a run a little larger than the one observed.
func roundLimit(n int64) int64 {
	if n <= 1 {
		return 1
	}
	return 1 << bits.Len64(uint64(n-1))
}

// gzipBound bounds the size of n bytes gzip-compressed: the data, stored-block
// framing of at most 5 bytes per 16 KiB, and the header and trailer, with room
// for the capture status written beside them.
func gzipBound(n int64) int64 {
	return n + n>>11 + 64<<10
}

// partialCapture reports a capture that left something out: an attachment
// over the artifact budget, or a journal over --max-records. The error names
// every flag the capture needs, sized from what this one observed.
func partialCapture(result captureResult) error {
	l := result.Limits
	if l == nil {
		l = &captureLimits{}
	}
	var what, flags []string
	var file int64
	for _, f := range result.OmittedFiles {
		file = max(file, f.Bytes)
	}
	if n := len(result.OmittedFiles); n > 0 {
		what = append(what, fmt.Sprintf("%d attachment(s) omitted over the artifact budget", n))
		// An omitted journal's size is its raw size. The compressed member
		// the budget is measured against is smaller, except for data that
		// does not compress, which gzip grows by its framing.
		if need := gzipBound(file); need > l.CaptureFileBytes {
			flags = append(flags, fmt.Sprintf("--capture-file-bytes=%d", roundLimit(need)))
		}
		if need := gzipBound(l.AttachmentBytes); need > l.CaptureTotalBytes {
			flags = append(flags, fmt.Sprintf("--capture-total-bytes=%d", roundLimit(need)))
		}
	}
	if l.MaxRecords > 0 && l.JournalRecords > l.MaxRecords {
		what = append(what, fmt.Sprintf("a journal holds %d records, over --max-records=%d", l.JournalRecords, l.MaxRecords))
		flags = append(flags, fmt.Sprintf("--max-records=%d", roundLimit(l.JournalRecords)))
	}
	if len(what) == 0 {
		return nil
	}
	msg := "capture partial: " + strings.Join(what, "; ")
	if len(flags) > 0 {
		msg += "; rerun with " + strings.Join(flags, " ")
	}
	return errors.New(msg + ", or pass --allow-partial to accept it")
}
