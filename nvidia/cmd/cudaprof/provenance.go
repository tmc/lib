package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// reportProvenance states when a profile was taken and checks that it was taken
// from the artifacts shipped beside it.
//
// The check is cheap because the evidence is already in the file: a mapping
// carries the build ID of the image it was recorded against, and this toolkit's
// build ID is the SHA-256 of that image's bytes, so a profile that came from a
// different compilation names a digest none of the capture's own files hash to.
//
// It is worth enforcing because the failure it catches is silent and total. A
// script copied a stale profile after its test exited rc=1, and the resulting
// ranking attributed 39.4% of samples to two kernels the current code never
// reaches. The tool had the mismatched build ID in hand throughout and said
// nothing. A mismatch is reported rather than refused: the profile is still a
// valid profile of something, and the reader is the one who knows of what.
func reportProvenance(w io.Writer, cap *cudacapture.Capture) {
	if cap == nil || cap.Profile == nil {
		return
	}
	if t := cap.Profile.TimeNanos; t > 0 {
		taken := time.Unix(0, t)
		// The age is printed next to the timestamp so that an artifact older
		// than the code under test is visible without arithmetic. That
		// subtraction is exactly what nobody performs on a profile they
		// believe they just captured.
		fmt.Fprintf(w, "cudaprof: profile taken %s (%s ago)\n",
			taken.Format(time.RFC3339), time.Since(taken).Round(time.Second))
	}

	digests := make(map[string]string, len(cap.Files))
	for name, data := range cap.Files {
		digests[filepath.Base(name)] = fmt.Sprintf("%x", sha256.Sum256(data))
	}
	type mismatch struct{ file, profileID, captureID string }
	var bad []mismatch
	var unrecorded []string
	for _, m := range cap.Profile.Mapping {
		if m == nil || m.File == "" {
			continue
		}
		base := filepath.Base(m.File)
		if m.BuildID == "" {
			// Silence here would be the same error one level down: an absent
			// build ID is not agreement, and a check that passes on it passes
			// on every profile ever captured without one.
			if _, ok := digests[base]; ok {
				unrecorded = append(unrecorded, base)
			}
			continue
		}
		have, ok := digests[base]
		if !ok {
			// A mapping for an image this capture does not carry -- the host
			// binary, a shared library -- is not evidence either way.
			continue
		}
		if have != m.BuildID {
			bad = append(bad, mismatch{base, m.BuildID, have})
		}
	}
	if len(bad) == 0 {
		if len(unrecorded) > 0 {
			sort.Strings(unrecorded)
			fmt.Fprintf(w, "cudaprof: provenance unverified: the profile records no build ID for %s, so it cannot be shown to come from the artifacts it ships with\n",
				strings.Join(unrecorded, ", "))
		}
		return
	}
	sort.Slice(bad, func(i, j int) bool { return bad[i].file < bad[j].file })
	fmt.Fprintf(w, "cudaprof: this profile did not come from the artifacts it ships with.\n")
	for _, b := range bad {
		fmt.Fprintf(w, "cudaprof:   %s: profile records build ID %s, the capture's copy hashes to %s\n",
			b.file, b.profileID, b.captureID)
	}
	fmt.Fprintf(w, "cudaprof:   Every attribution below names code that may never have run. Re-capture before reading it.\n")
}
