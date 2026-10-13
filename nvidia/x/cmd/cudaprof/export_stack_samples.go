package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	languagestack "github.com/tmc/lib/nvidia/x/cudaprof/stack"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

type languageSamples struct {
	Journal       string                    `json:"journal"`
	SHA256        string                    `json:"sha256"`
	Receipt       string                    `json:"receipt,omitempty"`
	ReceiptSHA256 string                    `json:"receipt_sha256,omitempty"`
	Producer      *languagestack.GoProducer `json:"producer,omitempty"`
	Samples       []languagestack.Sample    `json:"samples"`
	GoTrace       *languagestack.GoTrace    `json:"go_trace,omitempty"`
	Unavailable   string                    `json:"unavailable,omitempty"`
}

// goTraceOptions selects an admitted prebuilt cmd/trace and a caller-owned
// private scratch directory. Both or neither are set; values are absolute
// and clean. Parsing makes no file system call.
type goTraceOptions struct {
	decoder, scratch string
}

// decoderFor returns the selected decoder, or nil when none is selected.
func (o goTraceOptions) decoderFor() *languagestack.GoTraceDecoder {
	if o.decoder == "" {
		return nil
	}
	return &languagestack.GoTraceDecoder{Tool: o.decoder, Scratch: o.scratch}
}

// set records one option value. Repeats and non-absolute or unclean values
// are refused.
func (o *goTraceOptions) set(name, value string) error {
	p := &o.decoder
	if name == "go-trace-scratch" {
		p = &o.scratch
	}
	if *p != "" {
		return fmt.Errorf("repeated -%s", name)
	}
	if !filepath.IsAbs(value) || filepath.Clean(value) != value {
		return fmt.Errorf("-%s must be an absolute, clean path", name)
	}
	*p = value
	return nil
}

// check enforces the pairing after all options are parsed.
func (o goTraceOptions) check() error {
	if (o.decoder == "") != (o.scratch == "") {
		return errors.New("go trace decoder and scratch must be selected together")
	}
	return nil
}

// register adds both options to fs as set-once values.
func (o *goTraceOptions) register(fs *flag.FlagSet) {
	for _, name := range []string{"go-trace-decoder", "go-trace-scratch"} {
		fs.Func(name, "admitted prebuilt cmd/trace (decoder) or existing private 0700 directory (scratch); both or neither", func(v string) error { return o.set(name, v) })
	}
}

const goTraceOptionsHelp = "-go-trace-decoder PATH, -go-trace-scratch DIR  admitted prebuilt cmd/trace and an existing caller-owned private directory (absolute, clean, mode 0700 or stricter; never created); both or neither; unselected Go traces stay unavailable"

// goHostReceiptKeys are the exact keys cudaprof's host profile writes.
var goHostReceiptKeys = map[string]bool{
	"version": true, "pid": true, "process": true, "session": true, "generation": true,
	"start_wall_utc": true, "end_wall_utc": true, "gpu_clock_alignment": true,
	"cpu_sample_timestamps": true, "goos": true, "goarch": true, "go_version": true,
}

type goHostReceipt struct {
	Version    int    `json:"version"`
	PID        int    `json:"pid"`
	Process    string `json:"process"`
	Session    string `json:"session"`
	Generation uint64 `json:"generation"`
	GOOS       string `json:"goos"`
	GOARCH     string `json:"goarch"`
	GoVersion  string `json:"go_version"`
}

// readGoHostReceipt reads a host receipt strictly. A syntax error is
// returned as err. A duplicate, case-variant or unknown key, or a missing
// producer field, is returned as unavailable text.
func readGoHostReceipt(data []byte) (r goHostReceipt, unavailable string, err error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return r, "", err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return r, "", errors.New("host receipt is not a JSON object")
	}
	seen := make(map[string]bool)
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return r, "", err
		}
		key, _ := tok.(string)
		if !goHostReceiptKeys[key] || seen[key] {
			return r, "ambiguous host receipt: " + key, nil
		}
		seen[key] = true
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return r, "", err
		}
	}
	if err := json.Unmarshal(data, &r); err != nil {
		return r, "", err
	}
	switch {
	case r.Version != 1:
		return r, "producer receipt incomplete: version", nil
	case r.GoVersion == "":
		return r, "producer receipt incomplete: go_version", nil
	case r.GOOS == "":
		return r, "producer receipt incomplete: goos", nil
	case r.GOARCH == "":
		return r, "producer receipt incomplete: goarch", nil
	}
	return r, "", nil
}

// readLanguageSamples reads retained language samples. A Go trace is decoded
// only with the selected decoder; its producer comes from the paired host
// receipt. Pairing is attachment naming plus recorded hashes. File-name
// suffixes and pairing establish no clock, population, process or session
// authenticity.
func readLanguageSamples(ctx context.Context, cap *cudacapture.Capture, goTrace goTraceOptions) ([]languageSamples, error) {
	var names []string
	for name := range cap.Files {
		if strings.HasPrefix(name, "stacks-python-wall-") && strings.HasSuffix(name, ".jsonl") || strings.HasPrefix(name, "stacks-go-") && strings.HasSuffix(name, "-host.json") {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	var out []languageSamples
	var total int64
	for _, name := range names {
		data := cap.Files[name]
		total += int64(len(data))
		if total > 32<<20 {
			return nil, fmt.Errorf("host stack samples exceed total byte limit")
		}
		digest := sha256.Sum256(data)
		group := languageSamples{Journal: name, SHA256: hex.EncodeToString(digest[:])}
		if strings.HasSuffix(name, ".jsonl") {
			samples, err := languagestack.ReadSamples(bytes.NewReader(data), languagestack.Limits{})
			if err != nil {
				return nil, fmt.Errorf("read %s: %w", name, err)
			}
			group.Samples = samples
		} else {
			identity, unavailable, err := readGoHostReceipt(data)
			if err != nil {
				return nil, fmt.Errorf("read host receipt %s: %w", name, err)
			}
			if unavailable != "" {
				group.Unavailable = unavailable
				out = append(out, group)
				continue
			}
			producer := languagestack.GoProducer{GoVersion: identity.GoVersion, GOOS: identity.GOOS, GOARCH: identity.GOARCH}
			group.Producer = &producer
			traceName := strings.TrimSuffix(name, "-host.json") + ".trace"
			trace, ok := cap.Files[traceName]
			if !ok {
				group.Unavailable = "runtime trace attachment missing"
				out = append(out, group)
				continue
			}
			group.Receipt, group.ReceiptSHA256 = group.Journal, group.SHA256
			group.Journal = traceName
			traceDigest := sha256.Sum256(trace)
			group.SHA256 = hex.EncodeToString(traceDigest[:])
			total += int64(len(trace))
			if total > 32<<20 {
				return nil, fmt.Errorf("host stack samples exceed total byte limit")
			}
			dec := goTrace.decoderFor()
			if dec == nil {
				group.Unavailable = languagestack.ErrGoTraceDecoderUnselected.Error()
				out = append(out, group)
				continue
			}
			decoded, err := analyzeRetainedGoTrace(ctx, dec, trace, producer, languagestack.Record{PID: identity.PID, Process: identity.Process, Session: identity.Session, Generation: identity.Generation})
			if err != nil {
				group.Unavailable = err.Error()
			} else {
				group.GoTrace = decoded
				if err := languagestack.QualifyGoMonotonic(decoded, identity.GOOS, identity.GoVersion); err != nil {
					group.Unavailable = err.Error()
				}
				group.Samples = decoded.Samples
				metadata := *decoded
				metadata.Samples = nil
				group.GoTrace = &metadata
			}
		}
		out = append(out, group)
	}
	return out, nil
}

func nativeStackClocks(records []traceRecord) map[stackReference]bool {
	clocks := make(map[stackReference]bool)
	for _, t := range records {
		clocks[stackReference{session: t.session, pid: int(t.process), generation: t.generation}] = t.clocksUnified
	}
	return clocks
}

func appendLanguageSamples(ctx context.Context, cap *cudacapture.Capture, records []traceRecord, goTrace goTraceOptions) ([]traceRecord, error) {
	groups, err := readLanguageSamples(ctx, cap, goTrace)
	if err != nil {
		return nil, err
	}
	return appendLanguageGroups(records, groups)
}

// appendLanguageGroups appends the samples of groups read by
// readLanguageSamples to records.
func appendLanguageGroups(records []traceRecord, groups []languageSamples) ([]traceRecord, error) {
	clocks := nativeStackClocks(records)
	var count int
	for _, group := range groups {
		for i := range group.Samples {
			s := group.Samples[i]
			if s.Type != "sample" {
				continue
			}
			count++
			if count > 100000 {
				return nil, fmt.Errorf("host stack samples exceed total record limit")
			}
			key := stackReference{session: s.Session, pid: s.PID, generation: s.Generation}
			if !clocks[key] || s.Generation == 0 || s.Clock != "monotonic" || s.Time == 0 || s.TID > math.MaxUint32 {
				continue
			}
			s.Status = "native_clock_verified"
			annotations := []perfetto.DebugAnnotation{
				perfetto.NewStringAnnotation("cudaprof_stack_language", s.Language),
				perfetto.NewStringAnnotation("cudaprof_stack_attribution", s.Provenance),
				perfetto.NewStringAnnotation("cudaprof_stack_status", s.Status),
				perfetto.NewStringAnnotation("cudaprof_stack_journal", group.Journal),
				perfetto.NewStringAnnotation("cudaprof_stack_journal_sha256", group.SHA256),
				perfetto.NewUintAnnotation("cudaprof_stack_goroutine", s.Goroutine),
				perfetto.NewBoolAnnotation("cudaprof_stack_truncated", s.Truncated),
			}
			records = append(records, traceRecord{journal: group.Journal, sequence: uint64(i + 1), annotations: annotations, sample: &s, session: s.Session, process: uint64(s.PID), generation: s.Generation, clocksUnified: true,
				rec: cuptiactivity.Record{Kind: cupti.ActivityKindOverhead, Name: s.Language + " " + s.Provenance, Start: s.Time, End: s.Time, Complete: true, ProcessID: uint32(s.PID), ThreadID: uint32(s.TID)}, kind: trackStackSample, group: [2]uint32{uint32(s.PID), uint32(s.TID)}})
		}
	}
	return records, nil
}

// goTraceUnavailable records one language group left unavailable, detached
// from the group read. Journal and Receipt are bounded to 256 bytes and
// Detail to 1024 bytes, each plus a truncation marker; Detail is the
// recorded text with line breaks and other unprintable characters escaped.
type goTraceUnavailable struct {
	Journal       string `json:"journal"`
	SHA256        string `json:"sha256"`
	Receipt       string `json:"receipt,omitempty"`
	ReceiptSHA256 string `json:"receipt_sha256,omitempty"`
	Detail        string `json:"detail"`
}

const (
	maxGoTraceJournalText = 256
	maxGoTraceDetailText  = 1024

	// An unavailable report lists at most maxGoTraceUnavailableEntries
	// entries whose encoding (see goTraceUnavailableJSON) is at most
	// maxGoTraceUnavailableBytes, whichever binds first.
	maxGoTraceUnavailableEntries = 64
	maxGoTraceUnavailableBytes   = 256 << 10
)

// unavailableLanguageGroups returns an entry for each group with recorded
// unavailable text, in order, until the report budget binds. Only Go
// groups record such text. omitted counts the unavailable groups left out
// after that: diagnostic entries omitted from this report, not lost profile
// records, completed decoding, missing evidence or a health verdict.
func unavailableLanguageGroups(groups []languageSamples) (entries []goTraceUnavailable, omitted int) {
	for _, g := range groups {
		if g.Unavailable == "" {
			continue
		}
		if omitted > 0 || len(entries) >= maxGoTraceUnavailableEntries {
			omitted++
			continue
		}
		e := goTraceUnavailable{
			Journal:       boundGoTraceText(g.Journal, maxGoTraceJournalText),
			SHA256:        g.SHA256,
			Receipt:       boundGoTraceText(g.Receipt, maxGoTraceJournalText),
			ReceiptSHA256: g.ReceiptSHA256,
			Detail:        flattenGoTraceDetail(g.Unavailable, maxGoTraceDetailText),
		}
		entries = append(entries, e)
		if data, err := goTraceUnavailableJSON(entries); err != nil || len(data) > maxGoTraceUnavailableBytes {
			entries = entries[:len(entries)-1]
			omitted++
		}
	}
	return entries, omitted
}

// goTraceUnavailableJSON returns entries encoded as they appear in an
// alignment receipt, which json.MarshalIndent writes with HTML escaping and
// a two-space indent, three levels deep in arms[i].go_trace_unavailable.
func goTraceUnavailableJSON(entries []goTraceUnavailable) ([]byte, error) {
	return json.MarshalIndent(entries, "      ", "  ")
}

// goTraceOmittedLine renders the count of omitted unavailable groups as one status
// line.
func goTraceOmittedLine(omitted int) string {
	return fmt.Sprintf("go trace: %d unavailable group(s) omitted from this report", omitted)
}

// line renders u as one status line.
func (u goTraceUnavailable) line() string {
	return "go trace " + strconv.Quote(u.Journal) + ": unavailable: " + u.Detail
}

// boundGoTraceText returns s, or a prefix of s cut on a rune boundary with
// a truncation marker if s exceeds limit bytes.
func boundGoTraceText(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	n := limit
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return fmt.Sprintf("%s...[%d bytes truncated]", s[:n], len(s)-n)
}

// flattenGoTraceDetail renders s on one line: each newline, as between the
// errors of errors.Join, becomes "; " and every other unprintable character
// or invalid byte is escaped. The result is bounded by limit bytes plus a
// truncation marker.
func flattenGoTraceDetail(s string, limit int) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		var piece string
		switch {
		case r == '\n':
			piece = "; "
		case r == utf8.RuneError && size == 1:
			piece = fmt.Sprintf(`\x%02x`, s[i])
		case !strconv.IsPrint(r):
			q := strconv.QuoteRune(r)
			piece = q[1 : len(q)-1]
		default:
			piece = s[i : i+size]
		}
		if b.Len()+len(piece) > limit {
			fmt.Fprintf(&b, "...[%d bytes truncated]", len(s)-i)
			break
		}
		b.WriteString(piece)
		i += size
	}
	return b.String()
}

func analyzeRetainedGoTrace(ctx context.Context, dec *languagestack.GoTraceDecoder, data []byte, producer languagestack.GoProducer, identity languagestack.Record) (*languagestack.GoTrace, error) {
	return dec.Analyze(ctx, data, producer, identity, languagestack.Limits{})
}
