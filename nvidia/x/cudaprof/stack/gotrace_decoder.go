package stack

import (
	"bytes"
	"context"
	"crypto/sha256"
	"debug/buildinfo"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// A GoTraceDecoder decodes Go execution traces by running a prebuilt cmd/trace
// executable selected by the caller, after admitting the exact decoder bytes,
// trace header and producer against reviewed records. It never searches PATH,
// probes versions, builds, downloads, or selects a toolchain. Decoding is
// format evidence only; it establishes no timing, clock or attribution.
//
// No decoder is admitted yet, so every call is refused before any file
// system access.
//
// The zero value is not usable: Tool and Scratch are required.
type GoTraceDecoder struct {
	// Tool is the absolute, clean path of a prebuilt cmd/trace executable.
	Tool string

	// Scratch is an absolute, clean, existing directory, mode 0700 or
	// stricter, held with all its ancestors under the caller's trusted
	// writer custody for the call. It is never created. Each call creates
	// and removes its own private directory inside it. Concurrent tampering
	// by a process with the caller's own uid is not defended.
	Scratch string

	Limits GoTraceLimits
}

// GoTraceLimits bounds one call. Zero fields select the default, which is
// also the ceiling; negative or larger values are refused.
type GoTraceLimits struct {
	InputBytes  int64         // trace input, 32 MiB
	OutputBytes int64         // decoder stdout, 32 MiB
	StderrBytes int64         // decoder stderr, 1 MiB
	ToolBytes   int64         // decoder executable, 64 MiB
	Timeout     time.Duration // decoder child wall time, 30s
	WaitDelay   time.Duration // post-exit or post-kill pipe drain, 2s
}

var goTraceCeilings = GoTraceLimits{
	InputBytes:  32 << 20,
	OutputBytes: 32 << 20,
	StderrBytes: 1 << 20,
	ToolBytes:   64 << 20,
	Timeout:     30 * time.Second,
	WaitDelay:   2 * time.Second,
}

// GoProducer identifies the process that wrote a trace. It is always the
// caller's assertion (for the CLI, values read from a retained host receipt
// whose pairing with the trace rests only on attachment naming and recorded
// hashes). Admission compares it for exact equality with a reviewed record;
// that comparison does not authenticate it.
type GoProducer struct {
	GoVersion string `json:"go_version"`
	GOOS      string `json:"goos"`
	GOARCH    string `json:"goarch"`
}

// GoTraceDecode is the receipt for one successful decode.
type GoTraceDecode struct {
	Admission     string     `json:"admission"`
	ToolSHA256    string     `json:"tool_sha256"`
	ToolBytes     int64      `json:"tool_bytes"`
	ToolPath      string     `json:"tool_path"`
	ToolGoVersion string     `json:"tool_go_version"`
	ToolGOOS      string     `json:"tool_goos"`
	ToolGOARCH    string     `json:"tool_goarch"`
	RecipeSHA256  string     `json:"recipe_sha256"`
	Receipt       string     `json:"admission_receipt_sha256"`
	Review        string     `json:"admission_review_sha256"`
	Header        int        `json:"header"`
	Producer      GoProducer `json:"producer"` // caller-asserted; equal to the admitted record's producer, not authenticated
	InputSHA256   string     `json:"input_sha256"`
	InputBytes    int64      `json:"input_bytes"`
	OutputBytes   int64      `json:"output_bytes"`
	StderrBytes   int64      `json:"stderr_bytes"`
	Cleanup       string     `json:"cleanup"` // "removed" or "failed: DIR: ERROR"
}

// Errors returned by GoTraceDecoder. Only ErrGoTraceMalformed says that the
// input is bad; every other error leaves its validity unknown. Caller
// cancellation is returned as a wrapped context error.
var (
	ErrGoTraceDecoderUnselected = errors.New("go trace decoder not selected")
	ErrGoTraceScratch           = errors.New("go trace scratch invalid")
	ErrGoTraceLimits            = errors.New("go trace limits invalid")
	ErrGoTraceLimit             = errors.New("go trace limit exceeded")
	ErrGoTraceHeader            = errors.New("go trace header not recognized")
	ErrGoTraceUnadmitted        = errors.New("go trace decoder or producer not admitted")
	ErrGoTraceCustody           = errors.New("go trace decoder custody failed")
	ErrGoTraceStart             = errors.New("go trace decoder start failed")
	ErrGoTraceSignal            = errors.New("go trace decoder terminated by signal")
	ErrGoTraceReap              = errors.New("go trace decoder output not drained")
	ErrGoTraceDecoderFailed     = errors.New("go trace decoder failed")
	ErrGoTraceMalformed         = errors.New("go trace rejected by admitted decoder")
	ErrGoTraceCleanup           = errors.New("go trace scratch cleanup failed")
)

var (
	errOutputLimit = errors.New("decoder output limit exceeded")
	errChildTime   = errors.New("decoder time limit exceeded")
)

// goTraceAdmission is one reviewed decoder/producer tuple. Every field is an
// exact match key except ID, Receipt and Review, which identify the evidence.
type goTraceAdmission struct {
	ID            string     // hex sha256 of the canonical encoding
	ToolSHA256    string     // digest of the exact decoder bytes
	ToolBytes     int64      // exact decoder size
	ToolPath      string     // buildinfo Path, as observed in stage 1
	ToolGoVersion string     // buildinfo GoVersion, e.g. "go1.27.2"
	ToolGOOS      string     // buildinfo setting GOOS
	ToolGOARCH    string     // buildinfo setting GOARCH
	RecipeSHA256  string     // digest of the sealed stage-0 recipe manifest
	Header        int        // trace header version, e.g. 26
	Producer      GoProducer // exact GoVersion, GOOS, GOARCH
	Receipt       string     // hex sha256 of the reviewed stage-1 receipt
	Review        string     // hex sha256 of the independent review that froze the tuple
}

// goTraceAdmissions holds reviewed records. A record is added only by a
// separate signed change after stage-0 and stage-1 evidence is reviewed;
// neither an ID nor any digest string here authenticates that evidence.
var goTraceAdmissions []goTraceAdmission

var (
	goTraceDigest    = regexp.MustCompile(`^[0-9a-f]{64}$`)
	goTraceLabel     = regexp.MustCompile(`^[A-Za-z0-9._/+-]+$`)
	goTraceGoVersion = regexp.MustCompile(`^go1\.[0-9]+(\.[0-9]+)?$`)
)

// validAdmission checks a record's fields before anything is encoded.
func validAdmission(r goTraceAdmission) error {
	for _, f := range []struct{ name, v string }{
		{"tool sha256", r.ToolSHA256}, {"recipe sha256", r.RecipeSHA256},
		{"receipt", r.Receipt}, {"review", r.Review},
	} {
		if !goTraceDigest.MatchString(f.v) {
			return fmt.Errorf("invalid %s", f.name)
		}
	}
	if r.ToolBytes < 1 || r.ToolBytes > goTraceCeilings.ToolBytes {
		return fmt.Errorf("invalid tool bytes %d", r.ToolBytes)
	}
	if r.Header < 1 || r.Header > 255 {
		return fmt.Errorf("invalid header %d", r.Header)
	}
	for _, f := range []struct{ name, v string }{
		{"tool path", r.ToolPath}, {"tool go version", r.ToolGoVersion},
		{"tool goos", r.ToolGOOS}, {"tool goarch", r.ToolGOARCH},
		{"producer go version", r.Producer.GoVersion}, {"producer goos", r.Producer.GOOS},
		{"producer goarch", r.Producer.GOARCH},
	} {
		if len(f.v) > 64 || !goTraceLabel.MatchString(f.v) {
			return fmt.Errorf("invalid %s", f.name)
		}
	}
	if !goTraceGoVersion.MatchString(r.ToolGoVersion) || !goTraceGoVersion.MatchString(r.Producer.GoVersion) {
		return fmt.Errorf("invalid go version")
	}
	return nil
}

// encodeGoTraceAdmission returns the typed, length-prefixed canonical encoding
// of r. A string is 's', a big-endian uint32 length and the bytes; an integer
// is 'i' and a big-endian int64. ID is not encoded.
func encodeGoTraceAdmission(r goTraceAdmission) []byte {
	var b []byte
	str := func(s string) {
		b = append(b, 's')
		b = binary.BigEndian.AppendUint32(b, uint32(len(s)))
		b = append(b, s...)
	}
	num := func(n int64) {
		b = append(b, 'i')
		b = binary.BigEndian.AppendUint64(b, uint64(n))
	}
	str("cudaprof.gotrace.admission.v2")
	str(r.ToolSHA256)
	num(r.ToolBytes)
	str(r.ToolPath)
	str(r.ToolGoVersion)
	str(r.ToolGOOS)
	str(r.ToolGOARCH)
	str(r.RecipeSHA256)
	num(int64(r.Header))
	str(r.Producer.GoVersion)
	str(r.Producer.GOOS)
	str(r.Producer.GOARCH)
	str(r.Receipt)
	str(r.Review)
	return b
}

func goTraceAdmissionID(encoded []byte) string {
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

// validateGoTraceAdmissions checks a whole table in three pure phases:
// fields, duplicate keys, then canonical IDs. encode is reached only for
// tables that pass the first two phases.
func validateGoTraceAdmissions(table []goTraceAdmission, encode func(goTraceAdmission) []byte) error {
	for i, r := range table {
		if err := validAdmission(r); err != nil {
			return fmt.Errorf("record %d: %w", i, err)
		}
	}
	type key struct {
		tool   string
		header int
		p      GoProducer
	}
	seen := make(map[key]bool)
	for i, r := range table {
		k := key{r.ToolSHA256, r.Header, r.Producer}
		if seen[k] {
			return fmt.Errorf("record %d: duplicate key", i)
		}
		seen[k] = true
	}
	for i, r := range table {
		if goTraceAdmissionID(encode(r)) != r.ID {
			return fmt.Errorf("record %d: id mismatch", i)
		}
	}
	return nil
}

// resolveGoTraceLimits applies defaults and refuses negative or
// above-ceiling fields.
func resolveGoTraceLimits(l GoTraceLimits) (GoTraceLimits, error) {
	c := goTraceCeilings
	for _, f := range []struct {
		name string
		v    *int64
		max  int64
	}{
		{"input bytes", &l.InputBytes, c.InputBytes},
		{"output bytes", &l.OutputBytes, c.OutputBytes},
		{"stderr bytes", &l.StderrBytes, c.StderrBytes},
		{"tool bytes", &l.ToolBytes, c.ToolBytes},
		{"timeout", (*int64)(&l.Timeout), int64(c.Timeout)},
		{"wait delay", (*int64)(&l.WaitDelay), int64(c.WaitDelay)},
	} {
		switch {
		case *f.v < 0 || *f.v > f.max:
			return GoTraceLimits{}, fmt.Errorf("%w: %s %d", ErrGoTraceLimits, f.name, *f.v)
		case *f.v == 0:
			*f.v = f.max
		}
	}
	return l, nil
}

// parseGoTraceHeader parses the exact 16-byte "go 1.NN trace\x00\x00\x00"
// layout. Shorter one-digit versions predate the v2 format and are refused.
func parseGoTraceHeader(data []byte) (int, bool) {
	if len(data) < 16 || string(data[:5]) != "go 1." || string(data[7:16]) != " trace\x00\x00\x00" {
		return 0, false
	}
	hi, lo := data[5], data[6]
	if hi < '1' || hi > '9' || lo < '0' || lo > '9' {
		return 0, false
	}
	return int(hi-'0')*10 + int(lo-'0'), true
}

// goTraceHooks are test-only observation and injection points. Production
// passes nil.
type goTraceHooks struct {
	afterLstat     func()                // after Lstat of Tool, before Open
	afterBuildinfo func()                // after buildinfo.Read, before the copy
	copyChunk      func()                // before each copied chunk
	afterCopy      func(exe string)      // after the private copy is closed
	afterStart     func(cmd *exec.Cmd)   // after a successful Start
	removeAll      func(string) error    // replaces os.RemoveAll
	env            []string              // appended to the child environment (raw runner only)
	observe        func(r *goTraceRun)   // after classification
	childCtx       func(context.Context) // the child context, for cause assertions

	// closeFile replaces f.Close for the descriptor named "tool", "copy"
	// or "input".
	closeFile func(name string, f *os.File) error

	// writeInput replaces f.Write for the input file.
	writeInput func(f *os.File, b []byte) (int, error)
}

// close closes f through closeFile when it is set.
func (h *goTraceHooks) close(name string, f *os.File) error {
	if h != nil && h.closeFile != nil {
		return h.closeFile(name, f)
	}
	return f.Close()
}

// write writes b to f through writeInput when it is set.
func (h *goTraceHooks) write(f *os.File, b []byte) (int, error) {
	if h != nil && h.writeInput != nil {
		return h.writeInput(f, b)
	}
	return f.Write(b)
}

// goTraceStarts counts attempted child starts.
var goTraceStarts atomic.Int64

// Validate fully decodes data with the admitted decoder and discards the
// decoded events. It returns a non-nil *GoTraceDecode with a non-nil error
// only when the decode succeeded and cleanup failed (ErrGoTraceCleanup).
// A nil or zero decoder returns ErrGoTraceDecoderUnselected.
func (d *GoTraceDecoder) Validate(ctx context.Context, data []byte, p GoProducer) (*GoTraceDecode, error) {
	if d == nil {
		return nil, ErrGoTraceDecoderUnselected
	}
	r, _, err := d.decode(ctx, data, p, goTraceAdmissions, false, nil)
	return r, err
}

// Analyze is Validate followed by ReadGoTraceDebug over the decoder output,
// bounded by limits. The returned GoTrace.Decoder holds the receipt and
// ClockStatus stays "unavailable"; QualifyGoMonotonic remains a separate
// step. Like Validate, it returns a result with ErrGoTraceCleanup when only
// cleanup failed.
func (d *GoTraceDecoder) Analyze(ctx context.Context, data []byte, p GoProducer, identity Record, limits Limits) (*GoTrace, error) {
	if d == nil {
		return nil, ErrGoTraceDecoderUnselected
	}
	r, out, err := d.decode(ctx, data, p, goTraceAdmissions, true, nil)
	if r == nil {
		return nil, err
	}
	trace, perr := ReadGoTraceDebug(bytes.NewReader(out), identity, limits)
	if perr != nil {
		return nil, errors.Join(fmt.Errorf("read decoded go trace: %w", perr), err)
	}
	trace.InputSHA256 = r.InputSHA256
	trace.Backend = "cmd/trace " + r.ToolGoVersion + " admission " + r.Admission
	trace.Decoder = r
	return trace, err
}

// decode runs the pure prechecks P0-P7 and then the file system steps. table
// and h are seams for tests; production passes goTraceAdmissions and nil.
func (d *GoTraceDecoder) decode(ctx context.Context, data []byte, p GoProducer, table []goTraceAdmission, keep bool, h *goTraceHooks) (*GoTraceDecode, []byte, error) {
	// P0-P7: no file system call, no child.
	if d == nil {
		return nil, nil, ErrGoTraceDecoderUnselected
	}
	l, err := resolveGoTraceLimits(d.Limits)
	if err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, fmt.Errorf("go trace decode: %w", err)
	}
	if d.Tool == "" || d.Scratch == "" {
		return nil, nil, ErrGoTraceDecoderUnselected
	}
	if !filepath.IsAbs(d.Tool) || filepath.Clean(d.Tool) != d.Tool {
		return nil, nil, fmt.Errorf("%w: tool path must be absolute and clean", ErrGoTraceDecoderUnselected)
	}
	if !filepath.IsAbs(d.Scratch) || filepath.Clean(d.Scratch) != d.Scratch {
		return nil, nil, fmt.Errorf("%w: path must be absolute and clean", ErrGoTraceScratch)
	}
	if int64(len(data)) > l.InputBytes {
		return nil, nil, fmt.Errorf("%w: input %d bytes", ErrGoTraceLimit, len(data))
	}
	header, ok := parseGoTraceHeader(data)
	if !ok {
		return nil, nil, ErrGoTraceHeader
	}
	if p.GoVersion == "" || p.GOOS == "" || p.GOARCH == "" {
		return nil, nil, fmt.Errorf("%w: producer incomplete", ErrGoTraceUnadmitted)
	}
	if err := validateGoTraceAdmissions(table, encodeGoTraceAdmission); err != nil {
		return nil, nil, fmt.Errorf("%w: invalid admission table: %w", ErrGoTraceUnadmitted, err)
	}
	var candidates []goTraceAdmission
	for _, r := range table {
		if r.Header == header && r.Producer == p {
			candidates = append(candidates, r)
		}
	}
	if len(candidates) == 0 {
		return nil, nil, ErrGoTraceUnadmitted
	}

	// P8 onward: the file system steps.
	st, err := os.Lstat(d.Scratch)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrGoTraceScratch, err)
	}
	if !st.IsDir() || st.Mode().Perm()&0o077 != 0 {
		return nil, nil, fmt.Errorf("%w: %s must be a directory with mode 0700 or stricter", ErrGoTraceScratch, d.Scratch)
	}
	dir, err := os.MkdirTemp(d.Scratch, "gotrace-")
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrGoTraceCustody, err)
	}
	r, out, err := d.decodeIn(ctx, dir, data, header, p, candidates, l, keep, h)
	removeAll := os.RemoveAll
	if h != nil && h.removeAll != nil {
		removeAll = h.removeAll
	}
	if cerr := removeAll(dir); cerr != nil {
		cleanup := fmt.Errorf("%w: %s: %w", ErrGoTraceCleanup, dir, cerr)
		if r != nil {
			r.Cleanup = "failed: " + dir + ": " + cerr.Error()
		}
		return r, out, errors.Join(err, cleanup)
	}
	if r != nil {
		r.Cleanup = "removed"
	}
	return r, out, err
}

// decodeIn runs P8 onward inside the private per-call directory dir.
func (d *GoTraceDecoder) decodeIn(ctx context.Context, dir string, data []byte, header int, p GoProducer, candidates []goTraceAdmission, l GoTraceLimits, keep bool, h *goTraceHooks) (*GoTraceDecode, []byte, error) {
	if h == nil {
		h = new(goTraceHooks)
	}
	st, err := os.Lstat(dir)
	if err != nil || !st.IsDir() || st.Mode().Perm() != 0o700 {
		return nil, nil, fmt.Errorf("%w: private directory %s", ErrGoTraceCustody, dir)
	}
	exe := filepath.Join(dir, "trace")
	rec, bi, err := copyGoTraceTool(ctx, d.Tool, exe, candidates, l, h)
	if err != nil {
		return nil, nil, err
	}
	if h.afterCopy != nil {
		h.afterCopy(exe)
	}
	if err := verifyGoTraceCopy(exe, rec); err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, fmt.Errorf("go trace decode: %w", err)
	}
	input := sha256.Sum256(data)
	if err := writeGoTraceInput(filepath.Join(dir, "runtime.trace"), data, h); err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, fmt.Errorf("go trace decode: %w", err)
	}
	run, err := runGoTraceTool(ctx, dir, exe, l, keep, h)
	if err != nil {
		return nil, nil, err
	}
	return &GoTraceDecode{
		Admission:     rec.ID,
		ToolSHA256:    rec.ToolSHA256,
		ToolBytes:     rec.ToolBytes,
		ToolPath:      bi.Path,
		ToolGoVersion: bi.GoVersion,
		ToolGOOS:      rec.ToolGOOS,
		ToolGOARCH:    rec.ToolGOARCH,
		RecipeSHA256:  rec.RecipeSHA256,
		Receipt:       rec.Receipt,
		Review:        rec.Review,
		Header:        header,
		Producer:      p,
		InputSHA256:   hex.EncodeToString(input[:]),
		InputBytes:    int64(len(data)),
		OutputBytes:   run.stdoutBytes,
		StderrBytes:   run.stderrBytes,
	}, run.stdout, nil
}

func buildSetting(bi *buildinfo.BuildInfo, key string) string {
	for _, s := range bi.Settings {
		if s.Key == key {
			return s.Value
		}
	}
	return ""
}

// copyGoTraceTool opens Tool once, admits it by buildinfo and digest, and
// copies those same bytes into exe. It returns the one selected record.
// After the copy, the descriptor must still report the same file, size,
// mtime and mode, and Tool must still name that file.
//
// The tool descriptor is closed exactly once, on return. A close error is
// joined after any primary error, and on success it turns the result into
// a custody failure with a zero record and nil buildinfo.
func copyGoTraceTool(ctx context.Context, tool, exe string, candidates []goTraceAdmission, l GoTraceLimits, h *goTraceHooks) (rec goTraceAdmission, bi *buildinfo.BuildInfo, err error) {
	var none goTraceAdmission
	lst, err := os.Lstat(tool)
	if err != nil {
		return none, nil, fmt.Errorf("%w: %w", ErrGoTraceCustody, err)
	}
	if !lst.Mode().IsRegular() {
		return none, nil, fmt.Errorf("%w: %s is not a regular file", ErrGoTraceCustody, tool)
	}
	if h.afterLstat != nil {
		h.afterLstat()
	}
	f, err := os.Open(tool)
	if err != nil {
		return none, nil, fmt.Errorf("%w: %w", ErrGoTraceCustody, err)
	}
	defer func() {
		if cerr := h.close("tool", f); cerr != nil {
			rec, bi = none, nil
			err = errors.Join(err, fmt.Errorf("%w: close tool: %w", ErrGoTraceCustody, cerr))
		}
	}()
	st1, err := f.Stat()
	if err != nil {
		return none, nil, fmt.Errorf("%w: %w", ErrGoTraceCustody, err)
	}
	if !os.SameFile(lst, st1) || !st1.Mode().IsRegular() {
		return none, nil, fmt.Errorf("%w: %s changed while opening", ErrGoTraceCustody, tool)
	}
	size := st1.Size()
	if size <= 0 || size > l.ToolBytes {
		return none, nil, fmt.Errorf("%w: tool %d bytes", ErrGoTraceLimit, size)
	}
	bi, err = buildinfo.Read(io.NewSectionReader(f, 0, size))
	if err != nil {
		return none, nil, fmt.Errorf("%w: buildinfo: %w", ErrGoTraceUnadmitted, err)
	}
	goos, goarch := buildSetting(bi, "GOOS"), buildSetting(bi, "GOARCH")
	var matched []goTraceAdmission
	for _, r := range candidates {
		if r.ToolPath == bi.Path && r.ToolGoVersion == bi.GoVersion && r.ToolGOOS == goos && r.ToolGOARCH == goarch {
			matched = append(matched, r)
		}
	}
	if len(matched) == 0 {
		return none, nil, fmt.Errorf("%w: decoder build %s %s %s/%s", ErrGoTraceUnadmitted, bi.Path, bi.GoVersion, goos, goarch)
	}
	if h.afterBuildinfo != nil {
		h.afterBuildinfo()
	}
	if err := ctx.Err(); err != nil {
		return none, nil, fmt.Errorf("go trace decode: %w", err)
	}
	dst, err := os.OpenFile(exe, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o700)
	if err != nil {
		return none, nil, fmt.Errorf("%w: %w", ErrGoTraceCustody, err)
	}
	digest := sha256.New()
	n, err := copyGoTraceChunks(ctx, io.MultiWriter(dst, digest), io.NewSectionReader(f, 0, size+1), h)
	if cerr := h.close("copy", dst); cerr != nil {
		err = errors.Join(err, fmt.Errorf("%w: close copy: %w", ErrGoTraceCustody, cerr))
	}
	if err != nil {
		return none, nil, err
	}
	if n != size {
		return none, nil, fmt.Errorf("%w: copied %d of %d bytes", ErrGoTraceCustody, n, size)
	}
	st2, err := f.Stat()
	if err != nil {
		return none, nil, fmt.Errorf("%w: stat tool after copy: %w", ErrGoTraceCustody, err)
	}
	lst2, err := os.Lstat(tool)
	if err != nil {
		return none, nil, fmt.Errorf("%w: lstat tool after copy: %w", ErrGoTraceCustody, err)
	}
	if !os.SameFile(st1, st2) || !os.SameFile(lst, lst2) || st2.Size() != size || !st2.ModTime().Equal(st1.ModTime()) || st2.Mode() != st1.Mode() {
		return none, nil, fmt.Errorf("%w: %s changed during copy", ErrGoTraceCustody, tool)
	}
	sum := hex.EncodeToString(digest.Sum(nil))
	for _, r := range matched {
		if r.ToolSHA256 == sum && r.ToolBytes == size {
			return r, bi, nil
		}
	}
	return none, nil, fmt.Errorf("%w: decoder sha256 %s", ErrGoTraceUnadmitted, sum)
}

// copyGoTraceChunks copies in 1 MiB chunks, checking ctx between them.
func copyGoTraceChunks(ctx context.Context, w io.Writer, r io.Reader, h *goTraceHooks) (int64, error) {
	buf := make([]byte, 1<<20)
	var n int64
	for {
		if h.copyChunk != nil {
			h.copyChunk()
		}
		if err := ctx.Err(); err != nil {
			return n, fmt.Errorf("go trace decode: %w", err)
		}
		m, err := io.ReadFull(r, buf)
		if m > 0 {
			if _, werr := w.Write(buf[:m]); werr != nil {
				return n, fmt.Errorf("%w: copy: %w", ErrGoTraceCustody, werr)
			}
			n += int64(m)
		}
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return n, nil
		}
		if err != nil {
			return n, fmt.Errorf("%w: copy: %w", ErrGoTraceCustody, err)
		}
	}
}

// verifyGoTraceCopy re-hashes the private copy at rest.
func verifyGoTraceCopy(exe string, rec goTraceAdmission) error {
	lst, err := os.Lstat(exe)
	if err != nil || !lst.Mode().IsRegular() || lst.Mode().Perm() != 0o700 {
		return fmt.Errorf("%w: private copy is not a 0700 regular file", ErrGoTraceCustody)
	}
	f, err := os.Open(exe)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrGoTraceCustody, err)
	}
	digest := sha256.New()
	n, err := io.Copy(digest, f)
	st, serr := f.Stat()
	cerr := f.Close()
	if err = errors.Join(err, serr, cerr); err != nil {
		return fmt.Errorf("%w: verify copy: %w", ErrGoTraceCustody, err)
	}
	if n != rec.ToolBytes || st.Size() != rec.ToolBytes || !os.SameFile(lst, st) || hex.EncodeToString(digest.Sum(nil)) != rec.ToolSHA256 {
		return fmt.Errorf("%w: private copy changed", ErrGoTraceCustody)
	}
	return nil
}

// writeGoTraceInput writes data to a new file at path and closes it once,
// reporting both the write and the close error. A short write is an error.
func writeGoTraceInput(path string, data []byte, h *goTraceHooks) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrGoTraceCustody, err)
	}
	n, werr := h.write(f, data)
	if werr == nil && n != len(data) {
		werr = io.ErrShortWrite
	}
	if err := errors.Join(werr, h.close("input", f)); err != nil {
		return fmt.Errorf("%w: write input: %w", ErrGoTraceCustody, err)
	}
	return nil
}

// goTraceRun is the raw runner's observation of one child.
type goTraceRun struct {
	stdout      []byte
	stderr      []byte
	stdoutBytes int64
	stderrBytes int64
	overflow    string // stream that overflowed first, if any
	state       *os.ProcessState
	waitErr     error
}

// goTraceOverflow is shared by both output writers of one child.
type goTraceOverflow struct {
	mu     sync.Mutex
	stream string
	once   sync.Once
	cancel context.CancelCauseFunc
}

// goTraceWriter counts a stream, keeps at most limit bytes when keep is set,
// and cancels the child on the first write that would exceed limit.
type goTraceWriter struct {
	stream string
	limit  int64
	n      int64
	keep   bool
	buf    bytes.Buffer
	o      *goTraceOverflow
}

func (w *goTraceWriter) Write(p []byte) (int, error) {
	w.o.mu.Lock()
	defer w.o.mu.Unlock()
	if w.o.stream != "" {
		return 0, errOutputLimit
	}
	if int64(len(p)) > w.limit-w.n {
		w.o.stream = w.stream
		w.o.once.Do(func() { w.o.cancel(errOutputLimit) })
		return 0, errOutputLimit
	}
	w.n += int64(len(p))
	if w.keep {
		w.buf.Write(p)
	}
	return len(p), nil
}

// The closed allowlist of structural diagnostics printed by the admitted
// cmd/trace -d=parsed (internal/trace batch and generation framing). Each is
// a full match against the single stderr line.
var goTraceStructural = []*regexp.Regexp{
	regexp.MustCompile(`^expected batch event, got event [0-9]+$`),                                                                        // S1
	regexp.MustCompile(`^invalid batch size [0-9]+, maximum is 65536$`),                                                                   // S2
	regexp.MustCompile(`^error reading batch (gen|M ID|timestamp|size): (EOF|unexpected EOF|binary: varint overflows a 64-bit integer)$`), // S3
	regexp.MustCompile(`^incomplete generation found; trace likely truncated$`),                                                           // S5
	regexp.MustCompile(`^invalid generation number [0-9]+$`),                                                                              // S6
	regexp.MustCompile(`^broken trace: missing end-of-generation event, or generations are interleaved$`),                                 // S7
	regexp.MustCompile(`^no frequency event found$`),                                                                                      // S8
	regexp.MustCompile(`^no clock snapshot event found$`),                                                                                 // S9
}

// structuralDiagnostic reports whether stderr is exactly one line ending in
// one newline that fully matches an allowlist entry.
func structuralDiagnostic(stderr []byte) (string, bool) {
	line, ok := bytes.CutSuffix(stderr, []byte("\n"))
	if !ok || bytes.ContainsAny(line, "\r\n") {
		return "", false
	}
	for _, re := range goTraceStructural {
		if re.Match(line) {
			return string(line), true
		}
	}
	return "", false
}

func firstLine(b []byte) string {
	s, _, _ := strings.Cut(string(b), "\n")
	return s
}

// runGoTraceTool runs exe -d=parsed runtime.trace once in dir and classifies
// the outcome. It is the only place a child is started.
func runGoTraceTool(ctx context.Context, dir, exe string, limits GoTraceLimits, keep bool, h *goTraceHooks) (*goTraceRun, error) {
	l, err := resolveGoTraceLimits(limits)
	if err != nil {
		return nil, err
	}
	if h == nil {
		h = new(goTraceHooks)
	}
	parent, cancelCause := context.WithCancelCause(ctx)
	defer cancelCause(nil)
	o := &goTraceOverflow{cancel: cancelCause}
	stdout := &goTraceWriter{stream: "stdout", limit: l.OutputBytes, keep: keep, o: o}
	stderr := &goTraceWriter{stream: "stderr", limit: l.StderrBytes, keep: true, o: o}
	childCtx, stop := context.WithTimeoutCause(parent, l.Timeout, errChildTime)
	defer stop()
	if h.childCtx != nil {
		h.childCtx(childCtx)
	}
	cmd := exec.CommandContext(childCtx, exe, "-d=parsed", "runtime.trace")
	cmd.Dir = dir
	cmd.Env = append([]string{"TMPDIR=" + dir}, h.env...)
	cmd.Stdin = nil
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.WaitDelay = l.WaitDelay
	goTraceStarts.Add(1)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrGoTraceStart, err)
	}
	if h.afterStart != nil {
		h.afterStart(cmd)
	}
	waitErr := cmd.Wait()
	o.mu.Lock()
	run := &goTraceRun{
		stdout:      stdout.buf.Bytes(),
		stderr:      stderr.buf.Bytes(),
		stdoutBytes: stdout.n,
		stderrBytes: stderr.n,
		overflow:    o.stream,
		state:       cmd.ProcessState,
		waitErr:     waitErr,
	}
	o.mu.Unlock()
	if h.observe != nil {
		h.observe(run)
	}
	return run, classifyGoTraceRun(ctx, childCtx, run)
}

// classifyGoTraceRun maps one child outcome to an error class. The first
// cancellation cause wins.
func classifyGoTraceRun(ctx, childCtx context.Context, run *goTraceRun) error {
	cause := context.Cause(childCtx)
	if cause == errOutputLimit {
		return fmt.Errorf("%w: %s: %w", ErrGoTraceLimit, run.overflow, errOutputLimit)
	}
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("go trace decode: %w", err)
	}
	if cause == errChildTime {
		if run.overflow != "" {
			return fmt.Errorf("%w: %w (then %s overflowed)", ErrGoTraceLimit, errChildTime, run.overflow)
		}
		return fmt.Errorf("%w: %w", ErrGoTraceLimit, errChildTime)
	}
	if errors.Is(run.waitErr, exec.ErrWaitDelay) {
		return fmt.Errorf("%w: descendant may remain: %w", ErrGoTraceReap, run.waitErr)
	}
	var exit *exec.ExitError
	if errors.As(run.waitErr, &exit) {
		switch exit.ExitCode() {
		case -1:
			return fmt.Errorf("%w: %v", ErrGoTraceSignal, exit)
		case 1:
			if line, ok := structuralDiagnostic(run.stderr); ok {
				return fmt.Errorf("%w: %s", ErrGoTraceMalformed, line)
			}
		}
		return fmt.Errorf("%w: %v: %s", ErrGoTraceDecoderFailed, exit, firstLine(run.stderr))
	}
	if run.waitErr != nil {
		return fmt.Errorf("%w: %w", ErrGoTraceReap, run.waitErr)
	}
	if run.stderrBytes != 0 {
		return fmt.Errorf("%w: exit 0 with stderr: %s", ErrGoTraceDecoderFailed, firstLine(run.stderr))
	}
	return nil
}
