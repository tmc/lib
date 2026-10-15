package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/nsightsystems"
)

// sqliteProcess is per invocation: tests can exercise failures without a child
// process, while the command uses exec.CommandContext with fixed arguments.
type sqliteProcess func(context.Context, string, []string, io.Writer, io.Writer) error

func sqliteCommand(ctx context.Context, tool string, args []string, stdout, stderr io.Writer) error {
	cmd := exec.CommandContext(ctx, tool, args...)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.WaitDelay = 200 * time.Millisecond
	return cmd.Run()
}

type nsysOptions struct {
	release, build, clock, expectedSHA string
	shift                              int64
	timeout                            time.Duration
	inputBytes                         int64
	bytes, rows                        int
}

func (o nsysOptions) valid() error {
	if o.release != "2024.2" && o.release != "2025.3" {
		return errors.New("specify supported -nsys-release (2024.2 or 2025.3)")
	}
	if strings.TrimSpace(o.build) == "" {
		return errors.New("specify -nsys-build with the exporter build")
	}
	if o.clock != "report-relative" && o.clock != "unix-epoch" && o.clock != "unestablished" {
		return errors.New("specify -nsys-clock (report-relative, unix-epoch, or unestablished)")
	}
	if o.timeout <= 0 || o.timeout > 5*time.Minute || o.inputBytes < 1 || o.inputBytes > 64<<30 || o.bytes < 1 || o.bytes > 64<<20 || o.rows < 1 || o.rows > 262144 {
		return errors.New("invalid Nsight resource limits")
	}
	if o.expectedSHA != "" {
		b, err := hex.DecodeString(o.expectedSHA)
		if err != nil || len(b) != sha256.Size || strings.ToLower(o.expectedSHA) != o.expectedSHA {
			return errors.New("invalid -nsys-sha256")
		}
	}
	return nil
}

var errSQLiteLimit = errors.New("sqlite output limit exceeded")

type boundedOutput struct {
	buffer   bytes.Buffer
	left     int
	cancel   context.CancelFunc
	exceeded bool
}

func (b *boundedOutput) Len() int       { return b.buffer.Len() }
func (b *boundedOutput) Bytes() []byte  { return b.buffer.Bytes() }
func (b *boundedOutput) String() string { return b.buffer.String() }

func (b *boundedOutput) Write(p []byte) (int, error) {
	if len(p) > b.left {
		n, _ := b.buffer.Write(p[:b.left])
		b.left = 0
		b.exceeded = true
		b.cancel()
		return n, errSQLiteLimit
	}
	b.left -= len(p)
	return b.buffer.Write(p)
}

type population struct{ Full, Selected int64 }

type sqliteReader struct {
	tool, path, dir, original, digest string
	info                              os.FileInfo
	process                           sqliteProcess
	acquiredErrors                    []error
	window                            window
	maxKernels, bytesLeft, rowsLeft   int
	populations                       map[string]population
	keys                              map[string][]int64
}

// newSQLiteReader copies a quiescent SQLite export into a private, immutable
// read-only snapshot. Live WAL/journal databases are refused; this is not a
// SQLite backup protocol or an attestation of the declared exporter build.
func newSQLiteReader(ctx context.Context, path string, o nsysOptions, process sqliteProcess) (_ *sqliteReader, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	tool, err := exec.LookPath("sqlite3")
	if err != nil {
		return nil, err
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	if err := noSQLiteJournal(path); err != nil {
		return nil, err
	}
	// Refuse existing non-regular paths before opening: opening a FIFO can
	// block before a context-aware copy gets a chance to check cancellation.
	observed, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !observed.Mode().IsRegular() || observed.Size() > o.inputBytes {
		return nil, errors.New("SQLite input must be a regular file within -nsys-input-bytes")
	}
	in, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var dir string
	defer func() {
		err = errors.Join(err, in.Close())
		if err != nil && dir != "" {
			err = errors.Join(err, os.RemoveAll(dir))
		}
	}()
	before, err := in.Stat()
	if err != nil {
		return nil, err
	}
	if !before.Mode().IsRegular() || before.Size() > o.inputBytes || !os.SameFile(observed, before) {
		return nil, errors.New("SQLite input must be a regular file within -nsys-input-bytes")
	}
	dir, err = os.MkdirTemp("", "profile2trace-")
	if err != nil {
		return nil, err
	}
	snapshot := filepath.Join(dir, "capture.sqlite")
	out, err := os.OpenFile(snapshot, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return nil, err
	}
	h := sha256.New()
	n, copyErr := copyContext(ctx, io.MultiWriter(out, h), io.LimitReader(in, o.inputBytes+1))
	if err := errors.Join(copyErr, out.Close()); err != nil {
		return nil, err
	}
	if n != before.Size() || n > o.inputBytes {
		return nil, errors.New("SQLite source changed size during snapshot")
	}
	digest := hex.EncodeToString(h.Sum(nil))
	if o.expectedSHA != "" && o.expectedSHA != digest {
		return nil, errors.New("SQLite original digest differs from -nsys-sha256")
	}
	// Re-read the original handle and pathname, rather than labeling an expected
	// hash as observed. A concurrently changing or replaced source is refused.
	if _, err := in.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	h.Reset()
	if _, err := copyContext(ctx, h, io.LimitReader(in, o.inputBytes+1)); err != nil {
		return nil, err
	}
	after, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !os.SameFile(before, after) || before.Size() != after.Size() || before.ModTime() != after.ModTime() || digest != hex.EncodeToString(h.Sum(nil)) {
		return nil, errors.New("SQLite original changed during snapshot")
	}
	if err := noSQLiteJournal(path); err != nil {
		return nil, err
	}
	f, err := os.Open(snapshot)
	if err != nil {
		return nil, err
	}
	var header [16]byte
	_, readErr := io.ReadFull(f, header[:])
	if err := errors.Join(readErr, f.Close()); err != nil {
		return nil, err
	}
	if string(header[:]) != "SQLite format 3\x00" {
		return nil, errors.New("input is not a SQLite database")
	}
	if err := os.Chmod(snapshot, 0444); err != nil {
		return nil, err
	}
	info, err := os.Stat(snapshot)
	if err != nil {
		return nil, err
	}
	return &sqliteReader{tool: tool, path: snapshot, dir: dir, original: path, digest: digest, info: info, process: process, bytesLeft: o.bytes, rowsLeft: o.rows, populations: make(map[string]population), keys: make(map[string][]int64)}, nil
}

func noSQLiteJournal(path string) error {
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		if _, err := os.Lstat(path + suffix); err == nil {
			return errors.New("SQLite input has a live journal or WAL sidecar")
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

func copyContext(ctx context.Context, out io.Writer, in io.Reader) (int64, error) {
	buf := make([]byte, 32<<10)
	var total int64
	for {
		if err := ctx.Err(); err != nil {
			return total, err
		}
		n, err := in.Read(buf)
		if n > 0 {
			written, e := out.Write(buf[:n])
			total += int64(written)
			if e != nil {
				return total, e
			}
			if written != n {
				return total, io.ErrShortWrite
			}
		}
		if err == io.EOF {
			return total, nil
		}
		if err != nil {
			return total, err
		}
	}
}

func (r *sqliteReader) stable() error {
	info, err := os.Stat(r.path)
	if err != nil {
		return err
	}
	if !os.SameFile(r.info, info) || info.Size() != r.info.Size() || info.ModTime() != r.info.ModTime() {
		return errors.New("SQLite snapshot changed")
	}
	return nil
}

func (r *sqliteReader) verify(ctx context.Context) (err error) {
	if err := r.stable(); err != nil {
		return err
	}
	f, err := os.Open(r.path)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, f.Close()) }()
	h := sha256.New()
	if _, err := copyContext(ctx, h, f); err != nil {
		return err
	}
	if hex.EncodeToString(h.Sum(nil)) != r.digest {
		return errors.New("SQLite snapshot digest changed")
	}
	return nil
}

// query accepts only SQL assembled from the fixed table allowlist and numeric
// limits below. Neither report text nor a report-provided command is executed.
func (r *sqliteReader) query(ctx context.Context, query string, limit int) ([]json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := r.stable(); err != nil {
		return nil, err
	}
	child, cancel := context.WithCancel(ctx)
	defer cancel()
	stdout := &boundedOutput{left: r.bytesLeft, cancel: cancel}
	stderr := &boundedOutput{left: 64 << 10, cancel: cancel}
	uri := (&url.URL{Scheme: "file", Path: r.path, RawQuery: "mode=ro&immutable=1"}).String()
	args := []string{"-readonly", "-batch", "-init", "/dev/null", "-json", "-cmd", "PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;", "--", uri, query}
	processErr := r.process(child, r.tool, args, stdout, stderr)
	if processErr != nil {
		r.acquiredErrors = append(r.acquiredErrors, processErr)
	}
	r.bytesLeft -= stdout.Len()
	var limitErr error
	if stdout.exceeded || stderr.exceeded {
		limitErr = errSQLiteLimit
		r.acquiredErrors = append(r.acquiredErrors, limitErr)
	}
	if err := errors.Join(processErr, ctx.Err(), limitErr); err != nil {
		return nil, fmt.Errorf("sqlite3: %w: %s", err, stderr.String())
	}
	if err := r.stable(); err != nil {
		return nil, err
	}
	data := bytes.TrimSpace(stdout.Bytes())
	if len(data) == 0 {
		return nil, nil
	}
	d := json.NewDecoder(bytes.NewReader(data))
	token, err := d.Token()
	if err != nil {
		return nil, err
	}
	if token != json.Delim('[') {
		return nil, errors.New("sqlite JSON must be an array")
	}
	var rows []json.RawMessage
	for d.More() {
		if len(rows) >= limit {
			return nil, errors.New("SQLite row limit exceeded")
		}
		var row json.RawMessage
		if err := d.Decode(&row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return nil, errors.New("sqlite JSON has trailing content")
	}
	return rows, nil
}

var nsysTables = map[string]bool{
	"StringIds": false, "META_DATA_EXPORT": false, "TARGET_INFO_SESSION_START_TIME": false,
	"CUPTI_ACTIVITY_KIND_KERNEL": true, "CUPTI_ACTIVITY_KIND_MEMCPY": true, "CUPTI_ACTIVITY_KIND_MEMSET": true, "CUPTI_ACTIVITY_KIND_RUNTIME": true, "NVTX_EVENTS": true, "OSRT_API": true,
	"DIAGNOSTIC_EVENT": false, "ENUM_DIAGNOSTIC_SEVERITY_LEVEL": false,
}

func (r *sqliteReader) ReadTable(ctx context.Context, name string, limit int) (nsightsystems.Table, error) {
	event, ok := nsysTables[name]
	if !ok {
		return nsightsystems.Table{}, errors.New("unsupported SQLite table")
	}
	if limit < 0 || limit > 262144 {
		return nsightsystems.Table{}, errors.New("invalid SQLite row limit")
	}
	exists, err := r.query(ctx, fmt.Sprintf(`SELECT type,sql FROM sqlite_schema WHERE name='%s' LIMIT 2`, name), 2)
	if err != nil {
		return nsightsystems.Table{}, err
	}
	if len(exists) == 0 {
		return nsightsystems.Table{}, nsightsystems.ErrUnavailable
	}
	var declaration struct{ Type, SQL string }
	if len(exists) != 1 {
		return nsightsystems.Table{}, errors.New("ambiguous SQLite table")
	}
	if err := json.Unmarshal(exists[0], &declaration); err != nil {
		return nsightsystems.Table{}, err
	}
	if declaration.Type != "table" || strings.Contains(strings.ToUpper(declaration.SQL), "CREATE VIRTUAL") {
		return nsightsystems.Table{}, errors.New("SQLite table must be an ordinary table")
	}
	schema, err := r.query(ctx, fmt.Sprintf(`PRAGMA table_info("%s")`, name), 64)
	if err != nil {
		return nsightsystems.Table{}, err
	}
	t := nsightsystems.Table{DatabaseSHA256: r.digest}
	for _, raw := range schema {
		var c struct {
			Name, Type string
			Notnull    int
		}
		if err := json.Unmarshal(raw, &c); err != nil {
			return t, err
		}
		t.Columns = append(t.Columns, nsightsystems.Column{Name: c.Name, Type: c.Type, Nullable: c.Notnull == 0})
	}
	where := ""
	if event && !r.window.unbounded() {
		where = fmt.Sprintf(" WHERE start >= %d AND start < %d", r.window.lo, r.window.hi)
	}
	counts, err := r.query(ctx, fmt.Sprintf(`SELECT (SELECT count(*) FROM "%s") AS Full,count(*) AS Selected FROM "%s"%s`, name, name, where), 1)
	if err != nil {
		return t, err
	}
	if len(counts) != 1 {
		return t, errors.New("SQLite population unavailable")
	}
	var p population
	if err := json.Unmarshal(counts[0], &p); err != nil {
		return t, err
	}
	if p.Full < 0 || p.Selected < 0 || p.Selected > p.Full {
		return t, errors.New("invalid SQLite population")
	}
	r.populations[name] = p
	if name == "CUPTI_ACTIVITY_KIND_KERNEL" && (p.Selected == 0 || p.Selected > int64(r.maxKernels)) {
		return t, fmt.Errorf("window %s selects %d of %d kernel rows; check -max-slices", r.window, p.Selected, p.Full)
	}
	if p.Selected > int64(limit) || p.Selected > int64(r.rowsLeft) {
		return t, errors.New("SQLite row limit exceeded")
	}
	rows, err := r.query(ctx, fmt.Sprintf(`SELECT rowid AS "__profile2trace_rowid",* FROM "%s"%s ORDER BY rowid LIMIT %d`, name, where, limit+1), limit+1)
	if err != nil {
		return t, err
	}
	if int64(len(rows)) != p.Selected {
		return t, errors.New("SQLite population changed or truncated")
	}
	for _, raw := range rows {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(raw, &m); err != nil {
			return t, err
		}
		var id int64
		if err := json.Unmarshal(m["__profile2trace_rowid"], &id); err != nil {
			return t, err
		}
		keys := r.keys[name]
		if len(keys) > 0 && id <= keys[len(keys)-1] {
			return t, errors.New("invalid SQLite source row key")
		}
		r.keys[name] = append(keys, id)
		delete(m, "__profile2trace_rowid")
		projected, err := json.Marshal(m)
		if err != nil {
			return t, err
		}
		t.Rows = append(t.Rows, projected)
	}
	r.rowsLeft -= len(t.Rows)
	return t, nil
}

type nsysLog struct {
	out io.Writer
	err error
}

func (l *nsysLog) Write(p []byte) (int, error) {
	if l.err != nil {
		return 0, l.err
	}
	n, err := l.out.Write(p)
	if err == nil && n != len(p) {
		err = io.ErrShortWrite
	}
	l.err = err
	return n, err
}

func readNSYS(ctx context.Context, path string, w window, max int, o nsysOptions, process sqliteProcess, log io.Writer) (_ []slice, err error) {
	if ctx == nil || process == nil {
		return nil, errors.New("Nsight context and process required")
	}
	if err := o.valid(); err != nil {
		return nil, err
	}
	diagnosticsLog := &nsysLog{out: log}
	log = diagnosticsLog
	defer func() { err = errors.Join(err, diagnosticsLog.err) }()
	ctx, cancel := context.WithTimeout(ctx, o.timeout)
	defer cancel()
	r, err := newSQLiteReader(ctx, path, o, process)
	if err != nil {
		return nil, err
	}
	defer func() { err = errors.Join(err, os.RemoveAll(r.dir)) }()
	r.window, r.maxKernels = w, max
	source := nsightsystems.Source{ToolVersion: o.release, ToolBuild: o.build, ReportPath: r.original, ReportSHA256: r.digest, DatabaseSHA256: r.digest, Clock: o.clock, ShiftNS: o.shift}
	report, err := nsightsystems.Import(ctx, r, source, nsightsystems.Limits{Bytes: o.bytes, Rows: o.rows})
	if err != nil {
		for _, acquiredErr := range r.acquiredErrors {
			if !errors.Is(err, acquiredErr) {
				err = errors.Join(err, acquiredErr)
			}
		}
		return nil, err
	}
	if _, ok := r.populations["CUPTI_ACTIVITY_KIND_KERNEL"]; !ok {
		return nil, errors.New("Nsight export has no kernel table")
	}
	if err := r.diagnostics(ctx, log); err != nil {
		return nil, err
	}
	var rows []slice
	index := make(map[string]int)
	omitted := 0
	for _, e := range report.Events {
		i := index[e.Table]
		index[e.Table]++
		if e.EndNS == nil {
			omitted++
			continue
		}
		if e.StartNS < 0 || *e.EndNS < e.StartNS {
			return nil, errors.New("invalid Nsight interval")
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(e.Raw, &fields); err != nil {
			return nil, err
		}
		value := func(key string) string {
			if b, ok := fields[key]; ok && string(b) != "null" {
				return string(b)
			}
			return "unavailable"
		}
		category := map[string]string{"CUPTI_ACTIVITY_KIND_KERNEL": "cuda_kernel", "CUPTI_ACTIVITY_KIND_MEMCPY": "cuda_memcpy", "CUPTI_ACTIVITY_KIND_MEMSET": "cuda_memset", "CUPTI_ACTIVITY_KIND_RUNTIME": "cuda_runtime", "NVTX_EVENTS": "nvtx", "OSRT_API": "osrt"}[e.Table]
		processName := "Nsight process-key unavailable"
		if e.Process != nil {
			processName = fmt.Sprintf("Nsight process-key %d", *e.Process)
		}
		thread := "global thread " + value("globalTid")
		if strings.HasPrefix(e.Table, "CUPTI_ACTIVITY_KIND_") && e.Table != "CUPTI_ACTIVITY_KIND_RUNTIME" {
			thread = "device " + value("deviceId") + " context " + value("contextId") + " stream " + value("streamId")
		}
		name := e.Name
		if name == "" {
			name = category + " name unavailable"
		}
		p := r.populations[e.Table]
		attrs := map[string]any{
			"basis":                  report.Basis,
			"source_kind":            "sqlite-export",
			"sqlite_cli_path":        r.tool,
			"source_report_path":     source.ReportPath,
			"source_report_sha256":   source.ReportSHA256,
			"snapshot_sha256":        source.DatabaseSHA256,
			"source_table":           e.Table,
			"source_rowid":           r.keys[e.Table][i],
			"source_raw_json":        string(e.Raw),
			"tool_release_declared":  o.release,
			"tool_build_declared":    o.build,
			"clock_domain_declared":  o.clock,
			"exporter_ts_shift_ns":   o.shift,
			"schema_version":         report.SchemaVersion,
			"selected_schema_sha256": report.SelectedSchemaSHA256,
			"window":                 w.String(),
			"table_full_rows":        p.Full,
			"table_selected_rows":    p.Selected,
			"capture_completeness":   "unestablished",
			"name_unavailable":       e.NameUnavailable,
		}
		rows = append(rows, slice{name, category, processName, thread, e.StartNS, *e.EndNS - e.StartNS, attrs})
	}
	if err := r.verify(ctx); err != nil {
		return nil, err
	}
	fmt.Fprintf(log, "Nsight source sqlite-export sha256=%s; declared release=%s build=%s clock=%s shift_ns=%d; schema=%s; window=%s; completeness unestablished\n", r.digest, o.release, o.build, o.clock, o.shift, report.SchemaVersion, w)
	for _, metadata := range report.Metadata {
		fmt.Fprintf(log, "Nsight export metadata: %s\n", metadata)
	}
	for _, unjoined := range report.Unjoined {
		fmt.Fprintf(log, "Nsight unjoined event %d: %s\n", unjoined.Event, unjoined.Reason)
	}
	for _, p := range report.Providers {
		n := r.populations[p.Table]
		fmt.Fprintf(log, "Nsight provider %s available=%t full_rows=%d selected_rows=%d reason=%s\n", p.Table, p.Available, n.Full, n.Selected, p.Reason)
	}
	fmt.Fprintf(log, "Nsight imported intervals: %d; non-interval records omitted: %d; unique correlations: %d; unjoined activities: %d\n", len(rows), omitted, len(report.Joins), len(report.Unjoined))
	return rows, nil
}

func (r *sqliteReader) diagnostics(ctx context.Context, log io.Writer) error {
	levels, err := r.ReadTable(ctx, "ENUM_DIAGNOSTIC_SEVERITY_LEVEL", r.rowsLeft)
	levelsAvailable := !errors.Is(err, nsightsystems.ErrUnavailable)
	if err != nil && levelsAvailable {
		return fmt.Errorf("read Nsight diagnostic levels: %w", err)
	}
	if levelsAvailable {
		if err := diagnosticColumns(levels, map[string]string{"id": "INTEGER", "name": "TEXT", "label": "TEXT"}); err != nil {
			return err
		}
	}
	type level struct {
		ID          *int64
		Name, Label *string
	}
	byID := make(map[int64]level)
	for _, raw := range levels.Rows {
		var l level
		if err := json.Unmarshal(raw, &l); err != nil {
			return err
		}
		if l.ID == nil {
			return errors.New("invalid Nsight diagnostic severity")
		}
		if _, ok := byID[*l.ID]; ok {
			return errors.New("duplicate Nsight diagnostic severity")
		}
		byID[*l.ID] = l
	}
	diagnostics, err := r.ReadTable(ctx, "DIAGNOSTIC_EVENT", r.rowsLeft)
	if errors.Is(err, nsightsystems.ErrUnavailable) {
		fmt.Fprintln(log, "Nsight diagnostics unavailable; loss status unestablished")
		return nil
	}
	if err != nil {
		return fmt.Errorf("read Nsight diagnostics: %w", err)
	}
	if err := diagnosticColumns(diagnostics, map[string]string{"timestamp": "INTEGER", "severity": "INTEGER", "text": "TEXT"}); err != nil {
		return err
	}
	for _, raw := range diagnostics.Rows {
		var d struct {
			Timestamp, Severity *int64
			Text                *string
		}
		if err := json.Unmarshal(raw, &d); err != nil {
			return err
		}
		if d.Timestamp == nil || d.Severity == nil || d.Text == nil {
			return errors.New("invalid Nsight diagnostic row")
		}
		name := fmt.Sprint(*d.Severity)
		if l, ok := byID[*d.Severity]; ok {
			if l.Name != nil && (*l.Name == "Info" || *l.Name == "Verbose") {
				continue
			}
			if l.Label != nil && *l.Label != "" {
				name = *l.Label
			}
		}
		fmt.Fprintf(log, "Nsight diagnostic (%s): %s\n", name, *d.Text)
	}
	fmt.Fprintf(log, "Nsight diagnostics available=true full_rows=%d selected_rows=%d severity_enum_available=%t; cover the original capture; loss/completeness are not inferred from message text\n", r.populations["DIAGNOSTIC_EVENT"].Full, r.populations["DIAGNOSTIC_EVENT"].Selected, levelsAvailable)
	return nil
}

// Diagnostics have no canonical importer profile. Require the fields used by
// this supplemental projection, allow additional fields, and make no claim
// about machine-readable loss semantics outside this projection.
func diagnosticColumns(t nsightsystems.Table, required map[string]string) error {
	columns := make(map[string]string)
	for _, c := range t.Columns {
		columns[c.Name] = c.Type
	}
	for name, kind := range required {
		if columns[name] != kind {
			return fmt.Errorf("unsupported Nsight diagnostic column %s", name)
		}
	}
	return nil
}
