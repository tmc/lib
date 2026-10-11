package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// RetentionConfig bounds complete streams in an exclusively task-owned directory.
// Owner is a stable, non-secret task identifier. The directory must be private
// to the calling user and must not be writable by other users.
type RetentionConfig struct {
	Owner          string `json:"owner"`
	MaxFiles       int    `json:"max_files"`
	MaxBytes       int64  `json:"max_bytes"`
	MaxStreamBytes int64  `json:"max_stream_bytes"`
}

func (c RetentionConfig) validate() error {
	if len(c.Owner) == 0 || len(c.Owner) > 128 || c.MaxFiles < 1 || c.MaxFiles > 1024 || c.MaxStreamBytes < 1 || c.MaxStreamBytes > 64<<20 || c.MaxBytes < c.MaxStreamBytes || c.MaxBytes > 1<<30 {
		return fmt.Errorf("invalid retention configuration")
	}
	for _, b := range c.Owner {
		if b < '!' || b > '~' {
			return fmt.Errorf("invalid retention owner")
		}
	}
	return nil
}

type retentionMarker struct {
	Version int    `json:"version"`
	Owner   string `json:"owner"`
}

// Retention stores strictly validated complete streams. Its methods serialize
// access; a directory lease refuses duplicate writers across processes. There
// is no automatic takeover of a crashed writer. Close releases only the lease.
type Retention struct {
	mu            sync.Mutex
	root          *os.Root
	cfg           RetentionConfig
	stores        uint64
	storeFailures uint64
	prunedFiles   uint64
	prunedBytes   uint64
	lastError     string
}

// OpenRetention opens or initializes a private task directory. It rejects foreign
// files, symbolic links, corrupt streams and another writer's lease. Changing
// budgets is checked before deleting any retained data (configuration rollback).
func OpenRetention(path string, cfg RetentionConfig) (*Retention, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if err := os.Mkdir(path, 0700); err != nil && !os.IsExist(err) {
		return nil, err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() || info.Mode()&0077 != 0 {
		return nil, fmt.Errorf("retention directory must be private")
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			root.Close()
		}
	}()
	entries, err := retentionEntries(root)
	if err != nil {
		return nil, err
	}
	marker, err := retentionMarkerBytes(root)
	if os.IsNotExist(err) {
		if len(entries) != 0 {
			return nil, fmt.Errorf("retention directory contains foreign files")
		}
		marker, _ = json.Marshal(retentionMarker{1, cfg.Owner})
		if err := retentionWrite(root, "owner.json", marker); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	var m retentionMarker
	if err := decodeOperationsJSON(marker, &m); err != nil || m.Version != 1 || m.Owner != cfg.Owner {
		return nil, fmt.Errorf("retention owner mismatch or corrupt marker")
	}
	if err := root.Mkdir("lease", 0700); err != nil {
		return nil, fmt.Errorf("retention writer lease: %w", err)
	}
	r := &Retention{root: root, cfg: cfg}
	if _, _, err := r.scan(); err != nil {
		root.Remove("lease")
		return nil, err
	}
	// A pending file can exist only after an interrupted, explicitly recovered
	// writer. It was never published and is not considered retained evidence.
	if err := root.Remove("pending.jsonl"); err != nil && !os.IsNotExist(err) {
		root.Remove("lease")
		return nil, err
	}
	ok = true
	return r, nil
}

// RecoverRetention releases a stale directory lease after validating ownership.
// The caller must first establish that the previous writer has stopped. This
// function does not infer process death, stop a process or remove retained data.
func RecoverRetention(path, owner string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&0077 != 0 {
		return fmt.Errorf("retention directory must be private")
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return err
	}
	defer root.Close()
	if _, err := retentionEntries(root); err != nil {
		return err
	}
	b, err := retentionMarkerBytes(root)
	if err != nil {
		return err
	}
	var m retentionMarker
	if err := decodeOperationsJSON(b, &m); err != nil || m.Version != 1 || m.Owner != owner {
		return fmt.Errorf("retention owner mismatch or corrupt marker")
	}
	if _, err := retentionEntries(root); err != nil {
		return err
	}
	return root.Remove("lease")
}

func retentionEntries(root *os.Root) ([]os.DirEntry, error) {
	f, err := root.Open(".")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	entries, err := f.ReadDir(1028)
	if err != nil && err != io.EOF {
		return nil, err
	}
	if len(entries) > 1027 {
		return nil, fmt.Errorf("retention directory entry limit exceeded")
	}
	for _, e := range entries {
		if e.Type()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("retention symbolic link refused")
		}
		if e.Name() == "lease" && e.IsDir() {
			continue
		}
		if !e.Type().IsRegular() || (e.Name() != "owner.json" && e.Name() != "pending.jsonl" && !retentionName(e.Name())) {
			return nil, fmt.Errorf("retention foreign file %q", e.Name())
		}
	}
	return entries, nil
}

func retentionMarkerBytes(root *os.Root) ([]byte, error) {
	f, err := root.Open("owner.json")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, 1025))
	if len(b) > 1024 {
		return nil, fmt.Errorf("retention marker exceeds byte limit")
	}
	return b, err
}
func retentionName(s string) bool {
	if len(s) != 26 || !strings.HasPrefix(s, "stream-") || !strings.HasSuffix(s, ".jsonl") {
		return false
	}
	for _, c := range s[7:20] {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

type retainedFile struct {
	name string
	size int64
}

func (r *Retention) scan() ([]retainedFile, int64, error) {
	entries, err := retentionEntries(r.root)
	if err != nil {
		return nil, 0, err
	}
	var out []retainedFile
	var total int64
	for _, e := range entries {
		if !retentionName(e.Name()) {
			continue
		}
		f, err := r.root.Open(e.Name())
		if err != nil {
			return nil, 0, err
		}
		info, err := f.Stat()
		if err == nil && info.Size() > r.cfg.MaxStreamBytes {
			err = fmt.Errorf("retained stream exceeds configured limit")
		}
		if err == nil && (len(out) >= r.cfg.MaxFiles || info.Size() > r.cfg.MaxBytes-total) {
			err = fmt.Errorf("retained state exceeds configured budget")
		}
		if err == nil {
			_, _, _, err = ReadStreamStrict(f, r.cfg.MaxStreamBytes)
		}
		f.Close()
		if err != nil {
			return nil, 0, fmt.Errorf("retained stream %s: %w", e.Name(), err)
		}
		total += info.Size()
		out = append(out, retainedFile{e.Name(), info.Size()})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	if len(out) > r.cfg.MaxFiles || total > r.cfg.MaxBytes {
		return nil, 0, fmt.Errorf("retained state exceeds configured budget")
	}
	return out, total, nil
}
func retentionWrite(root *os.Root, name string, b []byte) error {
	f, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	n, err := f.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		root.Remove(name)
		return err
	}
	return closeErr
}

// Store validates and writes a bounded complete stream, removes the oldest
// owned streams to make room, then publishes it. A pending write needs at most
// MaxStreamBytes extra disk headroom. A failed publication can follow pruning;
// callers must retain the input until Store succeeds. IDs are increasing
// thirteen-digit integers; replaying an existing or older ID is refused.
func (r *Retention) Store(id uint64, input io.Reader) (err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	defer func() {
		if err != nil {
			r.storeFailures++
			r.lastError = "store_failed"
		} else {
			r.stores++
			r.lastError = ""
		}
	}()
	if r.root == nil {
		return fmt.Errorf("retention is closed")
	}
	if id > 9999999999999 || input == nil {
		return fmt.Errorf("invalid retained stream")
	}
	b, err := io.ReadAll(io.LimitReader(input, r.cfg.MaxStreamBytes+1))
	if err != nil {
		return err
	}
	if int64(len(b)) > r.cfg.MaxStreamBytes {
		return fmt.Errorf("retained stream exceeds byte limit")
	}
	if _, _, _, err := ReadStreamStrict(bytes.NewReader(b), r.cfg.MaxStreamBytes); err != nil {
		return err
	}
	files, total, err := r.scan()
	if err != nil {
		return err
	}
	name := fmt.Sprintf("stream-%013d.jsonl", id)
	if len(files) > 0 && name <= files[len(files)-1].name {
		return fmt.Errorf("retained stream identifier is not increasing")
	}
	if err := retentionWrite(r.root, "pending.jsonl", b); err != nil {
		return err
	}
	total += int64(len(b))
	for len(files)+1 > r.cfg.MaxFiles || total > r.cfg.MaxBytes {
		if err := r.root.Remove(files[0].name); err != nil {
			r.root.Remove("pending.jsonl")
			return fmt.Errorf("retention prune: %w", err)
		}
		r.prunedFiles++
		r.prunedBytes += uint64(files[0].size)
		total -= files[0].size
		files = files[1:]
	}
	if err := r.root.Rename("pending.jsonl", name); err != nil {
		r.root.Remove("pending.jsonl")
		return err
	}
	return nil
}

// Close releases this writer's lease. It does not delete streams or the owner marker.
func (r *Retention) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.root == nil {
		return nil
	}
	err := r.root.Remove("lease")
	closeErr := r.root.Close()
	r.root = nil
	if err != nil {
		r.lastError = "close_failed"
		return err
	}
	if closeErr != nil {
		r.lastError = "close_failed"
	}
	return closeErr
}

// RetentionStatus describes the managed streams and observations since open.
// Limits exclude the marker, lease, and a pending file; Store may require an
// additional MaxStreamBytes of disk space. AtCapacity indicates pruning pressure,
// not disk exhaustion. Readiness is a bounded validation, not a write probe.
// Error fields contain fixed categories, never paths or stream contents.
type RetentionStatus struct {
	CountsAvailable    bool          `json:"counts_available"`
	Files              int           `json:"files"`
	Bytes              int64         `json:"bytes"`
	MaxFiles           int           `json:"max_files"`
	MaxBytes           int64         `json:"max_bytes"`
	MaxStreamBytes     int64         `json:"max_stream_bytes"`
	Closed             bool          `json:"closed"`
	Ready              bool          `json:"ready"`
	AtCapacity         bool          `json:"at_capacity"`
	StateError         string        `json:"state_error,omitempty"`
	LastOperationError string        `json:"last_operation_error,omitempty"`
	Stores             uint64        `json:"stores"`
	StoreFailures      uint64        `json:"store_failures"`
	PrunedFiles        uint64        `json:"pruned_files"`
	PrunedBytes        uint64        `json:"pruned_bytes"`
	QueryDuration      time.Duration `json:"query_duration_nanos"`
}

// Status validates the bounded managed state under the writer lock. Corruption
// returns Ready=false and StateError=state_unavailable, with zero counts whose
// availability must not be inferred. A closed or zero Retention reports closed.
// It neither prunes nor probes writes and makes no free-disk-space claim.
func (r *Retention) Status() RetentionStatus {
	r.mu.Lock()
	defer r.mu.Unlock()
	start := time.Now()
	s := RetentionStatus{MaxFiles: r.cfg.MaxFiles, MaxBytes: r.cfg.MaxBytes, MaxStreamBytes: r.cfg.MaxStreamBytes, Closed: r.root == nil, Stores: r.stores, StoreFailures: r.storeFailures, PrunedFiles: r.prunedFiles, PrunedBytes: r.prunedBytes, LastOperationError: r.lastError}
	if s.Closed {
		return s
	}
	files, total, err := r.scan()
	s.QueryDuration = time.Since(start)
	if err != nil {
		s.StateError = "state_unavailable"
		return s
	}
	s.Ready = true
	s.CountsAvailable = true
	s.Files = len(files)
	s.Bytes = total
	s.AtCapacity = s.Files >= s.MaxFiles || s.Bytes >= s.MaxBytes
	return s
}
