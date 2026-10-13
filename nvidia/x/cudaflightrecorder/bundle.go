package cudaflightrecorder

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Manifest records metadata and time-window spans for a flight recorder snapshot.
type Manifest struct {
	Timestamp         time.Time `json:"timestamp"`
	Reason            string    `json:"reason"`
	WindowRequestedNs int64     `json:"window_requested_ns"`
	GoSpanNs          int64     `json:"go_span_ns"`
	GPUSpanNs         int64     `json:"gpu_span_ns"`
	OverlapSpanNs     int64     `json:"overlap_span_ns"`
	GPURecords        int       `json:"gpu_records"`
	GPUDropped        int64     `json:"gpu_dropped"`
	// GPUOutsideWindow counts retained records excluded by the snapshot cutoff.
	GPUOutsideWindow int   `json:"gpu_outside_window"`
	GoTraceBytes     int64 `json:"go_trace_bytes"`
	// ClockUncertaintyNs is the alignment uncertainty in nanoseconds; -1 means unavailable.
	ClockUncertaintyNs int64 `json:"clock_uncertainty_ns"`
	// ClockUncertaintySource says where ClockUncertaintyNs came from.
	// "estimated" follows a successful Recorder.CalibrateClock: half the
	// host-clock bracket measured around a CUPTI clock read, plus drift
	// since the read at an assumed 11 ppm. It is not a qualified bound on
	// the physical alignment error; the true drift is not measured.
	// "configured" is Config.ClockUncertaintyNs. "unavailable" means neither,
	// and "host_only" marks a fault snapshot; both report -1.
	ClockUncertaintySource string            `json:"clock_uncertainty_source"`
	OverlapReliable        bool              `json:"overlap_reliable"`
	ProviderStatus         map[string]string `json:"provider_status,omitempty"`
	LossReasons            []string          `json:"loss_reasons,omitempty"`
}

// WindowRequested reports the requested recording duration as time.Duration.
func (m *Manifest) WindowRequested() time.Duration {
	return time.Duration(m.WindowRequestedNs)
}

// GoSpan reports the observed Go execution trace duration.
func (m *Manifest) GoSpan() time.Duration {
	return time.Duration(m.GoSpanNs)
}

// GPUSpan reports the observed GPU observation duration.
func (m *Manifest) GPUSpan() time.Duration {
	return time.Duration(m.GPUSpanNs)
}

// OverlapSpan reports the overlapping time duration where both Go and GPU records are present.
func (m *Manifest) OverlapSpan() time.Duration {
	return time.Duration(m.OverlapSpanNs)
}

// Bundle represents a multi-modal flight recorder archive containing the diagnostic manifest
// and raw binary artifacts (such as gpu.pprof, go.trace, kernels.ptx, kernels.sass).
type Bundle struct {
	Manifest Manifest
	Files    map[string][]byte
}

// NewBundle creates an empty diagnostic bundle with the given manifest.
func NewBundle(m Manifest) *Bundle {
	return &Bundle{
		Manifest: m,
		Files:    make(map[string][]byte),
	}
}

// WriteBundle writes b to w as a standard TAR archive. The same payload, entry
// count and name limits as ReadBundle are checked before writing any bytes.
func WriteBundle(w io.Writer, b *Bundle) (err error) {
	if b == nil || w == nil {
		return errors.New("cudaflightrecorder: nil bundle")
	}

	// 1. Serialize manifest.json
	manifestBytes, err := json.MarshalIndent(b.Manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("cudaflightrecorder: marshal manifest: %w", err)
	}

	if len(b.Files) >= maxBundleFiles || int64(len(manifestBytes)) > maxBundleBytes {
		return ErrSnapshotBudgetExceeded
	}
	total := int64(len(manifestBytes))
	names := make([]string, 0, len(b.Files))
	for name, data := range b.Files {
		if !validBundleName(name) || name == "manifest.json" {
			return errors.New("cudaflightrecorder: invalid or reserved bundle name")
		}
		if int64(len(data)) > maxBundleBytes-total {
			return ErrSnapshotBudgetExceeded
		}
		total += int64(len(data))
		names = append(names, name)
	}
	sort.Strings(names)
	tw := tar.NewWriter(w)
	defer func() { err = errors.Join(err, tw.Close()) }()

	hdr := &tar.Header{
		Name:    "manifest.json",
		Mode:    0644,
		Size:    int64(len(manifestBytes)),
		ModTime: b.Manifest.Timestamp,
	}
	if hdr.ModTime.IsZero() {
		hdr.ModTime = time.Now()
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return fmt.Errorf("cudaflightrecorder: write manifest header: %w", err)
	}
	if _, err := tw.Write(manifestBytes); err != nil {
		return fmt.Errorf("cudaflightrecorder: write manifest content: %w", err)
	}

	// Write attached files sorted deterministically by name.
	for _, name := range names {
		data := b.Files[name]
		fileHdr := &tar.Header{
			Name:    name,
			Mode:    0644,
			Size:    int64(len(data)),
			ModTime: hdr.ModTime,
		}
		if err := tw.WriteHeader(fileHdr); err != nil {
			return fmt.Errorf("cudaflightrecorder: write file header %q: %w", name, err)
		}
		if _, err := tw.Write(data); err != nil {
			return fmt.Errorf("cudaflightrecorder: write file content %q: %w", name, err)
		}
	}

	return nil
}

const maxBundleBytes int64 = 256 << 20
const maxBundleFiles = 1024

func validBundleName(name string) bool {
	return name != "" && len(name) <= 4096 && name != "." && !strings.ContainsAny(name, "\\\x00") && !path.IsAbs(name) && path.Clean(name) == name && name != ".." && !strings.HasPrefix(name, "../")
}

// ReadBundle reads a diagnostic TAR bundle from r. It accepts at most 1024
// regular entries and 256 MiB of payload, matching the default snapshot budget.
// Duplicate entries, links, and unsafe names are refused; no files are extracted.
func ReadBundle(r io.Reader) (*Bundle, error) {
	if r == nil {
		return nil, errors.New("cudaflightrecorder: nil bundle reader")
	}
	input := &io.LimitedReader{R: r, N: maxBundleBytes + (8 << 20)}
	tr := tar.NewReader(input)

	bundle := &Bundle{
		Files: make(map[string][]byte),
	}
	var foundManifest bool
	seen := make(map[string]bool)
	var total int64

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			if input.N == 0 {
				return nil, ErrSnapshotBudgetExceeded
			}
			break
		}
		if err != nil {
			return nil, fmt.Errorf("cudaflightrecorder: read tar entry: %w", err)
		}

		if !validBundleName(hdr.Name) || (hdr.Typeflag != tar.TypeReg && hdr.Typeflag != tar.TypeRegA) {
			return nil, errors.New("cudaflightrecorder: unsafe bundle entry")
		}
		if seen[hdr.Name] || len(seen) >= maxBundleFiles {
			return nil, errors.New("cudaflightrecorder: duplicate or excessive bundle entries")
		}
		seen[hdr.Name] = true
		if hdr.Size < 0 || hdr.Size > maxBundleBytes-total {
			return nil, ErrSnapshotBudgetExceeded
		}
		total += hdr.Size
		var buf bytes.Buffer
		if _, err := io.Copy(&buf, tr); err != nil {
			return nil, fmt.Errorf("cudaflightrecorder: read tar content for %q: %w", hdr.Name, err)
		}

		data := buf.Bytes()
		if hdr.Name == "manifest.json" {
			if err := json.Unmarshal(data, &bundle.Manifest); err != nil {
				return nil, fmt.Errorf("cudaflightrecorder: unmarshal manifest: %w", err)
			}
			foundManifest = true
		} else {
			bundle.Files[hdr.Name] = data
		}
	}

	if !foundManifest {
		return nil, errors.New("cudaflightrecorder: bundle missing manifest.json")
	}

	return bundle, nil
}

// Budget and validation errors.
var (
	ErrDiskBudgetExceeded     = errors.New("cudaflightrecorder: disk retention budget exhausted")
	ErrProviderBudgetExceeded = errors.New("cudaflightrecorder: provider output budget exceeded")
	ErrSnapshotBudgetExceeded = errors.New("cudaflightrecorder: snapshot size budget exceeded")
	ErrPlaceholderRejected    = errors.New("cudaflightrecorder: placeholder payload rejected")
	ErrArtifactInvalid        = errors.New("cudaflightrecorder: invalid artifact format")
)

// FaultInfo describes a GPU fault or abort event for zero-device crash dump generation.
type FaultInfo struct {
	TimestampNs      int64             `json:"timestamp_ns"`
	ErrorCode        int               `json:"error_code"`
	ErrorName        string            `json:"error_name"`
	ErrorMessage     string            `json:"error_message"`
	OffendingKernel  string            `json:"offending_kernel,omitempty"`
	CorrelationID    uint64            `json:"correlation_id,omitempty"`
	ModuleID         string            `json:"module_id,omitempty"`
	ModuleDigest     string            `json:"module_digest,omitempty"`
	ModuleGeneration uint64            `json:"module_generation,omitempty"`
	FailedRuleNames  []string          `json:"failed_rule_names,omitempty"`
	Labels           map[string]string `json:"labels,omitempty"`
}

// NegativeRewriteConstraint represents a negative rewrite rule exported in a crash dump
// to feed negative constraints to egraph search.
type NegativeRewriteConstraint struct {
	CandidateDigest  string    `json:"candidate_digest"`
	OffendingKernel  string    `json:"offending_kernel"`
	QuarantinedRules []string  `json:"quarantined_rules"`
	Reason           string    `json:"reason"`
	PruneSubtree     bool      `json:"prune_subtree"`
	Timestamp        time.Time `json:"timestamp"`
}

// SnapshotDiskManager manages regular files in one snapshot directory. Use a
// single manager and no external writer for that directory. It may prune only
// snapshots created by this instance; existing and replaced files remain
// unowned but count toward the byte budget. Ownership does not survive restart.
// Subdirectories are outside the budget and are never removed.
type SnapshotDiskManager struct {
	dir      string
	maxBytes int64
	mu       sync.Mutex
	owned    map[string]os.FileInfo
}

// NewSnapshotDiskManager creates a new disk retention manager for the specified directory.
func NewSnapshotDiskManager(dir string, maxBytes int64) (*SnapshotDiskManager, error) {
	if dir == "" {
		return nil, errors.New("cudaflightrecorder: empty snapshot directory")
	}
	if maxBytes <= 0 {
		return nil, errors.New("cudaflightrecorder: invalid non-positive disk budget")
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("cudaflightrecorder: create snapshot dir: %w", err)
	}
	return &SnapshotDiskManager{
		dir:      dir,
		maxBytes: maxBytes,
		owned:    make(map[string]os.FileInfo),
	}, nil
}

type diskFile struct {
	path    string
	size    int64
	modTime time.Time
}

// SaveSnapshot writes a snapshot archive to disk while enforcing the retention budget.
// If the snapshot exceeds maxBytes by itself, or if existing files cannot be pruned
// to accommodate it, ErrDiskBudgetExceeded is returned.
func (m *SnapshotDiskManager) SaveSnapshot(filename string, data []byte) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if filename == "" || filename == "." || filename == ".." || filepath.IsAbs(filename) || filepath.Base(filename) != filename || strings.ContainsAny(filename, "/\\") {
		return "", errors.New("cudaflightrecorder: snapshot name must be a single filename")
	}
	dataLen := int64(len(data))
	if dataLen == 0 || dataLen > m.maxBytes {
		return "", fmt.Errorf("%w: invalid snapshot size %d for budget %d", ErrDiskBudgetExceeded, dataLen, m.maxBytes)
	}
	destPath := filepath.Join(m.dir, filename)
	if _, err := os.Lstat(destPath); err == nil {
		return "", errors.New("cudaflightrecorder: snapshot destination already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("inspect snapshot destination: %w", err)
	}
	entries, err := os.ReadDir(m.dir)
	if err != nil {
		return "", fmt.Errorf("read snapshot directory: %w", err)
	}
	var files []diskFile
	var currentTotal int64
	present := make(map[string]bool)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return "", fmt.Errorf("inspect snapshot %q: %w", entry.Name(), err)
		}
		if !info.Mode().IsRegular() {
			return "", fmt.Errorf("cudaflightrecorder: nonregular snapshot entry %q", entry.Name())
		}
		if info.Size() < 0 || info.Size() > int64(^uint64(0)>>1)-currentTotal {
			return "", fmt.Errorf("%w: directory size overflow", ErrDiskBudgetExceeded)
		}
		currentTotal += info.Size()
		if owned, ok := m.owned[entry.Name()]; ok && os.SameFile(owned, info) {
			present[entry.Name()] = true
			files = append(files, diskFile{path: filepath.Join(m.dir, entry.Name()), size: info.Size(), modTime: info.ModTime()})
		}
	}
	// Remove stale ownership, including files replaced by an external writer.
	for name := range m.owned {
		if !present[name] {
			delete(m.owned, name)
		}
	}
	sort.Slice(files, func(i, j int) bool {
		if files[i].modTime.Equal(files[j].modTime) {
			return files[i].path < files[j].path
		}
		return files[i].modTime.Before(files[j].modTime)
	})
	// Capacity is bounded independently of bytes, even for very small files.
	for _, f := range files {
		if currentTotal <= m.maxBytes-dataLen && len(m.owned) < 1024 {
			break
		}
		if err := os.Remove(f.path); err != nil {
			return "", fmt.Errorf("prune owned snapshot: %w", err)
		}
		currentTotal -= f.size
		delete(m.owned, filepath.Base(f.path))
	}
	if currentTotal > m.maxBytes-dataLen || len(m.owned) >= 1024 {
		return "", fmt.Errorf("%w: unowned files or retention count prevent admission", ErrDiskBudgetExceeded)
	}
	// Exclusive creation prevents overwriting an unowned file or following a symlink.
	file, err := os.OpenFile(destPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return "", fmt.Errorf("create snapshot file: %w", err)
	}
	info, statErr := file.Stat()
	n, writeErr := file.Write(data)
	if n != len(data) && writeErr == nil {
		writeErr = io.ErrShortWrite
	}
	closeErr := file.Close()
	if err := errors.Join(statErr, writeErr, closeErr); err != nil {
		if current, inspectErr := os.Lstat(destPath); inspectErr == nil && info != nil && os.SameFile(info, current) {
			_ = os.Remove(destPath)
		}
		return "", fmt.Errorf("write snapshot file: %w", err)
	}
	m.owned[filename] = info
	return destPath, nil
}

// RetainedBytes computes the current total disk usage for snapshots in the managed directory.
func (m *SnapshotDiskManager) RetainedBytes() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	entries, err := os.ReadDir(m.dir)
	if err != nil {
		return 0
	}
	var total int64
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if info, err := entry.Info(); err == nil {
			total += info.Size()
		}
	}
	return total
}
