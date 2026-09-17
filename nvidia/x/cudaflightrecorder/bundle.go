package cudaflightrecorder

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
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
	// ClockUncertaintySource is measured, configured, or unavailable.
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

// WriteBundle writes b to w as a standard TAR archive.
func WriteBundle(w io.Writer, b *Bundle) error {
	if b == nil {
		return errors.New("cudaflightrecorder: nil bundle")
	}

	tw := tar.NewWriter(w)
	defer tw.Close()

	// 1. Serialize manifest.json
	manifestBytes, err := json.MarshalIndent(b.Manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("cudaflightrecorder: marshal manifest: %w", err)
	}

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

	// 2. Write all attached files sorted deterministically by name
	names := make([]string, 0, len(b.Files))
	for name := range b.Files {
		if name != "manifest.json" {
			names = append(names, name)
		}
	}
	sort.Strings(names)

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

// ReadBundle reads a diagnostic TAR bundle from r.
func ReadBundle(r io.Reader) (*Bundle, error) {
	tr := tar.NewReader(r)

	bundle := &Bundle{
		Files: make(map[string][]byte),
	}
	var foundManifest bool

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("cudaflightrecorder: read tar entry: %w", err)
		}

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

// SnapshotDiskManager manages disk retention budgets for saved flight recorder snapshots.
type SnapshotDiskManager struct {
	dir      string
	maxBytes int64
	mu       sync.Mutex
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

	dataLen := int64(len(data))
	if dataLen > m.maxBytes {
		return "", fmt.Errorf("%w: snapshot size %d exceeds total disk budget %d", ErrDiskBudgetExceeded, dataLen, m.maxBytes)
	}

	entries, err := os.ReadDir(m.dir)
	if err != nil {
		return "", fmt.Errorf("cudaflightrecorder: read snapshot dir: %w", err)
	}

	var files []diskFile
	var currentTotal int64
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		files = append(files, diskFile{
			path:    filepath.Join(m.dir, entry.Name()),
			size:    info.Size(),
			modTime: info.ModTime(),
		})
		currentTotal += info.Size()
	}

	// Sort files by modTime ascending (oldest first)
	sort.Slice(files, func(i, j int) bool {
		return files[i].modTime.Before(files[j].modTime)
	})

	// Prune oldest files until currentTotal + dataLen <= maxBytes
	for _, f := range files {
		if currentTotal+dataLen <= m.maxBytes {
			break
		}
		if err := os.Remove(f.path); err == nil {
			currentTotal -= f.size
		}
	}

	if currentTotal+dataLen > m.maxBytes {
		return "", fmt.Errorf("%w: remaining usage %d plus snapshot %d exceeds budget %d", ErrDiskBudgetExceeded, currentTotal, dataLen, m.maxBytes)
	}

	destPath := filepath.Join(m.dir, filename)
	if err := os.WriteFile(destPath, data, 0644); err != nil {
		return "", fmt.Errorf("cudaflightrecorder: write snapshot file: %w", err)
	}

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
