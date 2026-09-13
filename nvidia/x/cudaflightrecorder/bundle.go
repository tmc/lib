package cudaflightrecorder

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"time"
)

// Manifest records metadata and time-window spans for a flight recorder snapshot.
type Manifest struct {
	Timestamp          time.Time `json:"timestamp"`
	Reason             string    `json:"reason"`
	WindowRequestedNs  int64     `json:"window_requested_ns"`
	GoSpanNs           int64     `json:"go_span_ns"`
	GPUSpanNs          int64     `json:"gpu_span_ns"`
	OverlapSpanNs      int64     `json:"overlap_span_ns"`
	GPURecords         int       `json:"gpu_records"`
	GPUDropped         int64     `json:"gpu_dropped"`
	GoTraceBytes       int64     `json:"go_trace_bytes"`
	ClockUncertaintyNs int64     `json:"clock_uncertainty_ns"`
	OverlapReliable    bool      `json:"overlap_reliable"`
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
