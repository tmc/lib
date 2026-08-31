package cudatrace

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/trace"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// SpanRecord represents a buffered trace slice for flight recording.
type SpanRecord struct {
	TrackID  uint64
	StartNs  uint64
	EndNs    uint64
	Name     string
	Category string
}

// Tracer manages trace capture, timeline tracks, and post-mortem flight recording.
type Tracer struct {
	mu           sync.Mutex
	pw           *perfetto.Writer
	streamTracks map[uintptr]uint64
	nextTrackID  uint64
	ringBuffer   []SpanRecord
	ringHead     int
	ringCap      int
}

// New creates a Tracer writing into w.
func New(w io.Writer) *Tracer {
	return &Tracer{
		pw:           perfetto.NewWriter(w),
		streamTracks: make(map[uintptr]uint64),
		nextTrackID:  1000,
		ringCap:      4096,
		ringBuffer:   make([]SpanRecord, 0, 4096),
	}
}

// RegisterStream creates a dedicated timeline track for a CUDA stream.
func (t *Tracer) RegisterStream(s *cuda.Stream, name string) (uint64, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	var streamKey uintptr
	if s != nil {
		streamKey = uintptr(s.Raw())
	}

	if id, exists := t.streamTracks[streamKey]; exists {
		return id, nil
	}

	t.nextTrackID++
	trackID := t.nextTrackID
	t.streamTracks[streamKey] = trackID

	if err := t.pw.WriteTrackDescriptor(trackID, name, 0); err != nil {
		return 0, fmt.Errorf("cudatrace: write track descriptor: %w", err)
	}
	return trackID, nil
}

// Span executes fn while recording a correlated trace slice on stream's track and Go runtime/trace.
func (t *Tracer) Span(s *cuda.Stream, name string, category string, fn func()) error {
	var streamKey uintptr
	if s != nil {
		streamKey = uintptr(s.Raw())
	}

	t.mu.Lock()
	trackID, exists := t.streamTracks[streamKey]
	if !exists {
		t.nextTrackID++
		trackID = t.nextTrackID
		t.streamTracks[streamKey] = trackID
		_ = t.pw.WriteTrackDescriptor(trackID, fmt.Sprintf("Stream %d", streamKey), 0)
	}
	t.mu.Unlock()

	// Go runtime region
	region := trace.StartRegion(nil, name)
	defer region.End()

	startNs := uint64(time.Now().UnixNano())
	if err := t.pw.WriteSliceBegin(trackID, startNs, name, category); err != nil {
		return err
	}

	fn()

	endNs := uint64(time.Now().UnixNano())
	if err := t.pw.WriteSliceEnd(trackID, endNs); err != nil {
		return err
	}

	t.mu.Lock()
	record := SpanRecord{
		TrackID:  trackID,
		StartNs:  startNs,
		EndNs:    endNs,
		Name:     name,
		Category: category,
	}
	if len(t.ringBuffer) < t.ringCap {
		t.ringBuffer = append(t.ringBuffer, record)
	} else {
		t.ringBuffer[t.ringHead] = record
		t.ringHead = (t.ringHead + 1) % t.ringCap
	}
	t.mu.Unlock()

	return nil
}

// Instant writes an instantaneous marker event on the stream's timeline.
func (t *Tracer) Instant(s *cuda.Stream, name string, category string) error {
	var streamKey uintptr
	if s != nil {
		streamKey = uintptr(s.Raw())
	}

	t.mu.Lock()
	trackID, exists := t.streamTracks[streamKey]
	if !exists {
		t.nextTrackID++
		trackID = t.nextTrackID
		t.streamTracks[streamKey] = trackID
		_ = t.pw.WriteTrackDescriptor(trackID, fmt.Sprintf("Stream %d", streamKey), 0)
	}
	t.mu.Unlock()

	nowNs := uint64(time.Now().UnixNano())
	return t.pw.WriteInstant(trackID, nowNs, name, category)
}

// DumpPostMortem immediately serializes the in-memory flight recording ring buffer to dest.
// This executes safely without invoking any CUDA driver API calls even if CUcontext is invalidated.
func (t *Tracer) DumpPostMortem(dest io.Writer) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	pw := perfetto.NewWriter(dest)

	// Write tracks
	for streamKey, trackID := range t.streamTracks {
		_ = pw.WriteTrackDescriptor(trackID, fmt.Sprintf("Stream %d (Post-Mortem)", streamKey), 0)
	}

	// Dump all buffered slices
	for _, rec := range t.ringBuffer {
		_ = pw.WriteSliceBegin(rec.TrackID, rec.StartNs, rec.Name, rec.Category)
		_ = pw.WriteSliceEnd(rec.TrackID, rec.EndNs)
	}

	return nil
}

// DumpToCrashFile writes post-mortem trace records to ~/tmp/cuda-crash-<timestamp>.pftrace.
func (t *Tracer) DumpToCrashFile() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, "tmp")
	_ = os.MkdirAll(dir, 0755)

	path := filepath.Join(dir, fmt.Sprintf("cuda-crash-%d.pftrace", time.Now().UnixNano()))
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	if err := t.DumpPostMortem(f); err != nil {
		return "", err
	}
	return path, nil
}
