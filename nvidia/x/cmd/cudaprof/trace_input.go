package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// traceSession owns the bytes parsed and the single native journal decode.
// CollectorError is evidence failure, not permission to qualify partial data.
type traceSession struct {
	Path           string
	SHA256         string
	Kind           string
	Capture        *cudacapture.Capture
	Members        map[string][]byte
	Records        timeline.Records
	CollectorError error
	Data           []byte
}

type traceInputLimits struct {
	InputBytes, JournalBytes, Records, ArchiveBytes int64
	ArchiveMembers                                  int
}

func defaultTraceInputLimits() traceInputLimits {
	return traceInputLimits{512 << 20, 256 << 20, 1 << 20, 512 << 20, 4096}
}
func loadTraceSession(path, domain string) (*traceSession, error) {
	return loadTraceSessionLimits(path, domain, defaultTraceInputLimits())
}
func loadTraceSessionLimits(path, domain string, limits traceInputLimits) (*traceSession, error) {
	if limits.InputBytes <= 0 || limits.JournalBytes <= 0 || limits.Records <= 0 || limits.ArchiveBytes <= 0 || limits.ArchiveMembers <= 0 || limits.InputBytes == int64(^uint64(0)>>1) {
		return nil, errors.New("trace: resource limits must be positive and bounded")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if domain != "driver" && domain != "runtime" {
		return nil, fmt.Errorf("trace: invalid API domain %q", domain)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(traceContextReader{ctx, f}, limits.InputBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limits.InputBytes {
		return nil, fmt.Errorf("trace: input exceeds %d bytes", limits.InputBytes)
	}
	hash := sha256.Sum256(data)
	s := &traceSession{Path: path, SHA256: hex.EncodeToString(hash[:]), Data: data}
	cap, err := cudacapture.ReadLimits(bytes.NewReader(data), cudacapture.Limits{InputBytes: limits.InputBytes, FileBytes: limits.JournalBytes, TotalBytes: limits.JournalBytes})
	if err != nil {
		if len(data) >= 512 && string(data[257:262]) == "ustar" {
			members, archiveErr := inspectTraceArchiveLimits(data, limits)
			if archiveErr != nil {
				return nil, archiveErr
			}
			s.Kind = "comparison_archive"
			s.Members = members
			s.CollectorError = fmt.Errorf("comparison archive lacks hash-bound source journals")
			return s, nil
		}
		if validTracePackets(data) {
			s.Kind = "pftrace"
			s.CollectorError = fmt.Errorf("standalone Perfetto input has no native collector evidence")
			return s, nil
		}
		return nil, fmt.Errorf("trace: unsupported or malformed input: %w", err)
	}
	s.Kind = "capture"
	s.Capture = cap
	_, s.Records, s.CollectorError = decodeCompleteCapture(cap, limits.Records, limits.JournalBytes)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s, nil
}

type traceContextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r traceContextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

// validTracePackets establishes packet framing, not importer health.
func validTracePackets(data []byte) bool {
	count := 0
	for len(data) > 0 {
		tag, n := binary.Uvarint(data)
		if n <= 0 || tag != 10 {
			return false
		}
		data = data[n:]
		size, n := binary.Uvarint(data)
		if n <= 0 {
			return false
		}
		data = data[n:]
		if size == 0 || size > uint64(len(data)) {
			return false
		}
		data = data[int(size):]
		count++
		if count > 1<<20 {
			return false
		}
	}
	return count > 0
}
