// Package nativereceipt records file identities observed in a native process's
// Linux mappings. It does not prove in-memory code integrity or transient loads.
package nativereceipt

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	maxMaps       = 1 << 20
	maxStages     = 256
	maxRawMaps    = 16 << 20
	maxFiles      = 128
	maxFileBytes  = 128 << 20
	maxTotalBytes = 512 << 20
)

type Process struct {
	PID              int    `json:"pid"`
	Parent           int    `json:"parent"`
	Group            int    `json:"group"`
	Start            uint64 `json:"start"`
	Executable       string `json:"executable"`
	ExecutableSHA256 string `json:"executable_sha256"`
}
type Mapping struct {
	Begin, End, Offset uint64
	Permissions        string
	Major, Minor       uint32
	Inode              uint64
	Path               string
}
type File struct {
	Path   string `json:"path"`
	Major  uint32 `json:"major"`
	Minor  uint32 `json:"minor"`
	Inode  uint64 `json:"inode"`
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}
type Stage struct {
	Name           string    `json:"name"`
	SourceSHA256   string    `json:"source_sha256,omitempty"`
	ArtifactSHA256 string    `json:"artifact_sha256,omitempty"`
	RawMaps        string    `json:"raw_maps"`
	Mappings       []Mapping `json:"mappings"`
	CaptureNanos   int64     `json:"capture_nanos"`
}
type Receipt struct {
	Process          Process  `json:"process"`
	Stages           []Stage  `json:"stages"`
	Files            []File   `json:"files"`
	DriverObserved   bool     `json:"driver_observed"`
	NVRTCObserved    bool     `json:"nvrtc_observed"`
	BuiltinsObserved bool     `json:"builtins_observed"`
	Unobserved       []string `json:"unobserved,omitempty"`
	HashNanos        int64    `json:"hash_nanos"`
	Limit            string   `json:"limit"`
}
type body struct {
	file     *os.File
	before   os.FileInfo
	identity File
}

// Session retains descriptors until Close. New and Capture observe this process,
// never a controller's child. Hashing is deferred to Write and timed separately.
type Session struct {
	mu       sync.Mutex
	receipt  Receipt
	bodies   []*body
	rawBytes int
	bytes    int64
	closed   bool
	written  bool
	failed   error
}

func validDigest(s string) bool {
	if s == "" {
		return true
	}
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil && s == strings.ToLower(s)
}

// New starts a receipt for the current process. Other platforms return an error.
func New() (*Session, error) { return newSession() }

// Capture retains one bounded maps snapshot and its selected file descriptors.
// Empty digests are permitted before compilation; populated digests identify
// source and actual compiled artifact bytes at the consumer call site.
func (s *Session) Capture(name, source, artifact string) error {
	if s == nil {
		return errors.New("native receipt: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.written {
		return errors.New("native receipt: session is finished")
	}
	if s.failed != nil {
		return s.failed
	}
	if name == "" || len(name) > 128 || !validDigest(source) || !validDigest(artifact) || len(s.receipt.Stages) >= maxStages {
		return errors.New("native receipt: invalid or excessive stage")
	}
	start := time.Now()
	stage, err := capture(s, name, source, artifact)
	if err != nil {
		s.failed = err
		return err
	}
	stage.CaptureNanos = time.Since(start).Nanoseconds()
	s.receipt.Stages = append(s.receipt.Stages, stage)
	return nil
}

// Write hashes retained descriptors and writes one exclusive receipt. Driver
// and NVRTC must actually have been observed; absent builtins remain unobserved.
// Hash time is reported; it must not be silently removed from process timing.
func (s *Session) Write(path string) error {
	return s.write(path, true)
}

// WriteDriver writes a precompiled-image receipt requiring an observed driver
// and executable. NVRTC and builtins are recorded only when actually observed.
// Their absence is explicit; this method does not load either library.
func (s *Session) WriteDriver(path string) error {
	return s.write(path, false)
}

func (s *Session) write(path string, requireNVRTC bool) error {
	if s == nil {
		return errors.New("native receipt: nil session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.written {
		return errors.New("native receipt: session is finished")
	}
	if s.failed != nil {
		return s.failed
	}
	if !filepath.IsAbs(path) {
		return errors.New("native receipt: output must be absolute")
	}
	if !s.receipt.DriverObserved || requireNVRTC && !s.receipt.NVRTCObserved {
		return errors.New("native receipt: required driver or NVRTC mapping absent")
	}
	start := time.Now()
	files := make([]File, 0, len(s.bodies))
	for _, b := range s.bodies {
		info, err := b.file.Stat()
		if err != nil {
			return err
		}
		if !sameBody(b.before, info) {
			return errors.New("native receipt: mapped body changed before hashing")
		}
		if _, err = b.file.Seek(0, 0); err != nil {
			return err
		}
		h := sha256.New()
		n, err := io.Copy(h, io.LimitReader(b.file, maxFileBytes+1))
		if err != nil {
			return err
		}
		after, err := b.file.Stat()
		if err != nil {
			return err
		}
		if n != b.before.Size() || n > maxFileBytes || !sameBody(info, after) {
			return errors.New("native receipt: mapped body changed during hashing")
		}
		f := b.identity
		f.SHA256 = hex.EncodeToString(h.Sum(nil))
		files = append(files, f)
		if f.Path == s.receipt.Process.Executable {
			s.receipt.Process.ExecutableSHA256 = f.SHA256
		}
	}
	s.receipt.HashNanos = time.Since(start).Nanoseconds()
	s.receipt.Files = files
	if s.receipt.Process.ExecutableSHA256 == "" {
		return errors.New("native receipt: executable mapping absent")
	}
	if !s.receipt.BuiltinsObserved {
		s.receipt.Unobserved = []string{"NVRTC builtins (possibly transient; not inferred)"}
	}
	if !s.receipt.NVRTCObserved {
		s.receipt.Unobserved = append(s.receipt.Unobserved, "NVRTC (not observed; no compiler mapping inferred)")
	}
	s.receipt.Limit = "mapped descriptor file identity; no in-memory integrity or exhaustive transient-load proof"
	if err := validateProofRequired(s.receipt, requireNVRTC); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s.receipt, "", "  ")
	if err != nil {
		return err
	}
	if len(data) > 32<<20 {
		return errors.New("native receipt: encoded receipt too large")
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	n, writeErr := f.Write(append(data, '\n'))
	closeErr := f.Close()
	if writeErr == nil && n != len(data)+1 {
		writeErr = io.ErrShortWrite
	}
	if err = errors.Join(writeErr, closeErr); err != nil {
		return fmt.Errorf("native receipt: write: %w", err)
	}
	s.written = true
	return nil
}
func sameBody(a, b os.FileInfo) bool {
	return a.Mode().IsRegular() && b.Mode().IsRegular() && os.SameFile(a, b) && a.Size() == b.Size() && a.ModTime() == b.ModTime() && metadataEqual(a, b)
}

// Close releases every retained descriptor, including after capture failure.
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	var err error
	for _, b := range s.bodies {
		err = errors.Join(err, b.file.Close())
	}
	s.bodies = nil
	return err
}

// Read decodes one bounded exclusive producer receipt. It does not infer child
// identity; the consumer must Check it against the process it actually started.
func Read(path string) (Receipt, error) {
	var r Receipt
	if !filepath.IsAbs(path) {
		return r, errors.New("native receipt: input must be absolute")
	}
	f, err := openReceipt(path)
	if err != nil {
		return r, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return r, err
	}
	if !info.Mode().IsRegular() || info.Size() > 32<<20 {
		return r, errors.New("native receipt: invalid receipt body")
	}
	data, err := io.ReadAll(io.LimitReader(f, (32<<20)+1))
	if err != nil {
		return r, err
	}
	if len(data) > 32<<20 {
		return r, errors.New("native receipt: receipt too large")
	}
	after, err := f.Stat()
	if err != nil {
		return r, err
	}
	if int64(len(data)) != info.Size() || !sameBody(info, after) {
		return r, errors.New("native receipt: receipt changed during read")
	}
	if err := uniqueJSON(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return r, err
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err = d.Decode(&r); err != nil {
		return r, err
	}
	if err = d.Decode(new(any)); err != io.EOF {
		return r, errors.New("native receipt: trailing receipt data")
	}
	return r, nil
}

// uniqueJSON rejects duplicate object keys before ordinary typed decoding.
func uniqueJSON(d *json.Decoder, depth int) error {
	if depth > 32 {
		return errors.New("native receipt: excessive JSON depth")
	}
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	if delim != '{' && delim != '[' {
		return errors.New("native receipt: invalid JSON delimiter")
	}
	seen := make(map[string]bool)
	for d.More() {
		if delim == '{' {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return errors.New("native receipt: duplicate JSON key")
			}
			seen[name] = true
		}
		if err := uniqueJSON(d, depth+1); err != nil {
			return err
		}
	}
	_, err = d.Token()
	return err
}

// Check binds a child receipt to its admitted executable/start/group identity,
// submitted source and actual compilation digest. Empty digests are refused.
func (r Receipt) Check(expected Process, source, artifact string) error {
	return r.check(expected, source, artifact, true)
}

// CheckDriver checks a precompiled-image receipt against the actual producer
// and submitted source/image bytes. Driver and executable mapping proofs are
// required; optional NVRTC and builtin claims must agree with all descriptors.
func (r Receipt) CheckDriver(expected Process, source, artifact string) error {
	return r.check(expected, source, artifact, false)
}

func (r Receipt) check(expected Process, source, artifact string, requireNVRTC bool) error {
	p := r.Process
	if p.PID != expected.PID || p.Parent != expected.Parent || p.Group != expected.Group || p.Start != expected.Start || p.Executable != expected.Executable || p.ExecutableSHA256 != expected.ExecutableSHA256 || p.PID <= 1 || p.Start == 0 || !validDigest(p.ExecutableSHA256) || p.ExecutableSHA256 == "" {
		return errors.New("native receipt: producer process identity disagrees")
	}
	if source == "" || artifact == "" || !validDigest(source) || !validDigest(artifact) {
		return errors.New("native receipt: compilation identity required")
	}
	if !r.DriverObserved || requireNVRTC && !r.NVRTCObserved || len(r.Stages) < 1 || len(r.Stages) > maxStages || len(r.Files) > maxFiles {
		return errors.New("native receipt: required mapping proof absent")
	}
	found := false
	for _, s := range r.Stages {
		if s.SourceSHA256 == source && s.ArtifactSHA256 == artifact && s.RawMaps != "" && len(s.Mappings) > 0 {
			found = true
		}
	}
	if !found {
		return errors.New("native receipt: submitted compilation not observed")
	}
	for _, f := range r.Files {
		if f.SHA256 == "" || !validDigest(f.SHA256) || f.Bytes < 1 || f.Bytes > maxFileBytes {
			return errors.New("native receipt: mapped body hash absent")
		}
	}
	return validateProofRequired(r, requireNVRTC)
}
