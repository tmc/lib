package dsv4kvstore

import (
	"bytes"
	"crypto/sha1"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// ErrNotFound reports that a store has no compatible prefix entry.
var ErrNotFound = errors.New("dsv4kvstore: entry not found")

// ErrBudgetExceeded reports that one entry cannot fit in the configured
// cache budget. A store never leaves an over-budget entry behind.
var ErrBudgetExceeded = errors.New("dsv4kvstore: entry exceeds budget")

const (
	storeSuffix       = ".dskv"
	storeMaxText      = 1 << 30
	storeMaxTrailer   = 1 << 30
	storeMaxPayload   = 1 << 34
	storeRecordPrefix = 8 // text length followed by trailer length
)

// Store is a private host-side prefix cache. The payload is opaque to this
// package: a backend owns its bytes and decides whether they can be restored.
// Store is safe for concurrent callers.
type Store struct {
	mu          sync.Mutex
	dir         string
	budgetBytes uint64
	options     Options
	hits        map[string]uint64
}

// Entry is a compatible cached prefix. Text and metadata are copied when the
// entry is returned; Payload returns a new reader over opaque backend bytes.
type Entry struct {
	Header    Header
	Text      []byte
	Trailer   []byte
	FileBytes uint64
	LastUsed  time.Time
	path      string
	payload   []byte
	store     *Store
	key       string
}

// Open creates or opens a private cache directory. A zero BudgetMB selects
// the documented 4096 MiB default. No upstream KVC payload is interpreted.
func Open(options Options) (*Store, error) {
	if options.Dir == "" {
		return nil, errors.New("dsv4kvstore: missing directory")
	}
	if options.BudgetMB <= 0 {
		options.BudgetMB = defaultBudgetMB
	}
	if options.BudgetMB > int(^uint64(0)>>20) {
		return nil, errors.New("dsv4kvstore: budget is too large")
	}
	if err := os.MkdirAll(options.Dir, 0700); err != nil {
		return nil, fmt.Errorf("create cache directory: %w", err)
	}
	return &Store{
		dir:         options.Dir,
		budgetBytes: uint64(options.BudgetMB) << 20,
		options:     options,
		hits:        make(map[string]uint64),
	}, nil
}

// Save atomically stores text, opaque backend payload, and an optional
// trailer. The filename binds rendered text, model identity, and quantization
// so incompatible states cannot overwrite one another.
func (s *Store) Save(h Header, text []byte, payload io.WriterTo, trailer []byte) error {
	if s == nil {
		return errors.New("dsv4kvstore: nil store")
	}
	if len(text) == 0 {
		return errors.New("dsv4kvstore: empty cache text")
	}
	if uint64(len(text)) > storeMaxText || uint64(len(trailer)) > storeMaxTrailer {
		return ErrTooLarge
	}
	if err := validateHeader(h); err != nil {
		return err
	}
	var opaque boundedBuffer
	opaque.limit = storeMaxPayload
	if payload != nil {
		if _, err := payload.WriteTo(&opaque); err != nil {
			return fmt.Errorf("write cache payload: %w", err)
		}
	}
	if uint64(len(opaque.buf)) > storeMaxPayload {
		return ErrTooLarge
	}
	var wrapped bytes.Buffer
	var lengths [storeRecordPrefix]byte
	binary.LittleEndian.PutUint32(lengths[0:4], uint32(len(text)))
	binary.LittleEndian.PutUint32(lengths[4:8], uint32(len(trailer)))
	wrapped.Write(lengths[:])
	wrapped.Write(text)
	wrapped.Write(trailer)
	wrapped.Write(opaque.buf)
	record := Record{Header: h, Payload: wrapped.Bytes()}
	key := textKey(text, h.ModelDigest, h.QuantBits)

	s.mu.Lock()
	defer s.mu.Unlock()
	name := filepath.Join(s.dir, key+storeSuffix)
	tmp, err := os.CreateTemp(s.dir, ".dskv-*")
	if err != nil {
		return fmt.Errorf("create cache temporary: %w", err)
	}
	tmpName := tmp.Name()
	removeTemp := true
	defer func() {
		_ = tmp.Close()
		if removeTemp {
			_ = os.Remove(tmpName)
		}
	}()
	if err := Encode(tmp, record); err != nil {
		return fmt.Errorf("encode cache entry: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("sync cache entry: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close cache entry: %w", err)
	}
	if err := os.Chmod(tmpName, 0600); err != nil {
		return fmt.Errorf("protect cache entry: %w", err)
	}
	if err := os.Rename(tmpName, name); err != nil {
		return fmt.Errorf("install cache entry: %w", err)
	}
	removeTemp = false
	delete(s.hits, key)
	if err := s.evictLocked(name); err != nil {
		return err
	}
	return nil
}

// Lookup returns the longest compatible text prefix. A zero model digest or
// quantization value is a wildcard; a nonzero context requires the saved
// context to be at least that large.
func (s *Store) Lookup(rendered []byte, context uint32, model [32]byte, quant uint8) (*Entry, error) {
	if s == nil {
		return nil, errors.New("dsv4kvstore: nil store")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	files, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, fmt.Errorf("read cache directory: %w", err)
	}
	var best *Entry
	for _, file := range files {
		if file.IsDir() || filepath.Ext(file.Name()) != storeSuffix {
			continue
		}
		path := filepath.Join(s.dir, file.Name())
		entry, err := s.readEntryLocked(path)
		if err != nil {
			continue
		}
		if !bytes.HasPrefix(rendered, entry.Text) || best != nil && len(entry.Text) <= len(best.Text) {
			continue
		}
		// A file written for a larger context may not be loaded into a
		// smaller one. A smaller saved context is compatible with a larger
		// active context, matching the upstream check.
		if context != 0 && entry.Header.Context > context {
			continue
		}
		if model != ([32]byte{}) && entry.Header.ModelDigest != ([32]byte{}) && entry.Header.ModelDigest != model {
			continue
		}
		if quant != 0 && entry.Header.QuantBits != quant {
			continue
		}
		best = entry
	}
	if best == nil {
		return nil, ErrNotFound
	}
	best.store = s
	best.key = textKey(best.Text, best.Header.ModelDigest, best.Header.QuantBits)
	s.hits[best.key]++
	return best, nil
}

// Payload returns a reader over the backend-owned bytes. The caller may close
// it; closing only releases the reader and never removes the cache entry.
func (e *Entry) Payload() (io.ReadCloser, error) {
	if e == nil {
		return nil, errors.New("dsv4kvstore: nil entry")
	}
	return io.NopCloser(bytes.NewReader(e.payload)), nil
}

// Touch updates the access time used by eviction. It rejects entries from a
// different store so callers cannot accidentally mutate an unrelated cache.
func (s *Store) Touch(e *Entry) error {
	if s == nil || e == nil || e.store != s || e.path == "" {
		return errors.New("dsv4kvstore: entry does not belong to store")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.Chtimes(e.path, time.Now(), time.Now()); err != nil {
		return fmt.Errorf("touch cache entry: %w", err)
	}
	s.hits[e.key]++
	return nil
}

func (s *Store) readEntryLocked(path string) (*Entry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	record, err := DecodeWithOptions(f, DecodeOptions{MaxTokens: 1, MaxPayload: storeMaxText + storeMaxTrailer + storeMaxPayload + storeRecordPrefix})
	if err != nil {
		return nil, err
	}
	if len(record.Payload) < storeRecordPrefix {
		return nil, ErrTruncated
	}
	textBytes := binary.LittleEndian.Uint32(record.Payload[0:4])
	trailerBytes := binary.LittleEndian.Uint32(record.Payload[4:8])
	offset := uint64(storeRecordPrefix) + uint64(textBytes) + uint64(trailerBytes)
	if uint64(len(record.Payload)) < offset || textBytes > storeMaxText || trailerBytes > storeMaxTrailer {
		return nil, ErrBadHeader
	}
	text := append([]byte(nil), record.Payload[storeRecordPrefix:storeRecordPrefix+int(textBytes)]...)
	trailerStart := storeRecordPrefix + int(textBytes)
	trailer := append([]byte(nil), record.Payload[trailerStart:trailerStart+int(trailerBytes)]...)
	payload := append([]byte(nil), record.Payload[offset:]...)
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return &Entry{Header: record.Header, Text: text, Trailer: trailer, payload: payload, path: path, FileBytes: uint64(info.Size()), LastUsed: info.ModTime()}, nil
}

func (s *Store) evictLocked(newPath string) error {
	files, err := os.ReadDir(s.dir)
	if err != nil {
		return fmt.Errorf("scan cache directory: %w", err)
	}
	type candidate struct {
		path  string
		score float64
		mtime time.Time
		bytes uint64
	}
	var entries []candidate
	var total uint64
	for _, file := range files {
		if file.IsDir() || filepath.Ext(file.Name()) != storeSuffix {
			continue
		}
		path := filepath.Join(s.dir, file.Name())
		entry, err := s.readEntryLocked(path)
		if err != nil {
			continue
		}
		total += entry.FileBytes
		entries = append(entries, candidate{path: path, bytes: entry.FileBytes, mtime: entry.LastUsed, score: EvictionScore(EvictionMetadata{Hits: s.hits[textKey(entry.Text, entry.Header.ModelDigest, entry.Header.QuantBits)], Tokens: uint64(len(entry.Text)), FileBytes: entry.FileBytes, LastUsed: entry.LastUsed})})
	}
	if total <= s.budgetBytes {
		return nil
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].score != entries[j].score {
			return entries[i].score < entries[j].score
		}
		return entries[i].mtime.Before(entries[j].mtime)
	})
	for _, entry := range entries {
		if total <= s.budgetBytes {
			break
		}
		if err := os.Remove(entry.path); err != nil {
			return fmt.Errorf("evict cache entry: %w", err)
		}
		total -= entry.bytes
		delete(s.hits, keyFromPath(entry.path))
	}
	if total > s.budgetBytes {
		_ = os.Remove(newPath)
		return ErrBudgetExceeded
	}
	return nil
}

func textKey(text []byte, model [32]byte, quant uint8) string {
	h := sha1.New()
	_, _ = h.Write(text)
	_, _ = h.Write(model[:])
	_, _ = h.Write([]byte{quant})
	return fmt.Sprintf("%x", h.Sum(nil))
}

func keyFromPath(path string) string {
	base := filepath.Base(path)
	return base[:len(base)-len(storeSuffix)]
}

type boundedBuffer struct {
	buf   []byte
	limit uint64
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	if uint64(len(b.buf))+uint64(len(p)) > b.limit {
		return 0, ErrTooLarge
	}
	b.buf = append(b.buf, p...)
	return len(p), nil
}
