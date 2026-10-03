package cudaprofilehistory

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

// Limits bound encoded record storage and all snapshot input/output bytes,
// including the header and newlines. MaxRecordBytes bounds one JSON row.
type Limits struct {
	MaxRecords     int `json:"max_records"`
	MaxBytes       int `json:"max_bytes"`
	MaxRecordBytes int `json:"max_record_bytes"`
}

func (l Limits) validate() error {
	if l.MaxRecords <= 0 || l.MaxRecords > 100000 || l.MaxBytes <= 0 || l.MaxBytes > 64<<20 || l.MaxRecordBytes <= 0 || l.MaxRecordBytes > l.MaxBytes {
		return fmt.Errorf("history: invalid store bounds")
	}
	minimum, _ := json.Marshal(snapshotHeader{Version: Version, RowsDigest: digest(nil)})
	if l.MaxBytes < len(minimum)+1 {
		return fmt.Errorf("history: snapshot header exceeds byte bound")
	}
	return nil
}

// Store owns independent encoded rows. The zero value is unusable; use NewStore.
type Store struct {
	mu     sync.RWMutex
	limits Limits
	rows   [][]byte
	keys   map[string]bool
	bytes  int
}

type snapshotHeader struct {
	Version    int    `json:"version"`
	Count      int    `json:"count"`
	RowsDigest string `json:"rows_digest"`
}

func (s *Store) header() snapshotHeader {
	h := sha256.New()
	for _, row := range s.rows {
		h.Write(row)
		h.Write([]byte{'\n'})
	}
	return snapshotHeader{Version: Version, Count: len(s.rows), RowsDigest: hex.EncodeToString(h.Sum(nil))}
}

// NewStore constructs an empty bounded store.
func NewStore(l Limits) (*Store, error) {
	if err := l.validate(); err != nil {
		return nil, err
	}
	return &Store{limits: l, keys: map[string]bool{}}, nil
}

// Append adds a validated observation once. Duplicate keys never overwrite rows.
func (s *Store) Append(r Record) (string, error) {
	if s == nil {
		return "", fmt.Errorf("history: nil store")
	}
	if err := boundedValue(reflect.ValueOf(r), s.limits.MaxRecordBytes); err != nil {
		return "", err
	}
	key, err := ObservationKey(r)
	if err != nil {
		return "", err
	}
	r.Key = key
	data, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.keys == nil {
		return "", fmt.Errorf("history: uninitialized store")
	}
	if s.keys[key] {
		return "", fmt.Errorf("history: duplicate observation")
	}
	header, _ := json.Marshal(snapshotHeader{Version: Version, Count: len(s.rows) + 1, RowsDigest: strings.Repeat("0", 64)})
	if len(data) > s.limits.MaxRecordBytes || len(s.rows) >= s.limits.MaxRecords || len(data)+1 > s.limits.MaxBytes-len(header)-1-s.bytes {
		return "", fmt.Errorf("history: store budget exceeded")
	}
	s.rows = append(s.rows, append([]byte(nil), data...))
	s.keys[key] = true
	s.bytes += len(data) + 1
	return key, nil
}

// Records decodes independent copies in append order.
func (s *Store) Records() ([]Record, error) {
	if s == nil {
		return nil, fmt.Errorf("history: nil store")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.keys == nil {
		return nil, fmt.Errorf("history: uninitialized store")
	}
	rows := make([]Record, 0, len(s.rows))
	for _, b := range s.rows {
		r, err := DecodeRecord(b, s.limits.MaxRecordBytes)
		if err != nil {
			return nil, err
		}
		rows = append(rows, r)
	}
	return rows, nil
}

// WriteSnapshot writes bounded NDJSON from a stable snapshot. It does not make
// publication atomic; callers own temporary-file replacement and file locking.
func (s *Store) WriteSnapshot(w io.Writer) error {
	if s == nil {
		return fmt.Errorf("history: nil store")
	}
	s.mu.RLock()
	if s.keys == nil {
		s.mu.RUnlock()
		return fmt.Errorf("history: uninitialized store")
	}
	header, _ := json.Marshal(s.header())
	all := make([]byte, 0, len(header)+1+s.bytes)
	all = append(all, header...)
	all = append(all, '\n')
	for _, row := range s.rows {
		all = append(all, row...)
		all = append(all, '\n')
	}
	s.mu.RUnlock()
	n, err := w.Write(all)
	if err == nil && n != len(all) {
		err = io.ErrShortWrite
	}
	return err
}

// ReadSnapshot reads a new store. An invalid row never exposes a partial import.
// It caps original input bytes before decoding and enforces row/count bounds.
func ReadSnapshot(r io.Reader, l Limits) (*Store, error) {
	s, err := NewStore(l)
	if err != nil {
		return nil, err
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(l.MaxBytes)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > l.MaxBytes {
		return nil, fmt.Errorf("history: snapshot byte budget exceeded")
	}
	scanner := bufio.NewScanner(bytes.NewReader(b))
	scanner.Buffer(make([]byte, min(4096, max(256, l.MaxRecordBytes+1))), max(256, l.MaxRecordBytes+1))
	if !scanner.Scan() {
		return nil, fmt.Errorf("history: missing snapshot header")
	}
	var h snapshotHeader
	if err := strictJSON(scanner.Bytes(), &h); err != nil {
		return nil, err
	}
	if h.Version != Version || h.Count < 0 || h.Count > l.MaxRecords || !validDigest(h.RowsDigest) {
		return nil, fmt.Errorf("history: invalid snapshot version")
	}
	for scanner.Scan() {
		if len(scanner.Bytes()) == 0 {
			return nil, fmt.Errorf("history: empty snapshot row")
		}
		r, err := DecodeRecord(scanner.Bytes(), l.MaxRecordBytes)
		if err != nil {
			return nil, err
		}
		if r.Key == "" {
			return nil, fmt.Errorf("history: snapshot row missing observation key")
		}
		if _, err = s.Append(r); err != nil {
			return nil, err
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("history row: %w", err)
	}
	if h != s.header() {
		return nil, fmt.Errorf("history: snapshot count or ordered row digest mismatch")
	}
	return s, nil
}

// DecodeRecord rejects duplicate, case-alias, unknown and trailing JSON fields.
// maxBytes caps original input before decoding, including unknown fields.
func DecodeRecord(b []byte, maxBytes int) (Record, error) {
	var r Record
	if maxBytes <= 0 || maxBytes > 64<<20 || len(b) > maxBytes {
		return r, fmt.Errorf("history: record byte budget exceeded")
	}
	if err := strictJSON(b, &r); err != nil {
		return Record{}, err
	}
	if err := r.Validate(); err != nil {
		return Record{}, err
	}
	return r, nil
}

func strictJSON(b []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := checkJSON(d, reflect.TypeOf(out).Elem(), 0); err != nil {
		return fmt.Errorf("history json: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("history json: trailing data")
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	return d.Decode(out)
}

// boundedValue limits caller-owned collections before JSON creates copies.
func boundedValue(v reflect.Value, limit int) error {
	remaining := limit
	var walk func(reflect.Value) bool
	walk = func(v reflect.Value) bool {
		if remaining < 0 {
			return false
		}
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				remaining -= 4
				return remaining >= 0
			}
			return walk(v.Elem())
		}
		if v.Type() == reflect.TypeOf(cudaevidence.Value{}) {
			return walk(reflect.ValueOf(v.Interface().(cudaevidence.Value).Record()))
		}
		remaining--
		switch v.Kind() {
		case reflect.String:
			remaining -= v.Len()
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				if v.Type().Field(i).PkgPath == "" && !walk(v.Field(i)) {
					return false
				}
			}
		case reflect.Slice, reflect.Array:
			if v.Type().Elem().Kind() == reflect.Uint8 {
				remaining -= v.Len()
				break
			}
			if v.Len() > remaining {
				return false
			}
			for i := 0; i < v.Len(); i++ {
				if !walk(v.Index(i)) {
					return false
				}
			}
		}
		return remaining >= 0
	}
	if limit <= 0 || !walk(v) {
		return fmt.Errorf("history: record allocation budget exceeded")
	}
	return nil
}

// checkJSON verifies exact schema spelling even for types with custom decoders.
func checkJSON(d *json.Decoder, t reflect.Type, depth int) error {
	if depth > 64 {
		return fmt.Errorf("nesting limit exceeded")
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == reflect.TypeOf(cudaevidence.Value{}) {
		t = reflect.TypeOf(cudaevidence.Record{})
	}
	tok, err := d.Token()
	if err != nil {
		return err
	}
	if tok == nil {
		return nil
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		fields := map[string]reflect.Type{}
		if t.Kind() == reflect.Struct {
			for i := 0; i < t.NumField(); i++ {
				f := t.Field(i)
				if f.PkgPath != "" {
					continue
				}
				name := strings.Split(f.Tag.Get("json"), ",")[0]
				if name == "-" {
					continue
				}
				if name == "" {
					name = f.Name
				}
				fields[name] = f.Type
			}
		}
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok {
				return fmt.Errorf("invalid object key")
			}
			if seen[name] {
				return fmt.Errorf("duplicate field %q", name)
			}
			seen[name] = true
			child, ok := fields[name]
			if t.Kind() == reflect.Map {
				child = t.Elem()
				ok = true
			}
			if !ok {
				return fmt.Errorf("unknown field %q", name)
			}
			if err := checkJSON(d, child, depth+1); err != nil {
				return err
			}
		}
		end, err := d.Token()
		if err != nil {
			return err
		}
		if end != json.Delim('}') {
			return fmt.Errorf("invalid object close")
		}
	case '[':
		if t.Kind() != reflect.Slice && t.Kind() != reflect.Array {
			return fmt.Errorf("unexpected array")
		}
		for d.More() {
			if err := checkJSON(d, t.Elem(), depth+1); err != nil {
				return err
			}
		}
		end, err := d.Token()
		if err != nil {
			return err
		}
		if end != json.Delim(']') {
			return fmt.Errorf("invalid array close")
		}
	default:
		return fmt.Errorf("unexpected delimiter")
	}
	return nil
}
