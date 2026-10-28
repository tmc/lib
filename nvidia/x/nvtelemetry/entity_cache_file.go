package nvtelemetry

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

// EntityCacheScope identifies the host and clock in which LoadFile's now is
// meaningful. All four fields are required. A matching scope does not establish
// that a saved assignment's supplying authority is authentic.
type EntityCacheScope struct {
	BootID      string
	NodeID      string
	ClockDomain string
	ClockEpoch  string
}

type entityCacheFile struct {
	Version int             `json:"version"`
	Data    json.RawMessage `json:"data"`
	SHA256  string          `json:"sha256"`
}
type entityCacheData struct {
	TTL       int64                  `json:"ttl_nanos"`
	Entries   []entityCacheFileEntry `json:"entries"`
	Completed []EntityOwner          `json:"completed_jobs"`
}
type entityCacheFileEntry struct {
	Inserted    int64              `json:"inserted_nanos"`
	Expires     int64              `json:"expires_nanos"`
	Assignments []EntityAssignment `json:"assignments"`
}

// SaveFile atomically replaces path with a bounded versioned cache snapshot.
// The caller must exclusively own its existing private parent directory and
// serialize other processes using the same file. Path must be absolute, clean,
// outside OS temporary directories and free of symlinks. Use a persistent
// project scratch directory; this method creates neither parent directories nor
// a second attribution model. Files have mode 0600.
//
// SaveFile syncs the new file, renames it and syncs the parent directory. An
// error after rename can leave the new file installed; an error is not a
// rollback promise. The checksum detects corruption, not malicious replacement
// or rollback, and persistence does not authenticate or upgrade metadata trust.
func (c *EntityCache) SaveFile(path string) (err error) {
	root, name, err := entityCacheRoot(path)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, root.Close()) }()
	if c == nil {
		return fmt.Errorf("entity cache: nil cache")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		return fmt.Errorf("entity cache: uninitialized cache")
	}
	data := entityCacheData{TTL: c.ttl}
	var keys []entityKey
	for key := range c.entries {
		keys = append(keys, key)
	}
	// JSON strings give a deterministic order across all typed key fields.
	sort.Slice(keys, func(i, j int) bool {
		a, b := c.entries[keys[i]].assignments[0], c.entries[keys[j]].assignments[0]
		x, _ := json.Marshal(a)
		y, _ := json.Marshal(b)
		return bytes.Compare(x, y) < 0
	})
	count := 0
	for _, key := range keys {
		e := c.entries[key]
		count += len(e.assignments)
		if count > entityMaxRecords {
			return fmt.Errorf("entity cache: assignment limit")
		}
		data.Entries = append(data.Entries, entityCacheFileEntry{e.inserted, e.expires, e.assignments})
	}
	for owner := range c.completed {
		data.Completed = append(data.Completed, owner)
	}
	sort.Slice(data.Completed, func(i, j int) bool {
		a, _ := json.Marshal(data.Completed[i])
		b, _ := json.Marshal(data.Completed[j])
		return bytes.Compare(a, b) < 0
	})
	body, err := json.Marshal(data)
	if err != nil || len(body) > entityMaxBytes {
		return fmt.Errorf("entity cache: snapshot too large or invalid")
	}
	sum := sha256.Sum256(body)
	encoded, err := json.Marshal(entityCacheFile{1, body, hex.EncodeToString(sum[:])})
	if err != nil || len(encoded) >= entityMaxBytes {
		return fmt.Errorf("entity cache: snapshot too large or invalid")
	}
	encoded = append(encoded, '\n')
	var nonce [16]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return fmt.Errorf("entity cache: temporary name: %w", err)
	}
	tmp := ".entity-cache-" + hex.EncodeToString(nonce[:])
	f, err := root.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return fmt.Errorf("entity cache: create snapshot: %w", err)
	}
	defer func() {
		if e := root.Remove(tmp); e != nil && !errors.Is(e, os.ErrNotExist) {
			err = errors.Join(err, fmt.Errorf("entity cache: remove temporary: %w", e))
		}
	}()
	_, writeErr := f.Write(encoded)
	if writeErr == nil {
		writeErr = f.Sync()
	}
	if err := errors.Join(writeErr, f.Close()); err != nil {
		return fmt.Errorf("entity cache: write snapshot: %w", err)
	}
	if err := entityCacheOrdinary(root, name); err != nil {
		return err
	}
	if err := root.Rename(tmp, name); err != nil {
		return fmt.Errorf("entity cache: replace snapshot: %w", err)
	}
	dir, err := root.Open(".")
	if err != nil {
		return fmt.Errorf("entity cache: open parent: %w", err)
	}
	if err := errors.Join(dir.Sync(), dir.Close()); err != nil {
		return fmt.Errorf("entity cache: sync parent: %w", err)
	}
	return nil
}

// LoadFile replaces relationships with current entries from path. It preserves
// their original expiry and caps it by this cache's TTL from the original insert
// time. Entries from other hosts, boots or clock domains/epochs are discarded.
// Completion markers already in memory and those saved in the file prevent
// stale relationships for an exactly matching owner from being loaded again.
// Clear explicitly resets those markers.
//
// Missing files leave a clean relationship cache. Corrupt, oversized, invalid
// or unsupported files also leave it clean and return discarded=true, err=nil;
// the corrupt file is retained until a successful SaveFile replaces it. I/O and
// path errors return an error and leave memory unchanged. The file checksum is
// not authentication or rollback protection. No TTL restarts on load.
func (c *EntityCache) LoadFile(path string, scope EntityCacheScope, now int64) (discarded bool, err error) {
	if c == nil || scope.BootID == "" || scope.NodeID == "" || scope.ClockDomain == "" || scope.ClockEpoch == "" {
		return false, fmt.Errorf("entity cache: missing cache or clock scope")
	}
	root, name, err := entityCacheRoot(path)
	if err != nil {
		return false, err
	}
	defer func() {
		if root != nil {
			err = errors.Join(err, root.Close())
		}
	}()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		return false, fmt.Errorf("entity cache: uninitialized cache")
	}
	f, err := root.Open(name)
	if errors.Is(err, os.ErrNotExist) {
		closeErr := root.Close()
		root = nil
		if closeErr != nil {
			return false, fmt.Errorf("entity cache: close parent: %w", closeErr)
		}
		clear(c.entries)
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("entity cache: open snapshot: %w", err)
	}
	info, statErr := f.Stat()
	pathInfo, pathErr := root.Lstat(name)
	if statErr != nil || pathErr != nil || !info.Mode().IsRegular() || !pathInfo.Mode().IsRegular() || !os.SameFile(info, pathInfo) {
		return false, errors.Join(fmt.Errorf("entity cache: nonordinary or changed snapshot"), statErr, pathErr, f.Close())
	}
	b, readErr := io.ReadAll(io.LimitReader(f, entityMaxBytes+1))
	if err := errors.Join(readErr, f.Close()); err != nil {
		return false, fmt.Errorf("entity cache: read snapshot: %w", err)
	}
	// Complete all fallible I/O before replacing or discarding live entries.
	closeErr := root.Close()
	root = nil
	if closeErr != nil {
		return false, fmt.Errorf("entity cache: close parent: %w", closeErr)
	}
	data, err := decodeEntityCache(b, c.limit)
	if err != nil {
		clear(c.entries)
		return true, nil
	}
	completed := make(map[EntityOwner]bool, len(c.completed)+len(data.Completed))
	for owner := range c.completed {
		completed[owner] = true
	}
	for _, owner := range data.Completed {
		completed[owner] = true
	}
	if len(completed) > c.limit {
		clear(c.entries)
		return true, nil
	}
	entries := make(map[entityKey]entityCacheEntry)
	for _, saved := range data.Entries {
		a := saved.Assignments[0]
		key, _ := assignmentKey(a)
		if key.boot != scope.BootID || key.node != scope.NodeID || key.domain != scope.ClockDomain || key.epoch != scope.ClockEpoch {
			continue
		}
		expires := saved.Expires
		if saved.Inserted <= int64(^uint64(0)>>1)-c.ttl && saved.Inserted+c.ttl < expires {
			expires = saved.Inserted + c.ttl
		}
		if now < saved.Inserted || now >= expires {
			continue
		}
		keep := true
		for _, a := range saved.Assignments {
			if now < a.StartNanos || completed[a.Owner] {
				keep = false
			}
		}
		if keep {
			entries[key] = entityCacheEntry{saved.Assignments, expires, saved.Inserted}
		}
	}
	c.entries, c.completed = entries, completed
	return false, nil
}

func decodeEntityCache(b []byte, capacity int) (entityCacheData, error) {
	var file entityCacheFile
	if err := readEntityJSON(bytes.NewReader(b), &file, true); err != nil {
		return entityCacheData{}, err
	}
	if file.Version != 1 || len(file.SHA256) != 64 {
		return entityCacheData{}, fmt.Errorf("entity cache: invalid version or checksum")
	}
	sum := sha256.Sum256(file.Data)
	if file.SHA256 != hex.EncodeToString(sum[:]) {
		return entityCacheData{}, fmt.Errorf("entity cache: checksum mismatch")
	}
	var data entityCacheData
	if err := readEntityJSON(bytes.NewReader(file.Data), &data, true); err != nil {
		return data, err
	}
	if data.TTL <= 0 || len(data.Entries) > capacity || len(data.Completed) > capacity {
		return data, fmt.Errorf("entity cache: invalid bounds")
	}
	completed := make(map[EntityOwner]bool)
	for _, owner := range data.Completed {
		if owner.JobID == "" || completed[owner] {
			return data, fmt.Errorf("entity cache: invalid completion marker")
		}
		completed[owner] = true
	}
	keys := make(map[entityKey]bool)
	count := 0
	for _, e := range data.Entries {
		count += len(e.Assignments)
		if len(e.Assignments) == 0 || count > entityMaxRecords || e.Expires <= e.Inserted || e.Inserted <= int64(^uint64(0)>>1)-data.TTL && e.Expires > e.Inserted+data.TTL {
			return data, fmt.Errorf("entity cache: invalid expiry or assignment count")
		}
		var key entityKey
		for i, a := range e.Assignments {
			if err := a.Validate(); err != nil {
				return data, err
			}
			k, err := assignmentKey(a)
			if err != nil || i != 0 && k != key || e.Inserted < a.StartNanos || e.Expires > a.EndNanos {
				return data, fmt.Errorf("entity cache: invalid cached relationship")
			}
			key = k
		}
		if keys[key] {
			return data, fmt.Errorf("entity cache: duplicate cache key")
		}
		keys[key] = true
	}
	return data, nil
}

func entityCacheOrdinary(root *os.Root, name string) error {
	info, err := root.Lstat(name)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("entity cache: snapshot metadata: %w", err)
	}
	if !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 {
		return fmt.Errorf("entity cache: snapshot must be a private regular file")
	}
	return nil
}

func entityCacheRoot(path string) (*os.Root, string, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || filepath.Base(path) == "." {
		return nil, "", fmt.Errorf("entity cache: path must be clean and absolute")
	}
	for _, tmp := range []string{"/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/var/folders", "/private/var/folders"} {
		if path == tmp || len(path) > len(tmp) && path[:len(tmp)+1] == tmp+string(filepath.Separator) {
			return nil, "", fmt.Errorf("entity cache: temporary cache path")
		}
	}
	parent := filepath.Dir(path)
	resolved, err := filepath.EvalSymlinks(parent)
	if err != nil || resolved != parent {
		return nil, "", fmt.Errorf("entity cache: missing or ambiguous parent directory")
	}
	info, err := os.Stat(parent)
	if err != nil || !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		return nil, "", fmt.Errorf("entity cache: parent must be a private directory")
	}
	root, err := os.OpenRoot(parent)
	if err != nil {
		return nil, "", fmt.Errorf("entity cache: open parent: %w", err)
	}
	name := filepath.Base(path)
	if err := entityCacheOrdinary(root, name); err != nil {
		return nil, "", errors.Join(err, root.Close())
	}
	return root, name, nil
}
