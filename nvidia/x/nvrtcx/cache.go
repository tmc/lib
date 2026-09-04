package nvrtcx

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

const cacheMagic = "nvrtcx\x00\x03"

var (
	memCacheMu sync.RWMutex
	memCache   = make(map[string]*CompiledModule)

	flightMu         sync.Mutex
	flightCalls      = make(map[string]*flightCall)
	cacheDirOverride string
)

type flightCall struct {
	wg  sync.WaitGroup
	val *CompiledModule
	err error
}

func computeCacheKey(src string, name string, flags []string, includes map[string]string) string {
	h := sha256.New()
	writeHashField := func(value string) {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(value)))
		h.Write(size[:])
		h.Write([]byte(value))
	}
	writeHashField("nvrtcx-cache-v2")
	writeHashField("source")
	writeHashField(src)
	writeHashField("name")
	writeHashField(name)
	writeHashField("flags")
	writeHashField(fmt.Sprint(len(flags)))
	for _, f := range flags {
		writeHashField(f)
	}
	names := make([]string, 0, len(includes))
	for name := range includes {
		names = append(names, name)
	}
	sort.Strings(names)
	writeHashField("headers")
	writeHashField(fmt.Sprint(len(names)))
	for _, k := range names {
		v := includes[k]
		writeHashField(k)
		writeHashField(v)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func getCacheDir() (string, error) {
	if cacheDirOverride != "" {
		return cacheDirOverride, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("nvrtcx: locate user cache: %w", err)
	}
	dir := filepath.Join(base, "tmc-lib", "nvrtcx")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", fmt.Errorf("nvrtcx: create cache directory: %w", err)
	}
	return dir, nil
}

func lookupCache(key string) (*CompiledModule, bool) {
	memCacheMu.RLock()
	cm, ok := memCache[key]
	memCacheMu.RUnlock()
	if ok {
		return cm, true
	}

	dir, err := getCacheDir()
	if err != nil {
		return nil, false
	}

	data, err := os.ReadFile(filepath.Join(dir, key+".cache"))
	if err != nil {
		return nil, false
	}
	mod, err := decodeCacheEntry(key, data)
	if err != nil {
		return nil, false
	}

	memCacheMu.Lock()
	memCache[key] = mod
	memCacheMu.Unlock()

	return mod, true
}

func storeCache(key string, cm *CompiledModule) error {
	dir, err := getCacheDir()
	if err != nil {
		return err
	}

	data, err := encodeCacheEntry(cm)
	if err != nil {
		return err
	}
	if err := writeCacheFile(filepath.Join(dir, key+".cache"), data); err != nil {
		return err
	}
	memCacheMu.Lock()
	memCache[key] = cm
	memCacheMu.Unlock()
	return nil
}

func encodeCacheEntry(cm *CompiledModule) ([]byte, error) {
	if cm == nil || cm.ptx == "" && len(cm.cubin) == 0 {
		return nil, fmt.Errorf("nvrtcx: module contains no ptx or cubin image")
	}
	var b bytes.Buffer
	b.WriteString(cacheMagic)
	var lengths [16]byte
	binary.LittleEndian.PutUint64(lengths[:8], uint64(len(cm.ptx)))
	binary.LittleEndian.PutUint64(lengths[8:], uint64(len(cm.cubin)))
	b.Write(lengths[:])
	b.WriteString(cm.ptx)
	b.Write(cm.cubin)
	sum := sha256.Sum256(b.Bytes())
	b.Write(sum[:])
	return b.Bytes(), nil
}

func decodeCacheEntry(key string, data []byte) (*CompiledModule, error) {
	const headerSize = len(cacheMagic) + 16
	if len(data) < headerSize+sha256.Size || string(data[:len(cacheMagic)]) != cacheMagic {
		return nil, fmt.Errorf("nvrtcx: invalid cache entry")
	}
	want := data[len(data)-sha256.Size:]
	got := sha256.Sum256(data[:len(data)-sha256.Size])
	if !bytes.Equal(got[:], want) {
		return nil, fmt.Errorf("nvrtcx: corrupt cache entry")
	}
	ptxLen := binary.LittleEndian.Uint64(data[len(cacheMagic) : len(cacheMagic)+8])
	cubinLen := binary.LittleEndian.Uint64(data[len(cacheMagic)+8 : headerSize])
	payloadLen := uint64(len(data) - headerSize - sha256.Size)
	if ptxLen > payloadLen || cubinLen != payloadLen-ptxLen || ptxLen == 0 && cubinLen == 0 {
		return nil, fmt.Errorf("nvrtcx: invalid cache entry lengths")
	}
	payload := data[headerSize : len(data)-sha256.Size]
	return &CompiledModule{
		ptx:   string(payload[:ptxLen]),
		cubin: append([]byte(nil), payload[ptxLen:]...),
		key:   key,
	}, nil
}

func writeCacheFile(path string, data []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".nvrtcx-*")
	if err != nil {
		return fmt.Errorf("nvrtcx: create cache file: %w", err)
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if err := f.Chmod(0o600); err != nil {
		f.Close()
		return fmt.Errorf("nvrtcx: protect cache file: %w", err)
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("nvrtcx: write cache file: %w", err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("nvrtcx: sync cache file: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("nvrtcx: close cache file: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("nvrtcx: install cache file: %w", err)
	}
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return fmt.Errorf("nvrtcx: open cache directory: %w", err)
	}
	defer dir.Close()
	if err := dir.Sync(); err != nil {
		return fmt.Errorf("nvrtcx: sync cache directory: %w", err)
	}
	return nil
}

// SingleFlightCompile deduplicates concurrent JIT compilation passes for the same key.
func singleFlightCompile(key string, fn func() (*CompiledModule, error)) (*CompiledModule, error) {
	flightMu.Lock()
	if c, ok := flightCalls[key]; ok {
		flightMu.Unlock()
		c.wg.Wait()
		return c.val, c.err
	}

	c := &flightCall{}
	c.wg.Add(1)
	flightCalls[key] = c
	flightMu.Unlock()

	c.val, c.err = fn()
	c.wg.Done()

	flightMu.Lock()
	delete(flightCalls, key)
	flightMu.Unlock()

	return c.val, c.err
}

// ClearCache removes all in-memory and disk-cached JIT compilation units.
func ClearCache() error {
	memCacheMu.Lock()
	memCache = make(map[string]*CompiledModule)
	memCacheMu.Unlock()

	dir, err := getCacheDir()
	if err != nil {
		return err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("nvrtcx: read cache directory: %w", err)
	}
	var errs []error
	for _, entry := range entries {
		name := entry.Name()
		if entry.Type().IsRegular() && (strings.HasSuffix(name, ".cache") || strings.HasSuffix(name, ".ptx") || strings.HasSuffix(name, ".cubin")) {
			if err := os.Remove(filepath.Join(dir, name)); err != nil && !os.IsNotExist(err) {
				errs = append(errs, fmt.Errorf("nvrtcx: remove cache entry %q: %w", name, err))
			}
		}
	}
	return errors.Join(errs...)
}

func (cm *CompiledModule) Key() string {
	if cm == nil {
		return ""
	}
	return cm.key
}

func (cm *CompiledModule) String() string {
	if cm == nil {
		return "CompiledModule(<nil>)"
	}
	return fmt.Sprintf("CompiledModule(key=%s, ptx_len=%d, cubin_len=%d)", cm.key, len(cm.ptx), len(cm.cubin))
}
