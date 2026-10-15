package nvrtcx

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

const (
	cacheMagic       = "nvrtcx\x00\x08" // Retained in cache-key identity.
	cacheEntryMagic  = "nvrtcx\x00\x09"
	legacyCacheMagic = "nvrtcx\x00\x07"
)

type cacheMeta struct {
	JIT                  cudakernel.JITOptions   `json:"jit,omitempty"`
	Deps                 []cudakernel.Dependency `json:"deps,omitempty"`
	CompilerOptions      []string                `json:"compiler_options,omitempty"`
	CompilerOptionsKnown bool                    `json:"compiler_options_known,omitempty"`
}

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

// compilerCacheKey separates artifacts produced by different loaded compilers.
// Framing both fields prevents an identity from being confused with source input.
func compilerCacheKey(input, compiler string) string {
	return computeCacheKey(input, "nvrtcx-compiler-cache-v1", nil, map[string]string{"compiler": compiler})
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
		if cm.compilerOptionsKnown {
			return cm, true
		}
		// Old cache metadata cannot attest which compiler options produced the
		// artifact. Treat it as a miss so this request recompiles with provenance.
		return nil, false
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
	if !mod.compilerOptionsKnown {
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
	if !cm.validArtifact() {
		return nil, fmt.Errorf("nvrtcx: module contains no compiler artifact")
	}
	if !cm.compilerOptionsKnown && len(cm.compilerOptions) != 0 {
		return nil, fmt.Errorf("nvrtcx: invalid compiler option provenance")
	}
	names, err := json.Marshal(cm.lowered)
	if err != nil {
		return nil, err
	}
	metaBytes, err := json.Marshal(cacheMeta{
		JIT:                  cm.jitOptions,
		Deps:                 cm.dependencies,
		CompilerOptions:      cm.compilerOptions,
		CompilerOptionsKnown: cm.compilerOptionsKnown,
	})
	if err != nil {
		return nil, err
	}
	var b bytes.Buffer
	b.WriteString(cacheEntryMagic)
	b.WriteByte(byte(cm.output))
	fields := [][]byte{[]byte(cm.ptx), cm.cubin, cm.ltoir, cm.tileir, cm.optixir, []byte(cm.log), names, metaBytes}
	for _, field := range fields {
		var length [8]byte
		binary.LittleEndian.PutUint64(length[:], uint64(len(field)))
		b.Write(length[:])
	}
	for _, field := range fields {
		b.Write(field)
	}
	sum := sha256.Sum256(b.Bytes())
	b.Write(sum[:])
	return b.Bytes(), nil
}

func decodeCacheEntry(key string, data []byte) (*CompiledModule, error) {
	isV9 := len(data) >= len(cacheEntryMagic) && string(data[:len(cacheEntryMagic)]) == cacheEntryMagic
	isV8 := len(data) >= len(cacheMagic) && string(data[:len(cacheMagic)]) == cacheMagic
	isV7 := len(data) >= len(legacyCacheMagic) && string(data[:len(legacyCacheMagic)]) == legacyCacheMagic
	if !isV9 && !isV8 && !isV7 {
		return nil, fmt.Errorf("nvrtcx: invalid cache entry")
	}
	numFields := 7
	magicLen := len(legacyCacheMagic)
	if isV9 {
		numFields = 8
		magicLen = len(cacheEntryMagic)
	} else if isV8 {
		numFields = 8
		magicLen = len(cacheMagic)
	}
	headerSize := magicLen + 1 + numFields*8
	if len(data) < headerSize+sha256.Size {
		return nil, fmt.Errorf("nvrtcx: invalid cache entry size")
	}
	want := data[len(data)-sha256.Size:]
	got := sha256.Sum256(data[:len(data)-sha256.Size])
	if !bytes.Equal(got[:], want) {
		return nil, fmt.Errorf("nvrtcx: corrupt cache entry")
	}
	kind := OutputKind(data[magicLen])
	if kind > OutputOptiXIR {
		return nil, fmt.Errorf("nvrtcx: invalid cached output kind")
	}
	payload := data[headerSize : len(data)-sha256.Size]
	fields := make([][]byte, numFields)
	for i := range fields {
		offset := magicLen + 1 + i*8
		size := binary.LittleEndian.Uint64(data[offset : offset+8])
		if size > uint64(len(payload)) {
			return nil, fmt.Errorf("nvrtcx: invalid cache entry lengths")
		}
		fields[i] = payload[:int(size)]
		payload = payload[int(size):]
	}
	if len(payload) != 0 {
		return nil, fmt.Errorf("nvrtcx: trailing cache data")
	}
	cm := &CompiledModule{
		output:  kind,
		ptx:     string(fields[0]),
		cubin:   append([]byte(nil), fields[1]...),
		ltoir:   append([]byte(nil), fields[2]...),
		tileir:  append([]byte(nil), fields[3]...),
		optixir: append([]byte(nil), fields[4]...),
		log:     string(fields[5]),
		key:     key,
	}
	if err := json.Unmarshal(fields[6], &cm.lowered); err != nil {
		return nil, fmt.Errorf("nvrtcx: cached lowered names: %w", err)
	}
	for expression, name := range cm.lowered {
		if expression == "" || name == "" || strings.ContainsRune(expression, 0) || strings.ContainsRune(name, 0) {
			return nil, fmt.Errorf("nvrtcx: invalid cached lowered name")
		}
	}
	if (isV9 || isV8) && len(fields) > 7 && len(fields[7]) > 0 {
		var meta cacheMeta
		if err := json.Unmarshal(fields[7], &meta); err != nil {
			if isV9 {
				return nil, fmt.Errorf("nvrtcx: cached metadata: %w", err)
			}
		} else {
			cm.jitOptions = meta.JIT
			cm.dependencies = meta.Deps
			if isV9 {
				if !meta.CompilerOptionsKnown && len(meta.CompilerOptions) != 0 {
					return nil, fmt.Errorf("nvrtcx: invalid cached compiler option provenance")
				}
				cm.compilerOptions = append([]string(nil), meta.CompilerOptions...)
				cm.compilerOptionsKnown = meta.CompilerOptionsKnown
			}
		}
	}
	if !cm.validArtifact() {
		return nil, fmt.Errorf("nvrtcx: cached output kind does not match artifact")
	}
	return cm, nil
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
	return fmt.Sprintf("CompiledModule(key=%s, ptx_len=%d, cubin_len=%d, ltoir_len=%d, tileir_len=%d, optixir_len=%d)",
		cm.key, len(cm.ptx), len(cm.cubin), len(cm.ltoir), len(cm.tileir), len(cm.optixir))
}
