package nvrtcx

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

var (
	memCacheMu sync.RWMutex
	memCache   = make(map[string]*CompiledModule)

	flightMu    sync.Mutex
	flightCalls = make(map[string]*flightCall)
)

type flightCall struct {
	wg  sync.WaitGroup
	val *CompiledModule
	err error
}

func computeCacheKey(src string, name string, flags []string, includes map[string]string) string {
	h := sha256.New()
	h.Write([]byte(src))
	h.Write([]byte("\x00" + name + "\x00"))
	for _, f := range flags {
		h.Write([]byte(f + "\x00"))
	}
	for k, v := range includes {
		h.Write([]byte(k + "=" + v + "\x00"))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func getCacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	dir := filepath.Join(home, "tmp", "cuda-jit-cache")
	_ = os.MkdirAll(dir, 0755)
	return dir
}

func lookupCache(key string) (*CompiledModule, bool) {
	memCacheMu.RLock()
	cm, ok := memCache[key]
	memCacheMu.RUnlock()
	if ok {
		return cm, true
	}

	dir := getCacheDir()
	if dir == "" {
		return nil, false
	}

	ptxPath := filepath.Join(dir, key+".ptx")
	ptxBytes, err := os.ReadFile(ptxPath)
	if err != nil {
		return nil, false
	}

	cubinPath := filepath.Join(dir, key+".cubin")
	cubinBytes, _ := os.ReadFile(cubinPath)

	mod := &CompiledModule{
		ptx:   string(ptxBytes),
		cubin: cubinBytes,
		key:   key,
	}

	memCacheMu.Lock()
	memCache[key] = mod
	memCacheMu.Unlock()

	return mod, true
}

func storeCache(key string, cm *CompiledModule) {
	memCacheMu.Lock()
	memCache[key] = cm
	memCacheMu.Unlock()

	dir := getCacheDir()
	if dir == "" {
		return
	}

	if cm.ptx != "" {
		_ = os.WriteFile(filepath.Join(dir, key+".ptx"), []byte(cm.ptx), 0644)
	}
	if len(cm.cubin) > 0 {
		_ = os.WriteFile(filepath.Join(dir, key+".cubin"), cm.cubin, 0644)
	}
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

	dir := getCacheDir()
	if dir != "" {
		entries, err := os.ReadDir(dir)
		if err == nil {
			for _, e := range entries {
				_ = os.Remove(filepath.Join(dir, e.Name()))
			}
		}
	}
	return nil
}

func (cm *CompiledModule) Key() string {
	return cm.key
}

func (cm *CompiledModule) String() string {
	return fmt.Sprintf("CompiledModule(key=%s, ptx_len=%d, cubin_len=%d)", cm.key, len(cm.ptx), len(cm.cubin))
}
