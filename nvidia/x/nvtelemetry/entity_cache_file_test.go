package nvtelemetry

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func entityCacheTestPath(t *testing.T) string {
	t.Helper()
	base, err := os.UserCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	base = filepath.Join(base, "nvtelemetry-cache-tests")
	if err := os.MkdirAll(base, 0700); err != nil {
		t.Fatal(err)
	}
	base, err = filepath.EvalSymlinks(base)
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "cache-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Error(err)
		}
	})
	return filepath.Join(dir, "entities.json")
}

func entityCacheTestScope(a EntityAssignment) EntityCacheScope {
	return EntityCacheScope{a.Process.BootID, a.Process.NodeID, a.ClockDomain, a.ClockEpoch}
}

func newEntityFileCache(t *testing.T, capacity int, ttl int64) *EntityCache {
	t.Helper()
	c, err := NewEntityCache(capacity, ttl)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestEntityCacheFileRoundTrip(t *testing.T) {
	path := entityCacheTestPath(t)
	a := entityAssignment()
	a.Provenance = []EntityProvenance{{a.Source, a.SourceRevision, a.StartNanos, a.EndNanos}}
	c := newEntityFileCache(t, 3, 10)
	if err := c.Put([]EntityAssignment{a}, 5); err != nil {
		t.Fatal(err)
	}
	if len(c.Get(a, 4)) != 0 {
		t.Fatal("cache served a relationship before insertion")
	}
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(before, after) {
		t.Fatal("snapshot not deterministic")
	}
	info, _ := os.Stat(path)
	if info.Mode().Perm() != 0600 {
		t.Fatal(info.Mode())
	}
	loaded := newEntityFileCache(t, 3, 100)
	if discarded, err := loaded.LoadFile(path, entityCacheTestScope(a), 14); err != nil || discarded {
		t.Fatal(discarded, err)
	}
	got := loaded.Get(a, 14)
	if len(got) != 1 || !reflect.DeepEqual(got[0], a) {
		t.Fatalf("metadata trust/provenance changed: %#v", got)
	}
	got[0].Process.NamespacePIDs[0]++
	got[0].Provenance[0].Revision = "mutated"
	if !reflect.DeepEqual(loaded.Get(a, 14)[0], a) {
		t.Fatal("loaded snapshot aliases returned metadata")
	}
	if len(loaded.Get(a, 15)) != 0 {
		t.Fatal("load reset original TTL")
	}
	shorter := newEntityFileCache(t, 3, 5)
	if discarded, err := shorter.LoadFile(path, entityCacheTestScope(a), 10); err != nil || discarded || len(shorter.Get(a, 10)) != 0 {
		t.Fatal("configured TTL did not cap original insert", discarded, err)
	}
	entries, _ := os.ReadDir(filepath.Dir(path))
	if len(entries) != 1 || entries[0].Name() != "entities.json" {
		t.Fatal("temporary snapshot leaked", entries)
	}
}

func TestEntityCacheFileScopeAndExpiry(t *testing.T) {
	a := entityAssignment()
	path := entityCacheTestPath(t)
	c := newEntityFileCache(t, 4, 10)
	if err := c.Put([]EntityAssignment{a}, 5); err != nil {
		t.Fatal(err)
	}
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		change func(*EntityCacheScope)
		now    int64
	}{
		{"boot", func(s *EntityCacheScope) { s.BootID = "next-boot" }, 6},
		{"node", func(s *EntityCacheScope) { s.NodeID = "another-node" }, 6},
		{"domain", func(s *EntityCacheScope) { s.ClockDomain = "wall" }, 6},
		{"epoch", func(s *EntityCacheScope) { s.ClockEpoch = "next-epoch" }, 6},
		{"expired", func(*EntityCacheScope) {}, 15},
		{"clock before insert", func(*EntityCacheScope) {}, 4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			scope := entityCacheTestScope(a)
			tt.change(&scope)
			loaded := newEntityFileCache(t, 4, 100)
			if discarded, err := loaded.LoadFile(path, scope, tt.now); err != nil || discarded || len(loaded.Get(a, 6)) != 0 {
				t.Fatal("stale/wrong-domain entry loaded", discarded, err)
			}
		})
	}
	for _, state := range []string{"unknown", "contradictory"} {
		a.State, a.Authoritative = state, false
		if err := c.Put([]EntityAssignment{a}, 5); err != nil {
			t.Fatal(err)
		}
		if err := c.SaveFile(path); err != nil {
			t.Fatal(err)
		}
		loaded := newEntityFileCache(t, 4, 10)
		if _, err := loaded.LoadFile(path, entityCacheTestScope(a), 6); err != nil {
			t.Fatal(err)
		}
		if got := loaded.Get(a, 6); len(got) != 1 || got[0].Authoritative || got[0].State != state {
			t.Fatal("persistence upgraded relationship trust", got)
		}
	}
}

func cacheFileData(t *testing.T, path string) entityCacheData {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data, err := decodeEntityCache(b, 4096)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
func cacheWriteData(t *testing.T, path string, data any) {
	t.Helper()
	body, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	b, err := json.Marshal(entityCacheFile{1, body, hex.EncodeToString(sum[:])})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestEntityCacheFileCorruptionDiscarded(t *testing.T) {
	a := entityAssignment()
	for _, name := range []string{"truncated", "checksum", "version", "unknown outer", "duplicate outer", "unknown data", "duplicate key", "expiry", "mixed key", "oversized", "capacity", "negative ttl"} {
		t.Run(name, func(t *testing.T) {
			path := entityCacheTestPath(t)
			c := newEntityFileCache(t, 2, 10)
			if err := c.Put([]EntityAssignment{a}, 0); err != nil {
				t.Fatal(err)
			}
			if err := c.SaveFile(path); err != nil {
				t.Fatal(err)
			}
			b, _ := os.ReadFile(path)
			data := cacheFileData(t, path)
			switch name {
			case "truncated":
				b = b[:len(b)/2]
			case "checksum":
				b = bytes.Replace(b, []byte(`"sha256":"`), []byte(`"sha256":"00`), 1)
			case "version":
				b = bytes.Replace(b, []byte(`"version":1`), []byte(`"version":2`), 1)
			case "unknown outer":
				b = append([]byte(`{"unexpected":1,`), b[1:]...)
			case "duplicate outer":
				b = append([]byte(`{"version":1,`), b[1:]...)
			case "oversized":
				b = bytes.Repeat([]byte(" "), entityMaxBytes+1)
			case "unknown data":
				cacheWriteData(t, path, map[string]any{"ttl_nanos": 10, "unexpected": 1})
				b, _ = os.ReadFile(path)
			case "duplicate key", "capacity":
				data.Entries = append(data.Entries, data.Entries[0])
				cacheWriteData(t, path, data)
				b, _ = os.ReadFile(path)
			case "expiry":
				data.Entries[0].Expires = 99
				cacheWriteData(t, path, data)
				b, _ = os.ReadFile(path)
			case "mixed key":
				other := a
				other.Process.Process.Generation++
				data.Entries[0].Assignments = append(data.Entries[0].Assignments, other)
				cacheWriteData(t, path, data)
				b, _ = os.ReadFile(path)
			case "negative ttl":
				data.TTL = -1
				cacheWriteData(t, path, data)
				b, _ = os.ReadFile(path)
			}
			if err := os.WriteFile(path, b, 0600); err != nil {
				t.Fatal(err)
			}
			if name == "capacity" {
				c = newEntityFileCache(t, 1, 10)
				if err := c.Put([]EntityAssignment{a}, 0); err != nil {
					t.Fatal(err)
				}
			}
			if discarded, err := c.LoadFile(path, entityCacheTestScope(a), 1); err != nil || !discarded || len(c.Get(a, 1)) != 0 {
				t.Fatal("corrupt snapshot did not leave clean cache", discarded, err)
			}
			if after, _ := os.ReadFile(path); !bytes.Equal(b, after) {
				t.Fatal("corrupt evidence removed or rewritten")
			}
		})
	}
}

func TestEntityCacheFilePathAndIOErrors(t *testing.T) {
	path := entityCacheTestPath(t)
	a := entityAssignment()
	c := newEntityFileCache(t, 2, 10)
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"relative", "temporary", "unclean", "parent symlink", "file symlink", "directory", "public parent", "missing parent"} {
		t.Run(name, func(t *testing.T) {
			p := path
			switch name {
			case "relative":
				p = "entities.json"
			case "temporary":
				p = "/private/tmp/nvtelemetry/entities.json"
			case "unclean":
				p = filepath.Dir(path) + "/./entities.json"
			case "parent symlink":
				link := filepath.Join(filepath.Dir(path), "alias")
				if err := os.Symlink(filepath.Dir(path), link); err != nil {
					t.Fatal(err)
				}
				p = filepath.Join(link, "entities.json")
			case "file symlink":
				p = filepath.Join(filepath.Dir(path), "link")
				if err := os.Symlink("entities.json", p); err != nil {
					t.Fatal(err)
				}
			case "directory":
				p = filepath.Join(filepath.Dir(path), "dir")
				if err := os.Mkdir(p, 0700); err != nil {
					t.Fatal(err)
				}
			case "public parent":
				dir := filepath.Join(filepath.Dir(path), "public")
				if err := os.Mkdir(dir, 0755); err != nil {
					t.Fatal(err)
				}
				p = filepath.Join(dir, "file")
			case "missing parent":
				p = filepath.Join(filepath.Dir(path), "missing", "file")
			}
			if err := c.SaveFile(p); err == nil {
				t.Fatal("unsafe path accepted")
			}
			if discarded, err := c.LoadFile(p, entityCacheTestScope(a), 1); err == nil || discarded || len(c.Get(a, 1)) != 1 {
				t.Fatal("path error discarded live cache", discarded, err)
			}
		})
	}
	missing := newEntityFileCache(t, 2, 10)
	if err := missing.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	if discarded, err := missing.LoadFile(path, entityCacheTestScope(a), 1); err != nil || discarded || len(missing.Get(a, 1)) != 0 {
		t.Fatal("missing snapshot not clean", discarded, err)
	}
}

func TestEntityCacheFileSaveBoundBeforeReplace(t *testing.T) {
	path := entityCacheTestPath(t)
	a := entityAssignment()
	c := newEntityFileCache(t, 2, 10)
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	a.SourceRevision = strings.Repeat("x", entityMaxBytes)
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.SaveFile(path); err == nil {
		t.Fatal("oversized snapshot accepted")
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(before, after) {
		t.Fatal("failed save damaged previous snapshot")
	}
}

func TestEntityCacheFileConcurrentOperations(t *testing.T) {
	path := entityCacheTestPath(t)
	a := entityAssignment()
	c := newEntityFileCache(t, 2, 10)
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	if err := c.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 3 {
		wg.Go(func() {
			for range 5 {
				if err := c.Put([]EntityAssignment{a}, 0); err != nil {
					t.Error(err)
				}
				if err := c.SaveFile(path); err != nil {
					t.Error(err)
				}
				if discarded, err := c.LoadFile(path, entityCacheTestScope(a), 1); err != nil || discarded {
					t.Error(discarded, err)
				}
				c.Get(a, 1)
			}
		})
	}
	wg.Wait()
}
