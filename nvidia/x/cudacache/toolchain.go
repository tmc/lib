package cudacache

import (
	"errors"
	"fmt"
	"os"
	"strings"
)

// ErrUnhealableToolchainMismatch reports that an entry was rebuilt once
// because its recorded toolchain differed from the active one, and the
// rebuild recorded the same difference again, so rebuilding cannot converge.
var ErrUnhealableToolchainMismatch = errors.New("cudacache: unhealable toolchain mismatch")

// ToolchainMismatchError reports an entry whose recorded toolchain differs
// from the active one by the same pair of fingerprints as the previous
// mismatch recorded in Marker.
type ToolchainMismatchError struct {
	Key      string         // compiler cache key of the entry
	Marker   string         // path of the mismatch marker file
	Active   ToolProvenance // toolchain the lookup ran under
	Recorded ToolProvenance // toolchain the entry was produced by
}

func (e *ToolchainMismatchError) Error() string {
	return fmt.Sprintf("cudacache: unhealable toolchain mismatch for key %s: active toolchain %s, recorded toolchain %s (%s); remove %s after fixing the toolchain",
		e.Key, e.Active.Fingerprint(), e.Recorded.Fingerprint(), strings.Join(e.Active.Diff(e.Recorded), "; "), e.Marker)
}

// Unwrap returns ErrUnhealableToolchainMismatch.
func (e *ToolchainMismatchError) Unwrap() error { return ErrUnhealableToolchainMismatch }

// markerPair is the marker's record of a mismatch, in the format of
// cuda-oxide's heal marker.
func markerPair(active, recorded ToolProvenance) string {
	return fmt.Sprintf("active toolchain:\n%s\n\nrecorded toolchain:\n%s\n", active.Fingerprint(), recorded.Fingerprint())
}

// healToolchain decides what to do with an entry, sealed with nonce, that
// was recorded under a different toolchain than the active one. The first
// time a pair is seen it writes a marker naming the pair and the manifest,
// and reports heal, so the caller evicts and rebuilds. The same pair on a
// different manifest means a rebuild reproduced the mismatch, and it returns
// a *ToolchainMismatchError instead. The same pair on the same manifest is a
// lookup that raced the eviction of that manifest, and heals again.
func (c *Cache) healToolchain(key string, active, recorded ToolProvenance, nonce string) (heal bool, err error) {
	path := c.markerPath(key)
	pair := markerPair(active, recorded)
	have, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return false, fmt.Errorf("read toolchain marker: %w", err)
	}
	if rest, ok := strings.CutPrefix(string(have), pair+"\nevicted manifest:\n"); ok {
		if strings.TrimSuffix(rest, "\n") == nonce {
			return true, nil
		}
		return false, &ToolchainMismatchError{Key: key, Marker: path, Active: active, Recorded: recorded}
	}
	if err := c.atomicWrite(path, []byte(pair+"\nevicted manifest:\n"+nonce+"\n")); err != nil {
		return false, fmt.Errorf("write toolchain marker: %w", err)
	}
	return true, nil
}

// clearToolchainMarker removes the marker for key after a lookup found the
// recorded and active toolchains in agreement.
func (c *Cache) clearToolchainMarker(key string) error {
	if err := os.Remove(c.markerPath(key)); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("clear toolchain marker: %w", err)
	}
	return nil
}
