package cudacapture

import "errors"

// ReannouncementLimits bounds a snapshot of already observed module metadata.
// Both limits must be positive. Bytes charges 256 bytes per entry, 16 per
// Sources element, and the bytes of all strings. It bounds retained metadata,
// not code artifact contents (CodeFiles contains filenames, not file bodies).
type ReannouncementLimits struct {
	MaxEntries uint
	MaxBytes   uint64
}

// Reannouncement is a detached snapshot of known module generations, ordered
// by base address then generation. Inactive generations are retained so an
// unloaded image cannot lend its code identity to a reload. Empty code roles
// remain unavailable; filenames do not prove symbols or line information exist.
// This is metadata only, not past samples or discovery of unobserved modules.
type Reannouncement struct {
	Limits  ReannouncementLimits
	Entries []ModuleEntry
	Bytes   uint64
	// OmittedCountLimit and OmittedByteLimit count entries refused by the respective
	// limit. Count admission is checked first; each omitted entry counts once.
	OmittedCountLimit uint64
	OmittedByteLimit  uint64
}

// Reannounce returns a bounded snapshot of metadata observed by the tracker.
// Limits are checked before copying each entry or its Sources slice. The
// tracker lock covers the snapshot; later loads and unloads appear in a later
// snapshot. No artifact files, driver APIs or execution are accessed.
func (mt *ModuleTracker) Reannounce(limits ReannouncementLimits) (Reannouncement, error) {
	r := Reannouncement{Limits: limits}
	if limits.MaxEntries == 0 || limits.MaxBytes == 0 {
		return r, errors.New("cudacapture: invalid reannouncement limits")
	}
	mt.mu.Lock()
	defer mt.mu.Unlock()
	// Select the next address without allocating an unbounded key inventory.
	var last uint64
	for {
		var next uint64
		for addr := range mt.generations {
			if addr > last && (next == 0 || addr < next) {
				next = addr
			}
		}
		if next == 0 {
			break
		}
		for _, e := range mt.generations[next] {
			if uint64(len(r.Entries)) >= uint64(limits.MaxEntries) {
				r.OmittedCountLimit++
				continue
			}
			n, ok := entryBytes(e)
			if !ok || n > limits.MaxBytes-r.Bytes {
				r.OmittedByteLimit++
				continue
			}
			e.Code = cloneCode(e.Code)
			r.Entries = append(r.Entries, e)
			r.Bytes += n
		}
		last = next
	}
	return r, nil
}

// Clone returns a detached copy, including each CodeFiles.Sources slice.
func (r Reannouncement) Clone() Reannouncement {
	r.Entries = append([]ModuleEntry(nil), r.Entries...)
	for i := range r.Entries {
		r.Entries[i].Code = cloneCode(r.Entries[i].Code)
	}
	return r
}

// Validate checks snapshot bounds, charges and generation ordering. It does not
// authenticate filenames or establish symbol/line coverage.
func (r Reannouncement) Validate() error {
	if r.Limits.MaxEntries == 0 || r.Limits.MaxBytes == 0 || uint64(len(r.Entries)) > uint64(r.Limits.MaxEntries) {
		return errors.New("cudacapture: invalid reannouncement limits")
	}
	var total uint64
	var prev ModuleEntry
	for i, e := range r.Entries {
		if e.BaseAddress == 0 || e.Generation == 0 || e.ImageDigest == "" {
			return errors.New("cudacapture: incomplete module identity")
		}
		if i > 0 && (e.BaseAddress < prev.BaseAddress || (e.BaseAddress == prev.BaseAddress && (e.Generation <= prev.Generation || prev.Active))) {
			return errors.New("cudacapture: invalid module generation order")
		}
		n, ok := entryBytes(e)
		if !ok || n > r.Limits.MaxBytes-total {
			return errors.New("cudacapture: reannouncement exceeds byte limit")
		}
		total += n
		prev = e
	}
	if total != r.Bytes {
		return errors.New("cudacapture: reannouncement byte charge mismatch")
	}
	return nil
}

func entryBytes(e ModuleEntry) (uint64, bool) {
	n := uint64(256)
	add := func(v uint64) bool {
		if v > ^uint64(0)-n {
			return false
		}
		n += v
		return true
	}
	for _, s := range []string{e.ID, e.ImageDigest, e.Code.Source, e.Code.PTX, e.Code.CUBIN, e.Code.SASS, e.Code.Unit, e.Code.TileIR, e.Code.CFG, e.Code.BBCFG} {
		if !add(uint64(len(s))) {
			return 0, false
		}
	}
	for _, s := range e.Code.Sources {
		if !add(16) || !add(uint64(len(s))) {
			return 0, false
		}
	}
	return n, true
}
