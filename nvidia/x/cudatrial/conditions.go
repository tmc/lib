package cudatrial

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// A ConditionKind identifies one observed machine-state quantity, in bytes.
//
// Each kind names the pool it describes. A device readout and a host readout
// measure different things and must not be substituted for one another: a CUDA
// free-memory readout describes one device memory pool and is blind to host
// page-cache pressure, which is usually the term that actually moves between
// two runs of the same arm.
type ConditionKind string

const (
	// ConditionDeviceFreeBytes is free memory in one CUDA device memory pool,
	// as reported by a cudaMemGetInfo-style readout. It describes that pool
	// only; it is not a whole-machine pressure signal. The pool is named by
	// Conditions.Device.
	ConditionDeviceFreeBytes ConditionKind = "device_free_bytes"
	// ConditionDeviceTotalBytes is total memory in the same pool, recorded so a
	// free readout can be read as a fraction rather than an absolute.
	ConditionDeviceTotalBytes ConditionKind = "device_total_bytes"
	// ConditionHostAvailableBytes is host memory available for a new allocation
	// without swapping, from MemAvailable in /proc/meminfo. It is deliberately
	// not MemFree: the two differ by reclaimable page cache, and MemFree
	// understates what a workload can actually obtain.
	ConditionHostAvailableBytes ConditionKind = "host_available_bytes"
	// ConditionHostTotalBytes is host physical memory, from MemTotal.
	ConditionHostTotalBytes ConditionKind = "host_total_bytes"
	// ConditionHostPageCacheBytes is host page cache, the sum of Cached and
	// Buffers in /proc/meminfo. It includes tmpfs and shared pages, which are
	// not reclaimable, and so is an upper bound on cache the kernel can drop.
	// It is recorded separately because it is the term that moves most between
	// neighbouring arms while a device readout stays nearly flat.
	ConditionHostPageCacheBytes ConditionKind = "host_page_cache_bytes"
	// ConditionRetainedBytes is artifact bytes retained by the trial harness and
	// visible to the arm at the observation boundary: at Before, what earlier
	// arms carried in; at After, that plus what this arm retained itself. In an
	// interleaved trial this is one arm's footprint as its neighbour sees it.
	ConditionRetainedBytes ConditionKind = "retained_bytes"
)

// Conditions records the machine state an arm ran under at one boundary.
//
// Like Timings, Conditions distinguishes a measured zero from an unread value
// by key presence: a kind that could not be read is absent from Values, and
// Bytes reports Unmeasured for it. The zero Conditions has observed nothing;
// it does not assert that memory was free.
type Conditions struct {
	// Observed is when the readout was taken, or the zero time if never taken.
	Observed time.Time
	// Device names the device memory pool the device_* kinds describe
	// (for example "cuda:0"). It is empty when no device was read.
	Device string
	// Values holds the kinds that were read. A kind absent from Values was not
	// read; a kind present with value 0 was read and was zero.
	Values map[ConditionKind]int64
}

// Any reports whether any condition at all was read.
func (c Conditions) Any() bool { return len(c.Values) > 0 }

// Has reports whether kind was read.
func (c Conditions) Has(kind ConditionKind) bool {
	_, ok := c.Values[kind]
	return ok
}

// Bytes returns the value read for kind, or Unmeasured if it was not read.
func (c Conditions) Bytes(kind ConditionKind) int64 {
	if v, ok := c.Values[kind]; ok {
		return v
	}
	return Unmeasured
}

// Kinds returns the kinds read, sorted.
func (c Conditions) Kinds() []ConditionKind {
	out := make([]ConditionKind, 0, len(c.Values))
	for k := range c.Values {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Record records a read value for kind. A negative value is rejected: an
// unread kind is expressed by its absence, never by storing Unmeasured.
func (c *Conditions) Record(kind ConditionKind, bytes int64) error {
	if kind == "" {
		return errors.New("cudatrial: condition kind cannot be empty")
	}
	if bytes < 0 {
		return fmt.Errorf("cudatrial: condition %q cannot be negative: %d", kind, bytes)
	}
	if c.Values == nil {
		c.Values = make(map[ConditionKind]int64)
	}
	c.Values[kind] = bytes
	return nil
}

// RecordDevicePool records a device memory pool readout taken with a
// cudaMemGetInfo-style call. device names the pool, for example "cuda:0".
func (c *Conditions) RecordDevicePool(device string, free, total int64) error {
	if device == "" {
		return errors.New("cudatrial: device pool name cannot be empty")
	}
	if err := c.Record(ConditionDeviceFreeBytes, free); err != nil {
		return err
	}
	if err := c.Record(ConditionDeviceTotalBytes, total); err != nil {
		return err
	}
	c.Device = device
	return nil
}

// ObserveHost reads the host conditions available on this machine and stamps
// Observed. On hosts without /proc/meminfo, and for any line it cannot read,
// the corresponding kinds are left unread rather than recorded as zero.
// It never reads a device: a device pool has no meaning without a CUDA context,
// so device kinds are supplied by the caller's own Options.Observe.
func ObserveHost(ctx context.Context) Conditions {
	c := Conditions{Observed: time.Now()}
	observeHost(&c)
	return c
}

// observeMeminfoFile records the host kinds found in a /proc/meminfo-format
// file. A file that cannot be opened or read leaves every host kind unread.
func observeMeminfoFile(path string, c *Conditions) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return parseMeminfo(f, c)
}

// parseMeminfo reads Linux /proc/meminfo text and records the host kinds it
// finds. Lines it does not recognize, and values it cannot parse, are skipped,
// leaving the corresponding kinds unread.
func parseMeminfo(r io.Reader, c *Conditions) error {
	var cached, buffers int64
	haveCached, haveBuffers := false, false
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		name, value, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		bytes, ok := meminfoBytes(value)
		if !ok {
			continue
		}
		switch name {
		case "MemTotal":
			_ = c.Record(ConditionHostTotalBytes, bytes)
		case "MemAvailable":
			_ = c.Record(ConditionHostAvailableBytes, bytes)
		case "Cached":
			cached, haveCached = bytes, true
		case "Buffers":
			buffers, haveBuffers = bytes, true
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("cudatrial: read meminfo: %w", err)
	}
	if haveCached && haveBuffers {
		_ = c.Record(ConditionHostPageCacheBytes, cached+buffers)
	}
	return nil
}

// meminfoBytes converts a /proc/meminfo value field ("  19725932 kB") to bytes.
func meminfoBytes(field string) (int64, bool) {
	f := strings.Fields(field)
	if len(f) == 0 {
		return 0, false
	}
	n, err := strconv.ParseInt(f[0], 10, 64)
	if err != nil || n < 0 {
		return 0, false
	}
	switch {
	case len(f) == 1:
		return n, true
	case strings.EqualFold(f[1], "kB"):
		return n * 1024, true
	}
	return 0, false
}

// A ConditionDelta reports one condition that differed between two
// observations of the same arm.
type ConditionDelta struct {
	Kind   ConditionKind
	Before int64 // value at the earlier round
	After  int64 // value at the later round
	Delta  int64 // After - Before
}

func (d ConditionDelta) String() string {
	return fmt.Sprintf("%s %+d (%d -> %d)", d.Kind, d.Delta, d.Before, d.After)
}

// A Disagreement records two rounds of the same arm that produced different
// digests, together with the conditions those two rounds ran under.
//
// Its purpose is to keep a pressure artifact from being charged to the code:
// an arm whose neighbour retained tens of gibibytes did not run under the same
// conditions as its earlier self, and a digest difference between them is not
// by itself evidence of nondeterminism.
type Disagreement struct {
	Arm        string
	Rounds     [2]int    // earlier round, later round
	Digests    [2]string // digest of each
	Compared   []ConditionKind
	Changed    []ConditionDelta
	Unobserved []ConditionKind
}

// Confounded reports whether a change in conditions remains a live explanation
// for the disagreement, so that it cannot yet be charged to the code.
//
// It is false only when the two rounds were compared on at least one condition,
// every compared condition held within tolerance, and no condition was read in
// only one of the two rounds. A pair of rounds that recorded no conditions is
// always confounded: an unread condition is not an unchanged one.
func (d Disagreement) Confounded() bool {
	return len(d.Changed) > 0 || len(d.Unobserved) > 0 || len(d.Compared) == 0
}

func (d Disagreement) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "arm %q rounds %d/%d digests %q/%q", d.Arm, d.Rounds[0], d.Rounds[1], d.Digests[0], d.Digests[1])
	if !d.Confounded() {
		fmt.Fprintf(&sb, ": conditions held across %d observed", len(d.Compared))
		return sb.String()
	}
	sb.WriteString(": confounded")
	for _, c := range d.Changed {
		fmt.Fprintf(&sb, "; %s", c)
	}
	for _, k := range d.Unobserved {
		fmt.Fprintf(&sb, "; %s read in only one round", k)
	}
	if len(d.Compared) == 0 {
		sb.WriteString("; no conditions recorded")
	}
	return sb.String()
}

// SameArmDisagreements reports same-arm digest disagreements, each paired with
// the conditions the two rounds ran under. Each completed receipt is compared
// against the first completed receipt for its arm, matching SameArmExact.
//
// Conditions are compared at the Before boundary: what an arm started under is
// what its result was produced under. A kind counts as changed when the two
// rounds differ by more than tolerance bytes; a tolerance of 0 counts any
// difference. A kind read in only one of the two rounds is reported as
// unobserved rather than treated as unchanged.
func SameArmDisagreements(receipts []Receipt, tolerance int64) []Disagreement {
	if tolerance < 0 {
		tolerance = 0
	}
	base := make(map[string]Receipt)
	var out []Disagreement
	for _, r := range receipts {
		if !r.Complete {
			continue
		}
		first, ok := base[r.Name]
		if !ok {
			base[r.Name] = r
			continue
		}
		if first.Digest == r.Digest {
			continue
		}
		d := Disagreement{
			Arm:     r.Name,
			Rounds:  [2]int{first.Round, r.Round},
			Digests: [2]string{first.Digest, r.Digest},
		}
		seen := make(map[ConditionKind]bool)
		for _, k := range append(first.Before.Kinds(), r.Before.Kinds()...) {
			if seen[k] {
				continue
			}
			seen[k] = true
			if !first.Before.Has(k) || !r.Before.Has(k) {
				d.Unobserved = append(d.Unobserved, k)
				continue
			}
			d.Compared = append(d.Compared, k)
			before, after := first.Before.Bytes(k), r.Before.Bytes(k)
			if diff := after - before; diff > tolerance || -diff > tolerance {
				d.Changed = append(d.Changed, ConditionDelta{Kind: k, Before: before, After: after, Delta: diff})
			}
		}
		sort.Slice(d.Compared, func(i, j int) bool { return d.Compared[i] < d.Compared[j] })
		sort.Slice(d.Unobserved, func(i, j int) bool { return d.Unobserved[i] < d.Unobserved[j] })
		sort.Slice(d.Changed, func(i, j int) bool { return d.Changed[i].Kind < d.Changed[j].Kind })
		out = append(out, d)
	}
	return out
}
