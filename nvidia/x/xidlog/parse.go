package xidlog

import (
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/tmc/lib/nvidia/x/xidcatalog"
)

// Kind identifies a fact, uninterpreted input, malformed record or loss notice.
type Kind uint8

const (
	Other Kind = iota
	XID
	GPU
	Fragment
	Malformed
	Gap
)

// Source is caller-supplied provenance. PreviousBoot labels the supplied input;
// it does not prove completeness or that events belong to the claimed boot.
type Source struct {
	BootID, InputID string
	PreviousBoot    bool
}

// Options supplies provenance and product-specific catalog interpretation.
// Driver is retained verbatim; parsing does not infer a driver from log text.
// Zero bounds use finite defaults. Larger limits than the defaults are refused.
type Options struct {
	Source                               Source
	Product                              xidcatalog.Architecture
	Driver                               string
	MaxRecordBytes, MaxBytes, MaxRecords int
}

const (
	MaxRecordBytes = 64 << 10
	MaxBytes       = 16 << 20
	MaxRecords     = 4096
)

func options(o Options) (Options, error) {
	if o.Source.BootID == "" || o.Source.InputID == "" || o.Product == "" || o.Driver == "" {
		return o, errors.New("xidlog: explicit boot, input, product and driver required")
	}
	for _, s := range []string{o.Source.BootID, o.Source.InputID, string(o.Product), o.Driver} {
		if len(s) > 256 || !utf8.ValidString(s) || strings.ContainsAny(s, "\x00\r\n") {
			return o, errors.New("xidlog: invalid provenance")
		}
	}
	for _, v := range []*int{&o.MaxRecordBytes, &o.MaxBytes, &o.MaxRecords} {
		if *v < 0 {
			return o, errors.New("xidlog: negative limit")
		}
	}
	if o.MaxRecordBytes == 0 {
		o.MaxRecordBytes = MaxRecordBytes
	}
	if o.MaxBytes == 0 {
		o.MaxBytes = MaxBytes
	}
	if o.MaxRecords == 0 {
		o.MaxRecords = MaxRecords
	}
	if o.MaxRecordBytes > MaxRecordBytes || o.MaxBytes > MaxBytes || o.MaxRecords > MaxRecords {
		return o, errors.New("xidlog: limit exceeds supported bound")
	}
	return o, nil
}

// Kmsg preserves the Linux record header. Extension fields and continuation
// metadata are raw observations, not invented timestamps or device identities.
type Kmsg struct {
	Priority, Sequence, MonotonicMicroseconds uint64
	Flags                                     string
	Extensions, Metadata                      []string
}

// Record retains raw input and caller provenance even when input is unknown.
// PID and Recovery are nil when absent or unsupported. Catalog is a raw entry
// plus separate applicability, not a health or recovery policy.
type Record struct {
	Kind                  Kind
	Source                Source
	Product               xidcatalog.Architecture
	Driver                string
	Raw, Message, Problem string
	Partial               bool
	Header                *Kmsg
	PCI, UUID             string
	Code                  uint64
	PID                   *uint64
	ProcessName           string
	Catalog               xidcatalog.Entry
	Applicability         xidcatalog.Applicability
	Recovery              *xidcatalog.RecoveryTransition
}

func record(raw string, o Options) Record {
	return Record{Raw: raw, Source: o.Source, Product: o.Product, Driver: o.Driver}
}

var pciRE = regexp.MustCompile(`^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}(?:\.[0-7])?$`)
var xidRE = regexp.MustCompile(`^NVRM: Xid \(PCI:([^()]+)\): ([0-9]+), (.*)$`)
var gpuRE = regexp.MustCompile(`^NVRM: GPU at PCI:([^ ]+) *: (GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$`)

func validPCI(s string) bool {
	if !pciRE.MatchString(s) {
		return false
	}
	parts := strings.Split(s, ":")
	d := strings.Split(parts[len(parts)-1], ".")[0]
	n, err := strconv.ParseUint(d, 16, 8)
	return err == nil && n <= 31
}
func malformed(r Record, problem string) Record { r.Kind = Malformed; r.Problem = problem; return r }
func parseMessage(r Record) Record {
	s := r.Message
	i := strings.Index(s, "NVRM: ")
	if i < 0 {
		return r
	}
	s = s[i:]
	if strings.Count(s, "NVRM: ") != 1 {
		return malformed(r, "multiple NVRM prefixes")
	}
	if m := gpuRE.FindStringSubmatch(s); m != nil {
		if !validPCI(m[1]) {
			return malformed(r, "invalid PCI token")
		}
		r.Kind = GPU
		r.PCI = m[1]
		r.UUID = m[2]
		return r
	}
	m := xidRE.FindStringSubmatch(s)
	if m == nil {
		if strings.HasPrefix(s, "NVRM: Xid") || strings.HasPrefix(s, "NVRM: GPU at PCI:") {
			return malformed(r, "unsupported or incomplete NVRM record")
		}
		return r
	}
	if !validPCI(m[1]) {
		return malformed(r, "invalid PCI token")
	}
	code, err := strconv.ParseUint(m[2], 10, 64)
	if err != nil || code == 0 {
		return malformed(r, "invalid XID code")
	}
	r.Kind = XID
	r.PCI = m[1]
	r.Code = code
	r.Catalog, r.Applicability = xidcatalog.LookupProduct(code, r.Product)
	rest := m[3]
	if strings.HasPrefix(rest, "pid=") {
		end := strings.Index(rest, ", ")
		if end < 0 {
			return malformed(r, "incomplete process PID")
		}
		pid, err := strconv.ParseUint(rest[4:end], 10, 64)
		if err != nil {
			return malformed(r, "invalid process PID")
		}
		r.PID = &pid
		rest = rest[end+2:]
	}
	if strings.HasPrefix(rest, "name=") {
		end := strings.Index(rest, ", ")
		if end < 0 {
			return malformed(r, "incomplete process name")
		}
		name := rest[5:end]
		if name == "" || len(name) > 256 {
			return malformed(r, "invalid process name")
		}
		r.ProcessName = name
		rest = rest[end+2:]
	}
	if code == 154 {
		v, err := xidcatalog.ParseRecoveryTransition("NVRM: Xid (PCI:" + r.PCI + "): 154, " + rest)
		if err != nil {
			r.Problem = err.Error()
		} else {
			r.Recovery = &v
		}
	}
	return r
}

// ParseText parses one supplied complete text line. Prefix text remains raw;
// dmesg or journal prefix timestamps are not interpreted as wall-clock time.
func ParseText(raw string, o Options) (Record, error) {
	o, err := options(o)
	if err != nil {
		return Record{}, err
	}
	r := record(raw, o)
	if len(raw) > o.MaxRecordBytes {
		return r, errors.New("xidlog: record exceeds byte limit")
	}
	s := strings.TrimSuffix(strings.TrimSuffix(raw, "\n"), "\r")
	r.Message = s
	if !utf8.ValidString(s) || strings.ContainsAny(s, "\x00\r\n") {
		return malformed(r, "invalid text encoding or line boundary"), nil
	}
	return parseMessage(r), nil
}

// ParseKmsg parses exactly one Linux kmsg record including its metadata lines.
// Unknown header extension fields and flags are retained. A c flag is a fragment
// and is never joined with another event or treated as a complete XID.
func ParseKmsg(raw string, o Options) (Record, error) {
	o, err := options(o)
	if err != nil {
		return Record{}, err
	}
	r := record(raw, o)
	if len(raw) > o.MaxRecordBytes {
		return r, errors.New("xidlog: record exceeds byte limit")
	}
	if !strings.HasSuffix(raw, "\n") {
		r.Partial = true
		return malformed(r, "unterminated kmsg record"), nil
	}
	for _, c := range raw {
		if (c < 0x20 && c != '\n') || c == 0x7f {
			return malformed(r, "unescaped kmsg control character"), nil
		}
	}
	lines := strings.Split(strings.TrimSuffix(raw, "\n"), "\n")
	semi := strings.IndexByte(lines[0], ';')
	if semi < 0 {
		return malformed(r, "missing kmsg separator"), nil
	}
	fields := strings.Split(lines[0][:semi], ",")
	if len(fields) < 4 {
		return malformed(r, "incomplete kmsg header"), nil
	}
	h := &Kmsg{Flags: fields[3], Extensions: append([]string(nil), fields[4:]...)}
	values := []*uint64{&h.Priority, &h.Sequence, &h.MonotonicMicroseconds}
	for i, v := range values {
		n, err := strconv.ParseUint(fields[i], 10, 64)
		if err != nil {
			return malformed(r, "invalid kmsg number"), nil
		}
		*v = n
	}
	if h.Flags == "" || !utf8.ValidString(raw) || strings.ContainsRune(raw, 0) {
		return malformed(r, "invalid kmsg encoding or flags"), nil
	}
	for _, line := range lines[1:] {
		if !strings.HasPrefix(line, " ") {
			return malformed(r, "invalid kmsg metadata boundary"), nil
		}
		h.Metadata = append(h.Metadata, line)
	}
	r.Header = h
	r.Message = lines[0][semi+1:]
	if strings.Contains(h.Flags, "c") {
		r.Kind = Fragment
		r.Partial = true
		return r, nil
	}
	return parseMessage(r), nil
}

// String returns a short factual label, without health classification.
func (k Kind) String() string {
	switch k {
	case Other:
		return "other"
	case XID:
		return "xid"
	case GPU:
		return "gpu"
	case Fragment:
		return "fragment"
	case Malformed:
		return "malformed"
	case Gap:
		return "gap"
	}
	return fmt.Sprintf("Kind(%d)", k)
}
