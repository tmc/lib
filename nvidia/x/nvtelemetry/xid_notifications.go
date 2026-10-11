package nvtelemetry

import (
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/xidcatalog"
	"github.com/tmc/lib/nvidia/x/xidlog"
)

// XIDNotification is a supplied kernel fact, not a fault diagnosis or action.
// DeviceUUID is known only through the caller's explicit PCI identity mapping.
// Kernel timestamps retain microseconds in the supplied boot's monotonic domain;
// they are neither sensor timestamps nor collector wall timestamps.
// Raw lines, process names, and driver error text are not retained.
type XIDNotification struct {
	Kind                        xidlog.Kind
	Source                      xidlog.Source
	Product                     xidcatalog.Architecture
	Driver                      string
	PCI, DeviceUUID             string
	DeviceIdentityKnown         bool
	Code                        uint64
	HasKernelTimestamp          bool
	KernelMonotonicMicroseconds uint64
	KernelSequence              uint64
	Catalog                     xidcatalog.Entry
	Applicability               xidcatalog.Applicability
}

// XIDNotifications is a bounded supplied-input result. InputEnded means EOF,
// never complete boot coverage. LossUnknown is always true: supplied log input
// cannot prove records were not lost before collection or filtering.
type XIDNotifications struct {
	Records     []XIDNotification
	InputEnded  bool
	LossUnknown bool
}

// ReadXIDNotifications reads caller-owned text or record-framed kmsg input.
// PCI identities are explicit caller assertions, never inferred from proximity.
// Duplicate/reversed kmsg sequence numbers and reversed timestamps are refused.
// Fragments, malformed
// records and gaps retain their class without exposing their raw payload.
func ReadXIDNotifications(input io.Reader, format xidlog.Format, options xidlog.Options, pciIdentities map[string]string) (XIDNotifications, error) {
	out := XIDNotifications{LossUnknown: true}
	if len(pciIdentities) > 4096 {
		return out, fmt.Errorf("xid notifications: identity limit")
	}
	for pci, uuid := range pciIdentities {
		if pci == "" || uuid == "" || !exportText(pci) || !exportText(uuid) {
			return out, fmt.Errorf("xid notifications: invalid identity")
		}
	}
	reader, err := xidlog.NewReader(input, format, options)
	if err != nil {
		return out, err
	}
	var last, lastTime uint64
	haveSequence := false
	for {
		r, err := reader.Next()
		if err == io.EOF {
			out.InputEnded = true
			return out, nil
		}
		if err != nil {
			return out, err
		}
		if r.Header != nil {
			if haveSequence && (r.Header.Sequence <= last || r.Header.MonotonicMicroseconds < lastTime) {
				return out, fmt.Errorf("xid notifications: duplicate sequence or reversed kernel clock")
			}
			last = r.Header.Sequence
			lastTime = r.Header.MonotonicMicroseconds
			haveSequence = true
		}
		if r.Kind == xidlog.Other {
			continue
		}
		n := XIDNotification{Kind: r.Kind, Source: r.Source, Product: r.Product, Driver: r.Driver, PCI: r.PCI, Code: r.Code, Catalog: r.Catalog, Applicability: r.Applicability}
		if uuid, ok := pciIdentities[r.PCI]; ok {
			n.DeviceUUID = uuid
			n.DeviceIdentityKnown = true
		}
		if r.UUID != "" && n.DeviceIdentityKnown && r.UUID != n.DeviceUUID {
			return out, fmt.Errorf("xid notifications: contradictory GPU identity")
		}
		if r.Header != nil {
			n.HasKernelTimestamp = true
			n.KernelMonotonicMicroseconds = r.Header.MonotonicMicroseconds
			n.KernelSequence = r.Header.Sequence
		}
		out.Records = append(out.Records, n)
	}
}
