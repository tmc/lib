package ncclinspector

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
)

// cursorWire is deliberately independent of os.FileInfo and process state.
type cursorWire struct {
	Version int    `json:"version"`
	Device  uint64 `json:"device"`
	Inode   uint64 `json:"inode"`
	Epoch   string `json:"epoch"`
	Offset  int64  `json:"offset"`
	High    int64  `json:"high_water"`
	Bytes   int64  `json:"committed_bytes"`
	Records int    `json:"records"`
	Limits  Limits `json:"limits"`
	Prefix  string `json:"prefix_sha256"`
}

func (w cursorWire) valid() error {
	l, err := w.Limits.checked()
	if err != nil || l != w.Limits || w.Version != 1 || w.Inode == 0 || w.Offset <= 0 || w.Bytes != w.Offset || w.High < w.Offset || w.Offset > int64(l.Bytes) || w.Records <= 0 || w.Records > l.Records || int64(w.Records) > w.Offset {
		return errors.New("inspector: invalid resume cursor")
	}
	epoch, err := hex.DecodeString(w.Epoch)
	if err != nil || len(epoch) != 16 || hex.EncodeToString(epoch) != w.Epoch {
		return errors.New("inspector: invalid cursor epoch")
	}
	digest, err := hex.DecodeString(w.Prefix)
	if err != nil || len(digest) != 32 || hex.EncodeToString(digest) != w.Prefix {
		return errors.New("inspector: invalid cursor prefix")
	}
	return nil
}

// MarshalText encodes a bounded version-1 JSON resume cursor. The cursor binds
// device/inode, a decoder epoch identifier, committed offset/prefix and limits.
// Store it only after the corresponding records have been consumed durably.
// A cursor is not authentication against a caller that forges its contents.
func (p FilePosition) MarshalText() ([]byte, error) {
	w := cursorWire{1, p.device, p.inode, p.epoch, p.offset, p.high, p.offset, p.records, p.limits, hex.EncodeToString(p.digest[:])}
	if err := w.valid(); err != nil {
		return nil, err
	}
	return json.Marshal(w)
}

// UnmarshalText decodes a cursor without opening a file. Unknown versions,
// fields, duplicate keys and malformed counters are refused. OpenFile then
// validates the actual file identity, high-water and complete-prefix digest.
// The receiver is unchanged on failure. The format is bounded to 4096 bytes.
func (p *FilePosition) UnmarshalText(b []byte) error {
	if p == nil || len(b) == 0 || len(b) > 4096 {
		return errors.New("inspector: invalid resume cursor size")
	}
	if err := uniqueJSON(b); err != nil {
		return err
	}
	fields, err := requiredObject(b, "version", "device", "inode", "epoch", "offset", "high_water", "committed_bytes", "records", "limits", "prefix_sha256")
	if err != nil {
		return err
	}
	if _, err := requiredObject(fields["limits"], "Bytes", "Line", "Records"); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	var w cursorWire
	if err := d.Decode(&w); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return errors.New("inspector: trailing cursor data")
	}
	if err := w.valid(); err != nil {
		return err
	}
	digest, _ := hex.DecodeString(w.Prefix)
	next := FilePosition{device: w.Device, inode: w.Inode, epoch: w.Epoch, limits: w.Limits, offset: w.Offset, high: w.High, records: w.Records}
	copy(next.digest[:], digest)
	*p = next
	return nil
}
