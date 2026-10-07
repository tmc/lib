package ncclinspector

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
)

// FilePosition is a resume token for a file epoch. It binds
// file identity, the complete-record prefix and cumulative epoch limits.
// A zero position is not a resume token. MarshalText preserves it across restarts.
type FilePosition struct {
	device, inode uint64
	epoch         string
	limits        Limits
	high          int64
	info          os.FileInfo
	offset        int64
	records       int
	digest        [32]byte
}

// Offset returns the committed complete-record offset.
func (p FilePosition) Offset() int64 { return p.offset }

// FileDecoder reads an ordinary append-only file. Replacement, observed size
// rollback and changes to the committed prefix are errors. It does not poll.
// The caller must serialize append/rotation with reads for a coherent snapshot;
// a file can change between checks. The zero value is not usable.
type FileDecoder struct {
	device, inode uint64
	epoch         string
	digest        [32]byte
	file          *os.File
	path          string
	info          os.FileInfo
	high          int64
	readOffset    int64
	decoder       *Decoder
}

// OpenFile opens an ordinary file epoch, optionally resuming a Position from
// that same file. Resume starts at the last complete record, rereading pending
// bytes. Limits remain cumulative from the start of the epoch.
func OpenFile(path string, limits Limits, resume *FilePosition) (*FileDecoder, error) {
	l, err := limits.checked()
	if err != nil {
		return nil, err
	}
	f, err := openOrdinary(path)
	if err != nil {
		return nil, fmt.Errorf("inspector open: %w", err)
	}
	fail := func(err error) (*FileDecoder, error) { return nil, errors.Join(err, f.Close()) }
	info, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	device, inode, err := fileIdentity(info)
	if err != nil {
		return fail(err)
	}
	var epoch [16]byte
	if _, err := rand.Read(epoch[:]); err != nil {
		return fail(err)
	}
	d := &FileDecoder{file: f, path: path, info: info, high: info.Size(), device: device, inode: inode, epoch: hex.EncodeToString(epoch[:])}
	d.decoder, err = NewDecoder(d, l)
	if err != nil {
		return fail(err)
	}
	if resume != nil {
		if resume.device != device || resume.inode != inode || resume.limits != l || resume.epoch == "" || resume.offset < 0 || resume.offset > int64(l.Bytes) || resume.records < 1 || resume.records > l.Records || info.Size() < resume.high {
			return fail(errors.New("inspector: invalid file resume"))
		}
		hash, err := d.prefix(resume.offset)
		if err != nil {
			return fail(err)
		}
		if hash != resume.digest {
			return fail(errors.New("inspector: changed file prefix"))
		}
		d.epoch = resume.epoch
		d.digest = resume.digest
		d.readOffset = resume.offset
		d.decoder.offset = resume.offset
		d.decoder.bytes = int(resume.offset)
		d.decoder.records = resume.records
	}
	return d, nil
}

func (d *FileDecoder) check() error {
	if d == nil || d.file == nil {
		return errors.New("inspector: closed file decoder")
	}
	info, err := d.file.Stat()
	if err != nil {
		return err
	}
	named, err := os.Lstat(d.path)
	if err != nil {
		return err
	}
	if !named.Mode().IsRegular() || !os.SameFile(info, named) || !os.SameFile(info, d.info) {
		return errors.New("inspector: file replaced")
	}
	if info.Size() < d.high {
		return errors.New("inspector: file truncated")
	}
	d.high = info.Size()
	return nil
}

func (d *FileDecoder) Read(b []byte) (int, error) {
	if err := d.check(); err != nil {
		return 0, err
	}
	n, err := d.file.ReadAt(b, d.readOffset)
	d.readOffset += int64(n)
	return n, err
}

func (d *FileDecoder) prefix(n int64) ([32]byte, error) {
	h := sha256.New()
	if _, err := io.CopyN(h, io.NewSectionReader(d.file, 0, n), n); err != nil {
		return [32]byte{}, err
	}
	var sum [32]byte
	copy(sum[:], h.Sum(nil))
	return sum, nil
}

// Next returns a complete record or temporary io.EOF, as Decoder.Next does.
func (d *FileDecoder) Next() (Record, error) {
	if err := d.check(); err != nil {
		return Record{}, err
	}
	if d.decoder.offset != 0 {
		h, err := d.prefix(d.decoder.offset)
		if err != nil {
			return Record{}, err
		}
		if h != d.digest {
			return Record{}, errors.New("inspector: changed file prefix")
		}
	}
	r, err := d.decoder.Next()
	if err == nil {
		d.digest, err = d.prefix(d.decoder.offset)
	}
	return r, err
}

// Position binds the successfully consumed prefix for a later OpenFile call.
// Pending bytes do not appear in the token.
func (d *FileDecoder) Position() (FilePosition, error) {
	if err := d.check(); err != nil {
		return FilePosition{}, err
	}
	if d.decoder.records == 0 {
		return FilePosition{}, errors.New("inspector: no committed record")
	}
	hash, err := d.prefix(d.decoder.Offset())
	if err != nil {
		return FilePosition{}, err
	}
	if err := d.check(); err != nil {
		return FilePosition{}, err
	}
	if hash != d.digest {
		return FilePosition{}, errors.New("inspector: changed file prefix")
	}
	return FilePosition{high: d.high, info: d.info, offset: d.decoder.Offset(), records: d.decoder.records, digest: hash, device: d.device, inode: d.inode, epoch: d.epoch, limits: d.decoder.limits}, nil
}

// Finish declares the current epoch complete without reading more input.
func (d *FileDecoder) Finish() error {
	position, err := d.Position()
	if err != nil {
		return err
	}
	if position.high != position.offset {
		return errors.New("inspector: unread or incomplete file input")
	}
	return d.decoder.Finish()
}

// Close closes the owned file. It is safe to call again after a successful close.
func (d *FileDecoder) Close() error {
	if d == nil || d.file == nil {
		return nil
	}
	err := d.file.Close()
	if err == nil {
		d.file = nil
	}
	return err
}
