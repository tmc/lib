package ncclinspector

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"os"
)

// File epoch errors. FileDecoder wraps them, so test with errors.Is. Neither is
// a temporary io.EOF: the epoch is over and a new decoder must start from offset
// zero of the current file.
var (
	// ErrFileReplaced reports that the path no longer names the opened regular
	// file, for example after rename-and-create rotation.
	ErrFileReplaced = errors.New("inspector: file replaced")

	// ErrFileTruncated reports that the file is shorter than a size already
	// observed or than the committed record prefix.
	ErrFileTruncated = errors.New("inspector: file truncated")
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

// FileDecoder reads an ordinary append-only file. Replacement (ErrFileReplaced),
// observed size rollback (ErrFileTruncated) and changes to the committed prefix
// are errors. It does not poll.
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
	hashed        int64 // bytes written to prefix hashes, for tests
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
		if resume.device != device || resume.inode != inode {
			return fail(fmt.Errorf("inspector resume: %w", ErrFileReplaced))
		}
		if resume.limits != l || resume.epoch == "" || resume.offset < 0 || resume.offset > int64(l.Bytes) || resume.records < 1 || resume.records > l.Records {
			return fail(errors.New("inspector: invalid file resume"))
		}
		if info.Size() < resume.high {
			return fail(fmt.Errorf("inspector resume: %w", ErrFileTruncated))
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
		return fmt.Errorf("inspector stat: %w", err)
	}
	named, err := os.Lstat(d.path)
	if err != nil {
		return fmt.Errorf("inspector stat: %w", err)
	}
	if !named.Mode().IsRegular() || !os.SameFile(info, named) || !os.SameFile(info, d.info) {
		return ErrFileReplaced
	}
	if info.Size() < d.high {
		return ErrFileTruncated
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

// prefixHash returns a SHA-256 state holding the first n bytes of the file.
func (d *FileDecoder) prefixHash(n int64) (hash.Hash, error) {
	h := sha256.New()
	written, err := io.CopyN(h, io.NewSectionReader(d.file, 0, n), n)
	d.hashed += written
	if err != nil {
		if err == io.EOF {
			// The size check passed, so the file shrank since: not temporary EOF.
			return nil, ErrFileTruncated
		}
		return nil, fmt.Errorf("inspector prefix: %w", err)
	}
	return h, nil
}

func (d *FileDecoder) prefix(n int64) ([32]byte, error) {
	h, err := d.prefixHash(n)
	if err != nil {
		return [32]byte{}, err
	}
	return sum(h), nil
}

func sum(h hash.Hash) [32]byte {
	var s [32]byte
	copy(s[:], h.Sum(nil))
	return s
}

// Next returns a complete record or temporary io.EOF, as Decoder.Next does.
// Each call first rehashes the committed prefix from the file and refuses a
// change. No hash state is kept between calls: the hash.Hash used for that
// verification is extended in place with the returned Raw (including its
// newline), and only its digest is stored. The digest therefore binds the
// bytes parsed, not a second read of the file, and the next Next, Position,
// Finish or resume compares it with the file. A change within one call is
// not guaranteed to be detected; callers must serialize appends and rotation.
func (d *FileDecoder) Next() (Record, error) {
	if err := d.check(); err != nil {
		return Record{}, err
	}
	start := d.decoder.offset
	h := sha256.New()
	if start != 0 {
		var err error
		if h, err = d.prefixHash(start); err != nil {
			return Record{}, err
		}
		if sum(h) != d.digest {
			return Record{}, errors.New("inspector: changed file prefix")
		}
	}
	r, err := d.decoder.Next()
	if err != nil {
		return r, err
	}
	if int64(len(r.Raw)) != d.decoder.offset-start {
		return Record{}, errors.New("inspector: record bytes differ from committed offset")
	}
	h.Write(r.Raw)
	d.hashed += int64(len(r.Raw))
	d.digest = sum(h)
	return r, nil
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
