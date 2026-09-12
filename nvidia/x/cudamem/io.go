package cudamem

import (
	"errors"
	"io"
)

// Reader implements io.Reader and io.WriterTo on top of a device byte slice.
type Reader struct {
	slice  Slice[byte]
	offset int
}

// NewReader returns a new Reader reading from device slice s.
func NewReader(s Slice[byte]) *Reader {
	return &Reader{
		slice:  s,
		offset: 0,
	}
}

// Read reads up to len(p) bytes from the device slice into p.
func (r *Reader) Read(p []byte) (n int, err error) {
	if r.offset >= r.slice.len {
		return 0, io.EOF
	}
	remaining := r.slice.len - r.offset
	toRead := len(p)
	if toRead > remaining {
		toRead = remaining
	}

	sub := r.slice.Slice(r.offset, r.offset+toRead)
	if err := sub.CopyToHost(p[:toRead]); err != nil {
		return 0, err
	}

	r.offset += toRead
	return toRead, nil
}

// WriteTo copies device data through a host staging buffer to w.
func (r *Reader) WriteTo(w io.Writer) (n int64, err error) {
	bufSize := 64 * 1024 // 64KB staging chunk
	buf := make([]byte, bufSize)

	var totalWritten int64
	for {
		readN, readErr := r.Read(buf)
		if readN > 0 {
			writtenN, writeErr := w.Write(buf[:readN])
			totalWritten += int64(writtenN)
			if writeErr != nil {
				return totalWritten, writeErr
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return totalWritten, readErr
		}
	}
	return totalWritten, nil
}

// Writer implements io.Writer and io.ReaderFrom on top of a device byte slice.
type Writer struct {
	slice  Slice[byte]
	offset int
}

// NewWriter returns a new Writer writing into device slice s.
func NewWriter(s Slice[byte]) *Writer {
	return &Writer{
		slice:  s,
		offset: 0,
	}
}

// Write writes len(p) bytes from p into the device slice.
func (w *Writer) Write(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	remaining := w.slice.len - w.offset
	if remaining <= 0 {
		return 0, errors.New("cudamem: device slice full")
	}
	toWrite := len(p)
	if toWrite > remaining {
		toWrite = remaining
	}

	sub := w.slice.Slice(w.offset, w.offset+toWrite)
	if err := sub.CopyFromHost(p[:toWrite]); err != nil {
		return 0, err
	}

	w.offset += toWrite
	if toWrite < len(p) {
		return toWrite, errors.New("cudamem: short write to device slice")
	}
	return toWrite, nil
}

// ReadFrom reads data from the specified io.Reader directly into the device slice.
func (w *Writer) ReadFrom(r io.Reader) (n int64, err error) {
	bufSize := 64 * 1024 // 64KB staging chunk
	buf := make([]byte, bufSize)

	var totalRead int64
	for {
		readN, readErr := r.Read(buf)
		if readN > 0 {
			writtenN, writeErr := w.Write(buf[:readN])
			totalRead += int64(writtenN)
			if writeErr != nil {
				return totalRead, writeErr
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return totalRead, readErr
		}
	}
	return totalRead, nil
}
