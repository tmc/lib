package nbd

import (
	"context"
	"errors"
	"io"
)

// Export describes one NBD export.
type Export struct {
	Name     string
	Device   Device
	ReadOnly bool
}

// Serve exports exp over rw until the client disconnects, ctx is canceled, or
// a protocol error occurs.
func Serve(ctx context.Context, rw io.ReadWriter, exp Export) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if rw == nil || exp.Device == nil {
		return ErrInvalidArgument
	}
	if closer, ok := rw.(io.Closer); ok {
		done := make(chan struct{})
		defer close(done)
		go func() {
			select {
			case <-ctx.Done():
				_ = closer.Close()
			case <-done:
			}
		}()
	}
	name := exp.Name
	if name == "" {
		name = "default"
	}
	flags := transmissionFlagHasFlags | transmissionFlagSendFlush
	if exp.ReadOnly {
		flags |= transmissionFlagReadOnly
	}
	if _, ok := exp.Device.(Trimmer); ok {
		flags |= transmissionFlagSendTrim
	}
	if err := writeHandshake(rw, name, exp.Device.Size(), flags); err != nil {
		return err
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		req, err := readRequest(rw)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		switch req.Type {
		case CommandRead:
			if errno := validateRequest(exp.Device.Size(), req); errno != 0 {
				_ = writeReply(rw, req.Handle, errno, nil)
				continue
			}
			data := make([]byte, req.Length)
			_, err := exp.Device.ReadAt(data, int64(req.Offset))
			if err != nil && !errors.Is(err, io.EOF) {
				_ = writeReply(rw, req.Handle, errEIO, nil)
				continue
			}
			if err := writeReply(rw, req.Handle, 0, data); err != nil {
				return err
			}
		case CommandWrite:
			if errno := validateRequest(exp.Device.Size(), req); errno != 0 {
				_ = writeReply(rw, req.Handle, errno, nil)
				continue
			}
			data := make([]byte, req.Length)
			if _, err := io.ReadFull(rw, data); err != nil {
				return err
			}
			if exp.ReadOnly {
				_ = writeReply(rw, req.Handle, errEPERM, nil)
				continue
			}
			if _, err := exp.Device.WriteAt(data, int64(req.Offset)); err != nil {
				_ = writeReply(rw, req.Handle, errEIO, nil)
				continue
			}
			if err := writeReply(rw, req.Handle, 0, nil); err != nil {
				return err
			}
		case CommandFlush:
			if s, ok := exp.Device.(Syncer); ok {
				if err := s.Sync(); err != nil {
					_ = writeReply(rw, req.Handle, errEIO, nil)
					continue
				}
			}
			if err := writeReply(rw, req.Handle, 0, nil); err != nil {
				return err
			}
		case CommandTrim:
			t, ok := exp.Device.(Trimmer)
			if !ok || exp.ReadOnly {
				_ = writeReply(rw, req.Handle, errEPERM, nil)
				continue
			}
			if errno := validateRequest(exp.Device.Size(), req); errno != 0 {
				_ = writeReply(rw, req.Handle, errno, nil)
				continue
			}
			if err := t.Trim(req.Offset, uint64(req.Length)); err != nil {
				_ = writeReply(rw, req.Handle, errEIO, nil)
				continue
			}
			if err := writeReply(rw, req.Handle, 0, nil); err != nil {
				return err
			}
		case CommandDisc:
			return nil
		default:
			_ = writeReply(rw, req.Handle, errEINVAL, nil)
		}
	}
}

func validateRequest(size uint64, req Request) uint32 {
	if req.Length > maxPayload || req.Offset > maxInt64 {
		return errEINVAL
	}
	length := uint64(req.Length)
	if req.Offset > size || length > size-req.Offset {
		return errENOSPC
	}
	return 0
}
