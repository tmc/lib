package nbd

import (
	"context"
	"io"
	"sync"
	"sync/atomic"
)

// Client is a client connection to one NBD export.
type Client struct {
	mu     sync.Mutex
	rw     io.ReadWriter
	size   uint64
	flags  uint16
	nextID atomic.Uint64
}

// Dial negotiates exportName over rw and returns a client for the export.
func Dial(ctx context.Context, rw io.ReadWriter, exportName string) (*Client, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if rw == nil {
		return nil, ErrInvalidArgument
	}
	if exportName == "" {
		exportName = "default"
	}
	size, flags, err := readHandshake(rw, exportName)
	if err != nil {
		return nil, err
	}
	c := &Client{rw: rw, size: size, flags: flags}
	c.nextID.Store(1)
	return c, nil
}

func (c *Client) Size() uint64 {
	if c == nil {
		return 0
	}
	return c.size
}

func (c *Client) ReadAt(p []byte, off int64) (int, error) {
	if c == nil || c.rw == nil {
		return 0, ErrClosed
	}
	if off < 0 {
		return 0, ErrInvalidArgument
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	done := 0
	for done < len(p) {
		n := len(p) - done
		if n > maxPayload {
			n = maxPayload
		}
		handle := c.nextID.Add(1)
		if err := writeRequest(c.rw, Request{Type: CommandRead, Handle: handle, Offset: uint64(off) + uint64(done), Length: uint32(n)}); err != nil {
			return done, err
		}
		if err := readReply(c.rw, handle, p[done:done+n]); err != nil {
			return done, err
		}
		done += n
	}
	return done, nil
}

func (c *Client) WriteAt(p []byte, off int64) (int, error) {
	if c == nil || c.rw == nil {
		return 0, ErrClosed
	}
	if off < 0 {
		return 0, ErrInvalidArgument
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	done := 0
	for done < len(p) {
		n := len(p) - done
		if n > maxPayload {
			n = maxPayload
		}
		handle := c.nextID.Add(1)
		if err := writeRequest(c.rw, Request{Type: CommandWrite, Handle: handle, Offset: uint64(off) + uint64(done), Length: uint32(n)}); err != nil {
			return done, err
		}
		if _, err := c.rw.Write(p[done : done+n]); err != nil {
			return done, err
		}
		if err := readReply(c.rw, handle, nil); err != nil {
			return done, err
		}
		done += n
	}
	return done, nil
}

func (c *Client) Sync() error {
	if c == nil || c.rw == nil {
		return ErrClosed
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	handle := c.nextID.Add(1)
	if err := writeRequest(c.rw, Request{Type: CommandFlush, Handle: handle}); err != nil {
		return err
	}
	return readReply(c.rw, handle, nil)
}

func (c *Client) Trim(off, length uint64) error {
	if c == nil || c.rw == nil {
		return ErrClosed
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	handle := c.nextID.Add(1)
	if err := writeRequest(c.rw, Request{Type: CommandTrim, Handle: handle, Offset: off, Length: uint32(length)}); err != nil {
		return err
	}
	return readReply(c.rw, handle, nil)
}

func (c *Client) Close() error {
	if c == nil || c.rw == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	handle := c.nextID.Add(1)
	err := writeRequest(c.rw, Request{Type: CommandDisc, Handle: handle})
	c.rw = nil
	return err
}
