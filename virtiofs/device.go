package virtiofs

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/fuse"
)

type Device struct {
	cfg       Config
	transport Transport
}

func NewDevice(cfg Config, transport Transport) (*Device, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if transport == nil {
		return nil, ErrInvalidArgument
	}
	return &Device{cfg: cfg, transport: transport}, nil
}

func (d *Device) Config() Config {
	if d == nil {
		return Config{}
	}
	return d.cfg
}

func (d *Device) Tag() string {
	if d == nil {
		return ""
	}
	return d.cfg.Tag
}

func (d *Device) RoundTrip(ctx context.Context, frame fuse.Frame) (fuse.OutHeader, []byte, error) {
	if d == nil || d.transport == nil {
		return fuse.OutHeader{}, nil, ErrClosed
	}
	var req bytes.Buffer
	if err := fuse.WriteRequest(&req, frame.Header, frame.Data); err != nil {
		return fuse.OutHeader{}, nil, fmt.Errorf("encode fuse request: %w", err)
	}
	resp, err := d.transport.RoundTrip(ctx, req.Bytes())
	if err != nil {
		return fuse.OutHeader{}, nil, err
	}
	h, data, err := fuse.ReadResponse(bytes.NewReader(resp))
	if err != nil {
		return h, nil, fmt.Errorf("decode fuse response: %w", err)
	}
	if h.Error != 0 {
		code := h.Error
		if code < 0 {
			code = -code
		}
		return h, data, ResponseError{Code: code}
	}
	return h, data, nil
}
