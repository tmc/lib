package virtiofs

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/fuse"
)

type Server struct {
	FS fuse.FileSystem
}

func NewServer(fs fuse.FileSystem) (*Server, error) {
	if fs == nil {
		return nil, ErrInvalidArgument
	}
	return &Server{FS: fs}, nil
}

func (s *Server) Handle(ctx context.Context, request []byte) ([]byte, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.FS == nil {
		return nil, ErrInvalidArgument
	}
	frame, err := fuse.ReadFrame(bytes.NewReader(request))
	if err != nil {
		return nil, fmt.Errorf("read fuse request: %w", err)
	}
	return fuse.HandleFrame(ctx, s.FS, frame)
}
