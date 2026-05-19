package fuse

import "context"

type FileSystem interface {
	Lookup(ctx context.Context, parent NodeID, name string) (Entry, error)
	Getattr(ctx context.Context, node NodeID) (Attr, error)
	Setattr(ctx context.Context, node NodeID, in Setattr) (Attr, error)
	Open(ctx context.Context, node NodeID, flags uint32) (HandleID, error)
	Read(ctx context.Context, node NodeID, handle HandleID, off uint64, size uint32) ([]byte, error)
	Write(ctx context.Context, node NodeID, handle HandleID, off uint64, data []byte) (uint32, error)
	Readdir(ctx context.Context, node NodeID, handle HandleID, off uint64, size uint32) ([]Dirent, error)
	Flush(ctx context.Context, node NodeID, handle HandleID) error
	Fsync(ctx context.Context, node NodeID, handle HandleID) error
	Release(ctx context.Context, node NodeID, handle HandleID) error
}
