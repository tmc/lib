package fuse

import (
	"context"
	"sort"
	"sync"
)

const maxMemoryFileSize = 1 << 30

type MemoryFS struct {
	mu     sync.Mutex
	next   NodeID
	nodes  map[NodeID]*memNode
	parent map[NodeID]NodeID
}

type memNode struct {
	id       NodeID
	name     string
	mode     uint32
	data     []byte
	children map[string]NodeID
}

func NewMemoryFS() *MemoryFS {
	root := &memNode{id: RootID, name: "/", mode: 0o040755, children: map[string]NodeID{}}
	return &MemoryFS{next: RootID + 1, nodes: map[NodeID]*memNode{RootID: root}, parent: map[NodeID]NodeID{}}
}

func (fs *MemoryFS) AddFile(parent NodeID, name string, data []byte) (NodeID, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	p := fs.nodes[parent]
	if p == nil || p.children == nil || name == "" || len(data) > maxMemoryFileSize {
		return 0, ErrInvalidArgument
	}
	id := fs.next
	fs.next++
	p.children[name] = id
	fs.parent[id] = parent
	fs.nodes[id] = &memNode{id: id, name: name, mode: 0o100644, data: append([]byte(nil), data...)}
	return id, nil
}

func (fs *MemoryFS) Lookup(ctx context.Context, parent NodeID, name string) (Entry, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	p := fs.nodes[parent]
	if p == nil || p.children == nil {
		return Entry{}, ErrNotExist
	}
	id, ok := p.children[name]
	if !ok {
		return Entry{}, ErrNotExist
	}
	return Entry{NodeID: id, Attr: fs.attrLocked(fs.nodes[id])}, nil
}

func (fs *MemoryFS) Getattr(ctx context.Context, node NodeID) (Attr, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	n := fs.nodes[node]
	if n == nil {
		return Attr{}, ErrNotExist
	}
	return fs.attrLocked(n), nil
}

func (fs *MemoryFS) Setattr(ctx context.Context, node NodeID, in Setattr) (Attr, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	n := fs.nodes[node]
	if n == nil {
		return Attr{}, ErrNotExist
	}
	if in.Valid&1 != 0 {
		n.mode = in.Mode
	}
	if in.Valid&8 != 0 {
		if in.Size > maxMemoryFileSize {
			return Attr{}, ErrInvalidArgument
		}
		if uint64(len(n.data)) < in.Size {
			n.data = append(n.data, make([]byte, int(in.Size)-len(n.data))...)
		} else {
			n.data = n.data[:in.Size]
		}
	}
	return fs.attrLocked(n), nil
}

func (fs *MemoryFS) Open(ctx context.Context, node NodeID, flags uint32) (HandleID, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if fs.nodes[node] == nil {
		return 0, ErrNotExist
	}
	return HandleID(node), nil
}

func (fs *MemoryFS) Read(ctx context.Context, node NodeID, handle HandleID, off uint64, size uint32) ([]byte, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	n := fs.nodes[node]
	if n == nil {
		return nil, ErrNotExist
	}
	if off > uint64(len(n.data)) {
		return nil, ErrInvalidArgument
	}
	end := off + uint64(size)
	if end > uint64(len(n.data)) {
		end = uint64(len(n.data))
	}
	return append([]byte(nil), n.data[off:end]...), nil
}

func (fs *MemoryFS) Write(ctx context.Context, node NodeID, handle HandleID, off uint64, data []byte) (uint32, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	n := fs.nodes[node]
	if n == nil || n.children != nil {
		return 0, ErrInvalidArgument
	}
	if off > maxMemoryFileSize || uint64(len(data)) > maxMemoryFileSize-off {
		return 0, ErrInvalidArgument
	}
	end := off + uint64(len(data))
	if end > uint64(len(n.data)) {
		n.data = append(n.data, make([]byte, int(end)-len(n.data))...)
	}
	copy(n.data[off:end], data)
	return uint32(len(data)), nil
}

func (fs *MemoryFS) Readdir(ctx context.Context, node NodeID, handle HandleID, off uint64, size uint32) ([]Dirent, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	n := fs.nodes[node]
	if n == nil || n.children == nil {
		return nil, ErrInvalidArgument
	}
	names := make([]string, 0, len(n.children))
	for name := range n.children {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]Dirent, 0, len(names))
	for i, name := range names {
		id := n.children[name]
		out = append(out, Dirent{Ino: uint64(id), Off: uint64(i + 1), Type: 8, Name: name})
	}
	return out, nil
}

func (fs *MemoryFS) Flush(ctx context.Context, node NodeID, handle HandleID) error   { return nil }
func (fs *MemoryFS) Fsync(ctx context.Context, node NodeID, handle HandleID) error   { return nil }
func (fs *MemoryFS) Release(ctx context.Context, node NodeID, handle HandleID) error { return nil }

func (fs *MemoryFS) attrLocked(n *memNode) Attr {
	size := uint64(len(n.data))
	if n.children != nil {
		size = 0
	}
	return Attr{Ino: uint64(n.id), Size: size, Mode: n.mode, Nlink: 1, Blksize: 4096}
}
