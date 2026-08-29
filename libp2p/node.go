package libp2p

import (
	"context"
	"fmt"

	golibp2p "github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	ma "github.com/multiformats/go-multiaddr"
)

type Node struct {
	host host.Host
}

type Handler func(Stream)

func New(ctx context.Context, cfg Config, opts ...golibp2p.Option) (*Node, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	base, err := cfg.options()
	if err != nil {
		return nil, err
	}
	base = append(base, opts...)
	h, err := golibp2p.New(base...)
	if err != nil {
		return nil, fmt.Errorf("libp2p: new host: %w", err)
	}
	if err := ctx.Err(); err != nil {
		_ = h.Close()
		return nil, err
	}
	return &Node{host: h}, nil
}

func Wrap(h host.Host) (*Node, error) {
	if h == nil {
		return nil, ErrInvalidArgument
	}
	return &Node{host: h}, nil
}

func (n *Node) Host() host.Host {
	if n == nil {
		return nil
	}
	return n.host
}

func (n *Node) ID() peer.ID {
	if n == nil || n.host == nil {
		return ""
	}
	return n.host.ID()
}

func (n *Node) Addrs() []ma.Multiaddr {
	if n == nil || n.host == nil {
		return nil
	}
	return n.host.Addrs()
}

func (n *Node) AddrInfo() peer.AddrInfo {
	if n == nil || n.host == nil {
		return peer.AddrInfo{}
	}
	return peer.AddrInfo{ID: n.host.ID(), Addrs: n.host.Addrs()}
}

func (n *Node) Connect(ctx context.Context, info peer.AddrInfo) error {
	if n == nil || n.host == nil {
		return ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if info.ID == "" {
		return fmt.Errorf("libp2p: missing peer id: %w", ErrInvalidArgument)
	}
	if err := n.host.Connect(ctx, info); err != nil {
		return fmt.Errorf("libp2p: connect %s: %w", info.ID, err)
	}
	return nil
}

func (n *Node) Handle(id protocol.ID, h Handler) error {
	if n == nil || n.host == nil {
		return ErrClosed
	}
	if id == "" || h == nil {
		return ErrInvalidArgument
	}
	n.host.SetStreamHandler(id, func(s network.Stream) {
		h(Stream{stream: s})
	})
	return nil
}

func (n *Node) RemoveHandler(id protocol.ID) {
	if n == nil || n.host == nil {
		return
	}
	n.host.RemoveStreamHandler(id)
}

func (n *Node) OpenStream(ctx context.Context, id peer.ID, proto protocol.ID) (Stream, error) {
	if n == nil || n.host == nil {
		return Stream{}, ErrClosed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if id == "" || proto == "" {
		return Stream{}, ErrInvalidArgument
	}
	s, err := n.host.NewStream(ctx, id, proto)
	if err != nil {
		return Stream{}, fmt.Errorf("libp2p: open stream %s %s: %w", id, proto, err)
	}
	return Stream{stream: s}, nil
}

func (n *Node) Close() error {
	if n == nil || n.host == nil {
		return nil
	}
	h := n.host
	n.host = nil
	return h.Close()
}
