package libp2p

import (
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
)

type Stream struct {
	stream network.Stream
}

func WrapStream(s network.Stream) (Stream, error) {
	if s == nil {
		return Stream{}, ErrInvalidArgument
	}
	return Stream{stream: s}, nil
}

func (s Stream) Raw() network.Stream {
	return s.stream
}

func (s Stream) Read(p []byte) (int, error) {
	if s.stream == nil {
		return 0, ErrClosed
	}
	return s.stream.Read(p)
}

func (s Stream) Write(p []byte) (int, error) {
	if s.stream == nil {
		return 0, ErrClosed
	}
	return s.stream.Write(p)
}

func (s Stream) Close() error {
	if s.stream == nil {
		return nil
	}
	return s.stream.Close()
}

func (s Stream) Reset() error {
	if s.stream == nil {
		return nil
	}
	return s.stream.Reset()
}

func (s Stream) Protocol() protocol.ID {
	if s.stream == nil {
		return ""
	}
	return s.stream.Protocol()
}

func (s Stream) RemotePeer() peer.ID {
	if s.stream == nil {
		return ""
	}
	return s.stream.Conn().RemotePeer()
}
