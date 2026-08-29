# Package Spec: libp2p

## Summary

Package `libp2p` provides a generated, small Go facade over
`github.com/libp2p/go-libp2p`. It is not a C binding and it is not a
symbol-by-symbol copy of go-libp2p.

The first slice covers:

- host construction from a small `Config`
- owned `Node` lifecycle
- host identity and listen addresses
- peer connection using `peer.AddrInfo`
- stream handler registration
- negotiated stream opening
- `io.Reader`/`io.Writer`/`io.Closer` access to streams

## Public API

- `type Config struct`
- `func DefaultConfig() Config`
- `func ConfigFromEnv() (Config, error)`
- `type Node struct`
- `func New(context.Context, Config, ...libp2p.Option) (*Node, error)`
- `func Wrap(host.Host) (*Node, error)`
- `func (*Node) Host() host.Host`
- `func (*Node) ID() peer.ID`
- `func (*Node) Addrs() []multiaddr.Multiaddr`
- `func (*Node) AddrInfo() peer.AddrInfo`
- `func (*Node) Connect(context.Context, peer.AddrInfo) error`
- `func (*Node) Handle(protocol.ID, Handler) error`
- `func (*Node) OpenStream(context.Context, peer.ID, protocol.ID) (Stream, error)`
- `type Stream struct`
- `func WrapStream(network.Stream) (Stream, error)`

## Semantic Sources

- `github.com/libp2p/go-libp2p@v0.48.0`: host construction and default stack.
- `github.com/libp2p/go-libp2p/core/host`: host lifecycle and streams.
- `github.com/libp2p/go-libp2p/core/network`: network stream contract.
- `libp2p.io/docs/getting-started-go`: basic host and stream-handler shape.

## Deferred

- DHT and routing helpers.
- Rendezvous protocol helpers.
- PubSub and topic lifecycle helpers.
- Relay, AutoNAT, hole punching, and resource-manager configuration.
- Browser/WebRTC-oriented APIs.
