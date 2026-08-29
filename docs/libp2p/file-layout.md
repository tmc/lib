# File Layout: libp2p

- `doc.go`: package documentation.
- `errors.go`: package errors.
- `config.go`: small generated Config and environment parsing.
- `node.go`: Node lifecycle, host access, peer connect, handlers, and streams.
- `stream.go`: Stream wrapper over go-libp2p network.Stream.
- `example_test.go`: package example.
- `config_test.go`: cheap Config tests.
- `node_test.go`: loopback host/stream smoke.
- `integration_test.go`: WAN manual gate placeholder.
- `manual-gates.md`: explicit manual gate notes.
