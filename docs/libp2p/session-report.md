# libp2p Session Report

Decision:

- Treat libp2p as a Go-semantic generated package, not a C/ABI binding.
- Keep the generated public API small.
- Use upstream go-libp2p as the implementation.
- Do not duplicate the full upstream symbol surface.

Local verification inputs:

- `go list -m -versions github.com/libp2p/go-libp2p`
- `go doc github.com/libp2p/go-libp2p.New`
- `go doc github.com/libp2p/go-libp2p/core/host.Host`
- `go doc github.com/libp2p/go-libp2p/core/network.Stream`
- `go doc github.com/libp2p/go-libp2p/core/peer.AddrInfo`

Selected upstream module:

- `github.com/libp2p/go-libp2p v0.48.0`

Manual gates not run:

- `LIBP2P_TEST_WAN=1 go test -run '^TestIntegrationWAN$' -count=1 ./libp2p`
