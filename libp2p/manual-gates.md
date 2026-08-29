# Manual Gates: libp2p

Default `go test` uses loopback-only hosts and does not require external
network access, bootstrap peers, NAT traversal, relay infrastructure, or a DHT.

## WAN Gate

Command:

```sh
LIBP2P_TEST_WAN=1 go test -run '^TestIntegrationWAN$' -count=1 ./libp2p
```

This gate is reserved for explicit external-network or bootstrap-peer testing.
It fails closed until a concrete environment and test body are generated.
