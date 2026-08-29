# Test Inventory: libp2p

Cheap tests:

- `TestConfigValidate` validates listen address parsing.
- `TestConfigFromEnv` validates small environment parsing.
- `TestLoopbackStream` creates two loopback hosts, connects them, opens a stream, and exchanges a fixed request/reply.
- `TestNodeClosed` validates closed-node behavior.
- `Example` checks the package example.

Manual tests:

- `TestIntegrationWAN` runs only with `LIBP2P_TEST_WAN=1` and currently fails closed until a real external-network body is supplied.
