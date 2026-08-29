// Package libp2p provides a small Go facade over go-libp2p hosts and
// streams.
//
// The package is intentionally not a copy of the go-libp2p API. Config covers
// common host construction, Node owns a host.Host, and Stream wraps negotiated
// network.Stream values with a small io-style surface. Advanced callers can use
// Node.Host to reach the underlying go-libp2p host.
package libp2p
