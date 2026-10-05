// Package httpapi exposes cudahotswap.Service as bounded JSON over HTTP.
//
// New accepts an application service and registries of named native resources.
// Clients receive opaque lifetime handles, never raw CUDA pointers. The
// application must keep its resources live until Handler.Close succeeds.
//
// Listen accepts unix:/absolute/socket or tcp:host:port. Use an SSH forward for
// remote TCP access; this package does not provide authentication or TLS.
// Subscribe uses /v1/events with a since cursor, as SSE or JSON lines selected
// by Accept: application/x-ndjson. Client.Events owns the response body and must
// be closed on disconnect.
package httpapi
