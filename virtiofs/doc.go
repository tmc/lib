// Package virtiofs provides pure-Go virtio-fs request helpers over the FUSE
// protocol.
//
// The package intentionally does not implement a full VMM virtqueue datapath.
// It defines the stable Go boundary used by such a datapath: Config describes
// the advertised device, Transport exchanges encoded FUSE request and response
// bytes, and Device offers typed round trips using github.com/tmc/lib/fuse.
//
// The generated fake transport is for package tests and deterministic tools.
// Real VM and guest integration is an explicit manual gate.
package virtiofs
