package cudacollective

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// ExportTestNCCLTransport returns an NCCLTransport whose vendor calls are
// info, invoke and complete, for external tests. It exists only in this
// package's test binary. Comms and Streams are CPU test-only handles that no
// vendor code receives; invoke sees each validated operation and its
// buffers. No CUDA context is looked up.
func ExportTestNCCLTransport(comms []nccl.Comm, streams []driver.Stream, info func(nccl.Comm) (int, int, error), invoke func(op OperationDescriptor, send, recv driver.DevicePtr) error, complete func(driver.Stream) error) *NCCLTransport {
	return &NCCLTransport{Comms: comms, Streams: streams, calls: &ncclCalls{
		info:     info,
		invoke:   func(q ncclInvocation) error { return invoke(q.op, q.send, q.recv) },
		complete: complete,
	}}
}
