# nvlshm

Cross-process NVLink shared GPU buffers for Go, built on the CUDA VMM API
(`cuMemCreate` / `cuMemMap`) with optional NVSwitch multicast. A port of the
memory substrate of [MoonEP](https://github.com/MoonshotAI/MoonEP)
(`csrc/nvl_shared_buffer.cuh` + `moonep/buffer.py`), using the pure-Go
(purego) driver bindings in `github.com/tmc/lib/nvidia/driver`.

Each rank (one process per GPU) contributes one physical chunk; every rank
maps all chunks into one contiguous, all-read-write VA range:

    VA:  [ chunk 0 ][ chunk 1 ][ chunk 2 ][ chunk 3 ]
           GPU 0      GPU 1      GPU 2      GPU 3

A kernel on any rank writes directly into any other rank's chunk over
NVLink. With `DistAllocMulticast`, a store to the multicast address is fanned
out by the NVSwitch to the same offset in every rank's chunk.

## Usage

```go
runtime.LockOSThread()
driver.Init(0)
dev := driver.Device(rank)
ctx, _ := dev.PrimaryCtxRetain()
ctx.SetCurrent()

b, err := tcpboot.New(rank, world, "127.0.0.1:29500") // or any nvlshm.Bootstrap
if err != nil { ... }
defer b.Close()

gran, _ := nvlshm.Granularity(dev)
buf, err := nvlshm.DistAlloc(b, dev, nvlshm.PadTo(myChunkBytes, gran))
if err != nil { ... }

buf.ChunkPtr(peer) // write here → lands in peer's GPU memory
buf.LocalPtr()     // this rank's own chunk

b.Barrier() // never unmap while a peer may still be writing
buf.Close()
```

`Bootstrap` is two methods beyond rank/world — `BroadcastString` and
`Barrier` — so an MPI- or collective-library-backed implementation is a small
adapter. `tcpboot` is a minimal localhost hub for tests and single-host jobs.

## Multicast registration order

The sequence inside `DistAllocMulticast` is a driver contract; each barrier
gates a state transition, and violating the order fails with
`CUDA_ERROR_ILLEGAL_STATE` or hangs:

    rank 0                       ranks 1..N-1
    ──────                       ────────────
    cuMulticastCreate
    export POSIX fd  ──────────► import fd
    cuMulticastAddDevice         cuMulticastAddDevice
    ════════════ barrier: all devices added ════════════
    cuMulticastBindMem(owned)    cuMulticastBindMem(owned)
    map multicast VA             map multicast VA
    ════════════ barrier: all bound ════════════════════
              multicast VA usable everywhere

`BindMem` requires each rank's *owned* allocation handle (from
`cuMemCreate`) — binding an imported allocation is rejected — which is why
the handle is kept alive internally until after the bind.

## fd lifetime rules

- Exported fds are sent to all peers via `SCM_RIGHTS` over AF_UNIX datagram
  sockets in a rank-0-owned temp directory; the sender's fd is closed as soon
  as the exchange returns (a trailing barrier guarantees delivery).
- Received fds are owned by the receiver and closed immediately after
  `cuMemImportFromShareableHandle`.
- No fd survives setup.

## Testing

Unit tests (fd exchange, padding math, tcpboot collectives) run anywhere:

    go test ./nvidia/x/nvlshm/...

The GPU integration test re-execs itself as one process per GPU and needs a
multi-GPU Linux box with NVLink peer access (skips otherwise):

    go test -run TestDistBufferGPU -v ./nvidia/x/nvlshm/
