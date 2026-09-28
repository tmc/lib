// The correlation fixture emits its work contract independently of CUPTI.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	defer cudaprofagent.Finalize()
	// NVTX push/pop and CUDA submission must share the OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev, err := cuda.NewDevice(0)
	check(err)
	defer dev.Close()
	a, err := dev.NewStream()
	check(err)
	defer a.Destroy()
	b, err := dev.NewStream()
	check(err)
	defer b.Destroy()
	event, err := dev.NewEvent()
	check(err)
	defer event.Destroy()
	buf, err := cudamem.Alloc[int32](dev, 32)
	check(err)
	defer buf.Free()
	hashes := map[string]string{}
	sourceHashes := map[string]string{}
	kernels := make([]*cuda.Kernel, 2)
	// Deliberately use the same symbol in different loaded modules.
	for i, op := range []string{"prepare", "advance"} {
		src := fmt.Sprintf(`extern "C" __global__ void update(int *p) { p[threadIdx.x] += %d; }`, i+1)
		cm, err := nvrtcx.Compile(src, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
		check(err)
		m, err := cm.LoadModule(dev)
		check(err)
		defer m.Close()
		kernels[i], err = m.Function("update")
		check(err)
		hashes[op] = fmt.Sprintf("%x", sha256.Sum256(cm.CUBIN()))
		sourceHashes[op] = fmt.Sprintf("%x", sha256.Sum256([]byte(src)))
		check(os.WriteFile(filepath.Join(os.Args[1], hashes[op]+".cu"), []byte(src), 0600))
	}
	nvtx.RangePush("prepare")
	check(buf.CopyFromHost(make([]int32, 32)))
	check(cudalaunch.New(kernels[0]).Grid(1).Block(32).On(a).Launch(buf))
	check(event.Record(a))
	nvtx.RangePop()
	check(b.WaitEvent(event))
	g, err := cudagraph.New(dev)
	check(err)
	defer g.Destroy()
	desc, err := cudalaunch.New(kernels[1]).Grid(1).Block(32).Describe()
	check(err)
	_, err = g.AddLaunch(desc, []any{buf})
	check(err)
	ex, err := g.Instantiate()
	check(err)
	defer ex.Destroy()
	nvtx.RangePush("advance")
	for i := 0; i < 3; i++ {
		check(ex.Launch(b))
	}
	check(b.Synchronize(context.Background()))
	out := make([]int32, 32)
	check(buf.CopyToHost(out))
	nvtx.RangePop()
	for i, v := range out {
		if v != 7 {
			panic(fmt.Sprintf("output[%d]=%d, want 7", i, v))
		}
	}
	check(json.NewEncoder(os.Stdout).Encode(struct {
		Kernels      int               `json:"kernels"`
		Copies       int               `json:"copies"`
		GraphReplays int               `json:"graph_replays"`
		Hashes       map[string]string `json:"hashes"`
		SourceHashes map[string]string `json:"source_hashes"`
	}{4, 2, 3, hashes, sourceHashes}))
}
