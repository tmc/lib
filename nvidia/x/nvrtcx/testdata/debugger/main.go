package main

import (
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	if len(os.Args) != 2 {
		return fmt.Errorf("usage: probe kernel.cu")
	}
	d, err := cuda.NewDevice(0)
	if err != nil {
		return err
	}
	defer d.Close()
	source, err := os.ReadFile(os.Args[1])
	if err != nil {
		return err
	}
	cm, err := nvrtcx.Compile(string(source), nvrtcx.WithName(os.Args[1]), nvrtcx.WithDeviceArch(d), nvrtcx.WithDeviceDebug(), nvrtcx.WithNoCache())
	if err != nil {
		return err
	}
	m, err := cm.LoadModule(d)
	if err != nil {
		return err
	}
	defer m.Close()
	k, err := m.Function("debug_probe")
	if err != nil {
		return err
	}
	out, err := cudamem.Alloc[int32](d, 4)
	if err != nil {
		return err
	}
	defer out.Free()
	if err := cudalaunch.OwnedKernel(k).Block(4, 1, 1).Launch(out, int32(37)); err != nil {
		return err
	}
	host := make([]int32, 4)
	if err := out.CopyToHost(host); err != nil {
		return err
	}
	fmt.Println("debug output:", host)
	return nil
}
