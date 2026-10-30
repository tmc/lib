package nvidiasmi_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvidiasmi"
)

const report = `<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
<nvidia_smi_log>
	<driver_version>570.148.08</driver_version>
	<attached_gpus>1</attached_gpus>
	<gpu id="00000000:01:00.0">
		<product_name>NVIDIA L4</product_name>
		<fb_memory_usage><total>23034 MiB</total><used>0 MiB</used></fb_memory_usage>
		<temperature><gpu_temp>45 C</gpu_temp><memory_temp>N/A</memory_temp></temperature>
		<gpu_power_readings>
			<average_power_draw>16.20 W</average_power_draw>
			<current_power_limit>72.00 W</current_power_limit>
		</gpu_power_readings>
	</gpu>
</nvidia_smi_log>`

func Example() {
	r, err := nvidiasmi.Parse(strings.NewReader(report))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(r.DTDVersion, r.DriverVersion)
	for _, g := range r.GPUs {
		w, _ := g.Power.Draw().Float()
		mem, _ := g.FBMemory.Total.Bytes()
		fmt.Println(g.ID, g.ProductName, w, g.Power.Limit, mem>>30, "GiB")
	}
	// Output:
	// 12 570.148.08
	// 00000000:01:00.0 NVIDIA L4 16.2 72.00 W 22 GiB
}

func ExampleValue_Float() {
	for _, v := range []nvidiasmi.Value{"45 C", "N/A", "[Not Supported]"} {
		f, ok := v.Float()
		fmt.Println(f, ok)
	}
	// Output:
	// 45 true
	// 0 false
	// 0 false
}

func ExampleParseTopology() {
	const out = "\tGPU0\tGPU1\tNIC0\tCPU Affinity\tNUMA Affinity\n" +
		"GPU0\t X \tNV12\tPIX\t0-15\t0\n" +
		"GPU1\tNV12\t X \tSYS\t16-31\t1\n" +
		"NIC0\tPIX\tSYS\t X \n" +
		"\n" +
		"NIC Legend:\n" +
		"\n" +
		"  NIC0: mlx5_0\n"
	topo, err := nvidiasmi.ParseTopology(strings.NewReader(out))
	if err != nil {
		log.Fatal(err)
	}
	l, _ := topo.Link("GPU0", "GPU1")
	n, _ := l.NVLinks()
	fmt.Println(l, n)
	l, _ = topo.Link("GPU0", "NIC0")
	fmt.Println(l, topo.NICs["NIC0"])
	cpus, _ := topo.Devices[1].CPUAffinity.IntList()
	fmt.Println(len(cpus), cpus[0])
	// Output:
	// NV12 12
	// PIX mlx5_0
	// 16 16
}

func ExampleValue_IntList() {
	list, ok := nvidiasmi.Value("0-3,8").IntList()
	fmt.Println(list, ok)
	// Output:
	// [0 1 2 3 8] true
}

func ExampleReset() {
	// Reset the GPU with this UUID. This needs root and an idle GPU.
	err := nvidiasmi.Reset(context.Background(), "GPU-25e04bdf-6a27-0a2d-f989-a9228a03f373")
	var e *nvidiasmi.Error
	switch {
	case errors.As(err, &e):
		log.Fatalf("nvidia-smi exited %d: %s%s", e.Code, e.Stdout, e.Stderr)
	case err != nil:
		log.Fatal(err)
	}
}
