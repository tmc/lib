package gpuhealth_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/gpuhealth"
)

func ExampleParseSMIXML() {
	out := `<?xml version="1.0" ?>
<nvidia_smi_log>
	<gpu id="00000000:19:00.0">
		<product_name>NVIDIA H200</product_name>
		<uuid>GPU-1234-abcd</uuid>
		<minor_number>7</minor_number>
	</gpu>
</nvidia_smi_log>`
	gpus, err := gpuhealth.ParseSMIXML([]byte(out))
	if err != nil {
		panic(err)
	}
	g, err := gpuhealth.Select(gpus, "GPU-1234-abcd")
	if err != nil {
		panic(err)
	}
	args, _ := gpuhealth.DockerArgs(g)
	fmt.Println(g.Name, g.DevicePath())
	fmt.Println(strings.Join(args[:2], " "))
	// Output:
	// NVIDIA H200 /dev/nvidia7
	// --device /dev/nvidia7:/dev/nvidia7:rw
}
