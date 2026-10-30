// Package nvidiasmi parses the output of nvidia-smi and runs its GPU reset.
//
// [Parse] decodes the XML report written by nvidia-smi -q -x into a
// [Log] holding one [GPU] per device. Leaf readings are kept as
// [Value], the text nvidia-smi printed, such as "45 C", "70.12 W", or
// "N/A". Its methods parse the number and unit and report ok=false for
// placeholders like "N/A" and "[Not Supported]", so a missing reading
// is never mistaken for zero.
//
// Reports from nvsmi_device_v10.dtd through v13 decode into the same
// types. Where drivers renamed elements, both spellings fill one field:
// power_readings and gpu_power_readings fill [GPU.Power], and
// clocks_throttle_reasons and clocks_event_reasons fill
// [GPU.ClockEventReasons]. Elements with no field are kept in the Other
// fields rather than rejected.
//
//	f, err := os.Open("smi.xml")
//	...
//	report, err := nvidiasmi.Parse(f)
//	...
//	for _, g := range report.GPUs {
//		if w, ok := g.Power.Draw().Float(); ok {
//			fmt.Println(g.ID, w)
//		}
//	}
//
// [Query] runs nvidia-smi -q -x and parses its output.
//
// [ParseTopology] decodes the matrix written by nvidia-smi topo -m into a
// [Topology]: the [Link] between each pair of GPUs and NICs, such as
// "PIX" or "NV12", and each GPU's CPU and NUMA affinity.
// [QueryTopology] runs nvidia-smi topo -m and parses its output.
//
// [Reset] resets named GPUs with nvidia-smi --gpu-reset. A nonzero exit
// status from nvidia-smi is returned as an [Error].
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package nvidiasmi
