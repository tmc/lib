package autotune

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

// HardwareTwin models discovered or calibrated GPU device, toolchain, and
// NVLink/C2C rack fabric attributes used by the e-graph autotuner.
type HardwareTwin struct {
	Target                       string  `json:"target"`
	ComputeMajor                 int     `json:"compute_major"`
	ComputeMinor                 int     `json:"compute_minor"`
	SMCount                      int     `json:"sm_count"`
	L2CacheBytes                 int64   `json:"l2_cache_bytes"`
	HBMBytes                     int64   `json:"hbm_bytes"`
	HBMBandwidthGBps             float64 `json:"hbm_bandwidth_gbps"`
	TMEMColumns                  int     `json:"tmem_columns"`
	MaxClusterCTAs               int     `json:"max_cluster_ctas"`
	SupportsTCGen05              bool    `json:"supports_tcgen05"`
	SupportsScaleVec4X           bool    `json:"supports_scale_vec_4x"`
	SupportsLUTDecompressB       bool    `json:"supports_lut_decompress_b"`
	SupportsSyncRestrictReadA    bool    `json:"supports_sync_restrict_read_a"`
	SupportsHardwareCvtE2M1ToF16 bool    `json:"supports_hardware_cvt_e2m1_to_f16"`
	SupportsFabricAsyncTensor    bool    `json:"supports_fabric_async_tensor"`
	PTXISAVersion                string  `json:"ptx_isa_version"`
	NVLinkBandwidthGBps          float64 `json:"nvlink_bandwidth_gbps"`
	C2CBandwidthGBps             float64 `json:"c2c_bandwidth_gbps"`
	RackRanks                    int     `json:"rack_ranks"`
}

const (
	mib = 1024 * 1024
	gib = 1024 * 1024 * 1024
)

// twinProducts names the product whose [cudadevicetable.Profile] supplies a
// target's SM count, memory and NVLink bandwidth, Tensor Memory, cluster
// limit and tcgen05 support. Where the table holds a probe of the product,
// the probed values, including its L2 size, replace the datasheet ones.
// sm_107a and sm_107f have no profile and are entirely hand-entered.
var twinProducts = map[string]string{
	"sm_90a":  "H200",
	"sm_100a": "B200",
	"sm_103a": "GB300",
	"sm_120a": "RTX 5090",
	"sm_121a": "GB10",
}

// tmemColumnBytes is the size of one Tensor Memory column: 128 lanes of 32
// bits.
const tmemColumnBytes = 128 * 4

// TwinForTarget returns the calibrated [HardwareTwin] for one of the supported
// target architecture identifiers: "sm_90a", "sm_100a", "sm_103a", "sm_107a",
// "sm_107f", "sm_120a", or "sm_121a".
func TwinForTarget(target string) (HardwareTwin, error) {
	t := strings.ToLower(strings.TrimSpace(target))
	twin, ok := handTwin(t)
	if !ok {
		return HardwareTwin{}, fmt.Errorf("twin for target: unsupported target %q", target)
	}
	product, ok := twinProducts[t]
	if !ok {
		return twin, nil
	}
	p, err := cudadevicetable.LookupProfile(product)
	if err != nil {
		return HardwareTwin{}, fmt.Errorf("twin for target %s: %w", t, err)
	}
	tcgen05, err := p.Supports("tcgen05")
	if err != nil {
		return HardwareTwin{}, fmt.Errorf("twin for target %s: %w", t, err)
	}
	twin.SMCount = int(p.SMs.Value)
	twin.HBMBandwidthGBps = float64(p.MemoryGBps.Value)
	twin.NVLinkBandwidthGBps = float64(p.NVLinkGBps.Value)
	twin.TMEMColumns = int(p.TMEMPerSM.Value / tmemColumnBytes)
	twin.MaxClusterCTAs = int(p.ClusterSizeOptin.Value)
	twin.SupportsTCGen05 = tcgen05
	if l2, ok := probedL2(t, product); ok {
		twin.L2CacheBytes = l2
	}
	return twin, nil
}

// probedL2 returns the L2 size the table read from product, if the table's
// probe of target's architecture is of that product.
func probedL2(target, product string) (int64, bool) {
	a, ok := cudadevicetable.Lookup(target)
	if !ok || a.Probe == nil || !strings.EqualFold(strings.TrimPrefix(a.Probe.Device, "NVIDIA "), product) {
		return 0, false
	}
	l2, err := a.ProbedAttr("l2CacheSize")
	return l2, err == nil
}

// handTwin returns the hand-entered part of target's twin: the facts the
// device table does not hold.
func handTwin(t string) (HardwareTwin, bool) {
	switch t {
	case "sm_90a":
		return HardwareTwin{
			Target:                       "sm_90a",
			ComputeMajor:                 9,
			ComputeMinor:                 0,
			L2CacheBytes:                 50 * mib,
			HBMBytes:                     141 * gib,
			SupportsScaleVec4X:           false,
			SupportsLUTDecompressB:       false,
			SupportsSyncRestrictReadA:    false,
			SupportsHardwareCvtE2M1ToF16: false,
			SupportsFabricAsyncTensor:    false,
			PTXISAVersion:                "8.5",
			C2CBandwidthGBps:             900.0,
			RackRanks:                    8,
		}, true
	case "sm_100a":
		return HardwareTwin{
			Target:                       "sm_100a",
			ComputeMajor:                 10,
			ComputeMinor:                 0,
			L2CacheBytes:                 126 * mib,
			HBMBytes:                     192 * gib,
			SupportsScaleVec4X:           false,
			SupportsLUTDecompressB:       false,
			SupportsSyncRestrictReadA:    false,
			SupportsHardwareCvtE2M1ToF16: true,
			SupportsFabricAsyncTensor:    false,
			PTXISAVersion:                "8.5",
			C2CBandwidthGBps:             900.0,
			RackRanks:                    72,
		}, true
	case "sm_103a":
		return HardwareTwin{
			Target:                       "sm_103a",
			ComputeMajor:                 10,
			ComputeMinor:                 3,
			L2CacheBytes:                 126 * mib,
			HBMBytes:                     288 * gib,
			SupportsScaleVec4X:           true,
			SupportsLUTDecompressB:       false,
			SupportsSyncRestrictReadA:    false,
			SupportsHardwareCvtE2M1ToF16: true,
			SupportsFabricAsyncTensor:    false,
			PTXISAVersion:                "9.0",
			C2CBandwidthGBps:             900.0,
			RackRanks:                    72,
		}, true
	case "sm_107a":
		return HardwareTwin{
			Target:                       "sm_107a",
			ComputeMajor:                 10,
			ComputeMinor:                 7,
			SMCount:                      224,
			L2CacheBytes:                 256 * mib,
			HBMBytes:                     288 * gib,
			HBMBandwidthGBps:             22000.0,
			TMEMColumns:                  512,
			MaxClusterCTAs:               32,
			SupportsTCGen05:              true,
			SupportsScaleVec4X:           true,
			SupportsLUTDecompressB:       true,
			SupportsSyncRestrictReadA:    true,
			SupportsHardwareCvtE2M1ToF16: true,
			SupportsFabricAsyncTensor:    true,
			PTXISAVersion:                "9.4",
			NVLinkBandwidthGBps:          3600.0,
			C2CBandwidthGBps:             1800.0,
			RackRanks:                    72,
		}, true
	case "sm_107f":
		return HardwareTwin{
			Target:                       "sm_107f",
			ComputeMajor:                 10,
			ComputeMinor:                 7,
			SMCount:                      224,
			L2CacheBytes:                 256 * mib,
			HBMBytes:                     288 * gib,
			HBMBandwidthGBps:             22000.0,
			TMEMColumns:                  512,
			MaxClusterCTAs:               32,
			SupportsTCGen05:              true,
			SupportsScaleVec4X:           true,
			SupportsLUTDecompressB:       true,
			SupportsSyncRestrictReadA:    true,
			SupportsHardwareCvtE2M1ToF16: true,
			SupportsFabricAsyncTensor:    true,
			PTXISAVersion:                "9.4",
			NVLinkBandwidthGBps:          3600.0,
			C2CBandwidthGBps:             1800.0,
			RackRanks:                    144,
		}, true
	case "sm_120a":
		return HardwareTwin{
			Target:                       "sm_120a",
			ComputeMajor:                 12,
			ComputeMinor:                 0,
			L2CacheBytes:                 96 * mib,
			HBMBytes:                     32 * gib,
			SupportsScaleVec4X:           false,
			SupportsLUTDecompressB:       false,
			SupportsSyncRestrictReadA:    false,
			SupportsHardwareCvtE2M1ToF16: true,
			SupportsFabricAsyncTensor:    false,
			PTXISAVersion:                "8.5",
			C2CBandwidthGBps:             0.0,
			RackRanks:                    1,
		}, true
	case "sm_121a":
		return HardwareTwin{
			Target:                       "sm_121a",
			ComputeMajor:                 12,
			ComputeMinor:                 1,
			HBMBytes:                     128 * gib,
			SupportsScaleVec4X:           false,
			SupportsLUTDecompressB:       false,
			SupportsSyncRestrictReadA:    false,
			SupportsHardwareCvtE2M1ToF16: false,
			SupportsFabricAsyncTensor:    false,
			PTXISAVersion:                "9.0",
			C2CBandwidthGBps:             600.0,
			RackRanks:                    1,
		}, true
	default:
		return HardwareTwin{}, false
	}
}

// AutoDiscoverTarget maps a runtime GPU compute capability (major, minor) and
// fabric/TMEM capability flags to the optimal [HardwareTwin].
func AutoDiscoverTarget(major, minor int, hasNVLinkFabric, hasExclusiveTMEM bool) (HardwareTwin, error) {
	var target string
	switch {
	case major == 9 && minor == 0:
		target = "sm_90a"
	case major == 10 && minor == 0:
		target = "sm_100a"
	case major == 10 && minor == 3:
		target = "sm_103a"
	case major == 10 && minor == 7:
		if hasNVLinkFabric {
			target = "sm_107f"
		} else {
			target = "sm_107a"
		}
	case major == 12 && minor == 0:
		target = "sm_120a"
	case major == 12 && minor == 1:
		target = "sm_121a"
	default:
		return HardwareTwin{}, fmt.Errorf("auto discover target: unsupported compute capability %d.%d", major, minor)
	}

	twin, err := TwinForTarget(target)
	if err != nil {
		return HardwareTwin{}, err
	}
	if !hasNVLinkFabric {
		twin.NVLinkBandwidthGBps = 0
		twin.RackRanks = 1
	}
	if hasExclusiveTMEM && twin.SupportsTCGen05 {
		twin.TMEMColumns = 576
	}
	return twin, nil
}
