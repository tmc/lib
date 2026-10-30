package nvidiasmi

import (
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// A Log is one nvidia-smi -q -x report.
type Log struct {
	// DTD is the system identifier from the report's DOCTYPE,
	// such as "nvsmi_device_v12.dtd", or "" if there is none.
	DTD string `xml:"-"`

	// DTDVersion is the NN in nvsmi_device_vNN.dtd, or 0 if DTD
	// does not have that form.
	DTDVersion int `xml:"-"`

	Timestamp     string `xml:"timestamp"`
	DriverVersion string `xml:"driver_version"`
	CUDAVersion   string `xml:"cuda_version"`

	// AttachedGPUs is the count nvidia-smi reports. It can differ
	// from len(GPUs) when the report was filtered or edited.
	AttachedGPUs int   `xml:"attached_gpus"`
	GPUs         []GPU `xml:"gpu"`

	// Other holds top-level elements with no field above.
	Other []Element `xml:",any"`
}

// A GPU is the report for one device.
type GPU struct {
	// ID is the PCI bus ID from the id attribute, such as "00000000:01:00.0".
	ID string `xml:"id,attr"`

	ProductName         string `xml:"product_name"`
	ProductBrand        string `xml:"product_brand"`
	ProductArchitecture string `xml:"product_architecture"`
	UUID                string `xml:"uuid"`
	VBIOSVersion        string `xml:"vbios_version"`
	Serial              Value  `xml:"serial"`
	MinorNumber         Value  `xml:"minor_number"`
	BoardID             Value  `xml:"board_id"`
	BoardPartNumber     Value  `xml:"board_part_number"`
	GPUPartNumber       Value  `xml:"gpu_part_number"`
	ModuleID            Value  `xml:"gpu_module_id"`
	GSPFirmwareVersion  Value  `xml:"gsp_firmware_version"`

	DisplayMode        Value `xml:"display_mode"`
	DisplayActive      Value `xml:"display_active"`
	PersistenceMode    Value `xml:"persistence_mode"`
	AddressingMode     Value `xml:"addressing_mode"`
	ComputeMode        Value `xml:"compute_mode"`
	PerformanceState   Value `xml:"performance_state"`
	FanSpeed           Value `xml:"fan_speed"`
	VirtualizationMode Value `xml:"gpu_virtualization_mode>virtualization_mode"`
	MIGMode            Value `xml:"mig_mode>current_mig"`
	PendingMIGMode     Value `xml:"mig_mode>pending_mig"`
	ECCMode            Value `xml:"ecc_mode>current_ecc"`
	PendingECCMode     Value `xml:"ecc_mode>pending_ecc"`

	// ResetRequired and DrainAndResetRecommended come from
	// gpu_reset_status, which newer reports replace with
	// RecoveryAction.
	ResetRequired            Value `xml:"gpu_reset_status>reset_required"`
	DrainAndResetRecommended Value `xml:"gpu_reset_status>drain_and_reset_recommended"`
	RecoveryAction           Value `xml:"gpu_recovery_action"`

	MIGDevices []MIGDevice `xml:"mig_devices>mig_device"`
	Platform   Platform    `xml:"platformInfo"`
	InfoROM    InfoROM     `xml:"inforom_version"`
	PCI        PCI         `xml:"pci"`

	// ClockEventReasons is decoded from clocks_event_reasons or,
	// in older reports, clocks_throttle_reasons.
	ClockEventReasons ClockEventReasons `xml:"-"`

	FBMemory     Memory       `xml:"fb_memory_usage"`
	BAR1Memory   Memory       `xml:"bar1_memory_usage"`
	Utilization  Utilization  `xml:"utilization"`
	ECCErrors    ECCErrors    `xml:"ecc_errors"`
	RetiredPages RetiredPages `xml:"retired_pages"`
	RemappedRows RemappedRows `xml:"remapped_rows"`
	Temperature  Temperature  `xml:"temperature"`

	// Power is decoded from gpu_power_readings or, in older
	// reports, power_readings.
	Power       Power `xml:"-"`
	ModulePower Power `xml:"module_power_readings"`
	MemoryPower Power `xml:"gpu_memory_power_readings"`

	Clocks                    Clocks `xml:"clocks"`
	ApplicationsClocks        Clocks `xml:"applications_clocks"`
	DefaultApplicationsClocks Clocks `xml:"default_applications_clocks"`
	MaxClocks                 Clocks `xml:"max_clocks"`
	GraphicsVoltage           Value  `xml:"voltage>graphics_volt"`

	Fabric    Fabric    `xml:"fabric"`
	Processes []Process `xml:"processes>process_info"`

	// Other holds elements of the gpu element with no field above.
	Other []Element `xml:",any"`
}

// UnmarshalXML decodes a gpu element, accepting both the old and new
// names of the power and clock event reason elements.
func (g *GPU) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	type gpu GPU // drops this method
	var x struct {
		gpu
		Power         *Power             `xml:"gpu_power_readings"`
		LegacyPower   *Power             `xml:"power_readings"`
		Reasons       *ClockEventReasons `xml:"clocks_event_reasons"`
		LegacyReasons *ClockEventReasons `xml:"clocks_throttle_reasons"`
	}
	if err := d.DecodeElement(&x, &start); err != nil {
		return err
	}
	*g = GPU(x.gpu)
	switch {
	case x.Power != nil:
		g.Power = *x.Power
	case x.LegacyPower != nil:
		g.Power = *x.LegacyPower
	}
	switch {
	case x.Reasons != nil:
		g.ClockEventReasons = *x.Reasons
	case x.LegacyReasons != nil:
		g.ClockEventReasons = *x.LegacyReasons
	}
	return nil
}

// A MIGDevice is one MIG instance of a GPU in MIG mode.
type MIGDevice struct {
	Index             Value `xml:"index"`
	GPUInstanceID     Value `xml:"gpu_instance_id"`
	ComputeInstanceID Value `xml:"compute_instance_id"`

	MultiprocessorCount Value `xml:"device_attributes>shared>multiprocessor_count"`
	CopyEngineCount     Value `xml:"device_attributes>shared>copy_engine_count"`
	EncoderCount        Value `xml:"device_attributes>shared>encoder_count"`
	DecoderCount        Value `xml:"device_attributes>shared>decoder_count"`
	OFACount            Value `xml:"device_attributes>shared>ofa_count"`
	JPEGCount           Value `xml:"device_attributes>shared>jpg_count"`

	FBMemory   Memory `xml:"fb_memory_usage"`
	BAR1Memory Memory `xml:"bar1_memory_usage"`
}

// Platform is the rack position of a GPU in a multi-node system such as
// GB200 NVL72. Single-node systems report N/A.
type Platform struct {
	ChassisSerialNumber Value `xml:"chassis_serial_number"`
	SlotNumber          Value `xml:"slot_number"`
	TrayIndex           Value `xml:"tray_index"`
	HostID              Value `xml:"host_id"`
	PeerType            Value `xml:"peer_type"`
	ModuleID            Value `xml:"module_id"`
	GPUFabricGUID       Value `xml:"gpu_fabric_guid"`
}

// InfoROM holds the versions of the GPU's InfoROM objects.
type InfoROM struct {
	Image Value `xml:"img_version"`
	OEM   Value `xml:"oem_object"`
	ECC   Value `xml:"ecc_object"`
	Power Value `xml:"pwr_object"`
}

// PCI describes a GPU's PCI identity and link.
type PCI struct {
	Bus         Value `xml:"pci_bus"`
	Device      Value `xml:"pci_device"`
	Domain      Value `xml:"pci_domain"`
	DeviceID    Value `xml:"pci_device_id"`
	BusID       Value `xml:"pci_bus_id"`
	SubSystemID Value `xml:"pci_sub_system_id"`

	MaxLinkGen       Value `xml:"pci_gpu_link_info>pcie_gen>max_link_gen"`
	CurrentLinkGen   Value `xml:"pci_gpu_link_info>pcie_gen>current_link_gen"`
	MaxLinkWidth     Value `xml:"pci_gpu_link_info>link_widths>max_link_width"`
	CurrentLinkWidth Value `xml:"pci_gpu_link_info>link_widths>current_link_width"`

	ReplayCounter         Value `xml:"replay_counter"`
	ReplayRolloverCounter Value `xml:"replay_rollover_counter"`
	TxUtil                Value `xml:"tx_util"`
	RxUtil                Value `xml:"rx_util"`
}

// ClockEventReasons reports why clocks are below their maximum, each as
// "Active" or "Not Active".
type ClockEventReasons struct {
	GPUIdle                   Value
	ApplicationsClocksSetting Value
	SWPowerCap                Value
	HWSlowdown                Value
	HWThermalSlowdown         Value
	HWPowerBrakeSlowdown      Value
	SyncBoost                 Value
	SWThermalSlowdown         Value
	DisplayClocksSetting      Value

	// Other holds reasons with no field above, keyed by element name
	// without its clocks_event_reason_ or clocks_throttle_reason_ prefix.
	Other map[string]Value
}

// UnmarshalXML decodes a clocks_event_reasons or clocks_throttle_reasons
// element.
func (r *ClockEventReasons) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	var x struct {
		Reasons []struct {
			XMLName xml.Name
			Text    Value `xml:",chardata"`
		} `xml:",any"`
	}
	if err := d.DecodeElement(&x, &start); err != nil {
		return err
	}
	for _, e := range x.Reasons {
		name := e.XMLName.Local
		name = strings.TrimPrefix(name, "clocks_event_reason_")
		name = strings.TrimPrefix(name, "clocks_throttle_reason_")
		var p *Value
		switch name {
		case "gpu_idle":
			p = &r.GPUIdle
		case "applications_clocks_setting":
			p = &r.ApplicationsClocksSetting
		case "sw_power_cap":
			p = &r.SWPowerCap
		case "hw_slowdown":
			p = &r.HWSlowdown
		case "hw_thermal_slowdown":
			p = &r.HWThermalSlowdown
		case "hw_power_brake_slowdown":
			p = &r.HWPowerBrakeSlowdown
		case "sync_boost":
			p = &r.SyncBoost
		case "sw_thermal_slowdown":
			p = &r.SWThermalSlowdown
		case "display_clocks_setting":
			p = &r.DisplayClocksSetting
		default:
			if r.Other == nil {
				r.Other = make(map[string]Value)
			}
			r.Other[name] = e.Text
			continue
		}
		*p = e.Text
	}
	return nil
}

// Memory is a memory usage report, in MiB.
type Memory struct {
	Total    Value `xml:"total"`
	Reserved Value `xml:"reserved"`
	Used     Value `xml:"used"`
	Free     Value `xml:"free"`
}

// Utilization holds engine utilization percentages.
type Utilization struct {
	GPU     Value `xml:"gpu_util"`
	Memory  Value `xml:"memory_util"`
	Encoder Value `xml:"encoder_util"`
	Decoder Value `xml:"decoder_util"`
	JPEG    Value `xml:"jpeg_util"`
	OFA     Value `xml:"ofa_util"`
}

// ECCErrors holds ECC error counts since the driver loaded (Volatile)
// and over the GPU's lifetime (Aggregate).
type ECCErrors struct {
	Volatile  ECCCounts `xml:"volatile"`
	Aggregate ECCCounts `xml:"aggregate"`
}

// ECCCounts holds one set of ECC error counts. Most reports use the
// SRAM and DRAM fields; newer ones split SRAMUncorrectable into
// SRAMUncorrectableParity and SRAMUncorrectableSECDED, and some older
// ones use SingleBit and DoubleBit instead.
type ECCCounts struct {
	SRAMCorrectable         Value `xml:"sram_correctable"`
	SRAMUncorrectable       Value `xml:"sram_uncorrectable"`
	SRAMUncorrectableParity Value `xml:"sram_uncorrectable_parity"`
	SRAMUncorrectableSECDED Value `xml:"sram_uncorrectable_secded"`
	DRAMCorrectable         Value `xml:"dram_correctable"`
	DRAMUncorrectable       Value `xml:"dram_uncorrectable"`
	SRAMThresholdExceeded   Value `xml:"sram_threshold_exceeded"`

	SingleBit ECCLocations `xml:"single_bit"`
	DoubleBit ECCLocations `xml:"double_bit"`
}

// ECCLocations holds ECC error counts by location, as in some older
// reports.
type ECCLocations struct {
	DeviceMemory  Value `xml:"device_memory"`
	RegisterFile  Value `xml:"register_file"`
	L1Cache       Value `xml:"l1_cache"`
	L2Cache       Value `xml:"l2_cache"`
	TextureMemory Value `xml:"texture_memory"`
	TextureShm    Value `xml:"texture_shm"`
	CBU           Value `xml:"cbu"`
	Total         Value `xml:"total"`
}

// RetiredPages reports dynamic page retirement, used before Ampere.
type RetiredPages struct {
	SingleBitCount    Value `xml:"multiple_single_bit_retirement>retired_count"`
	DoubleBitCount    Value `xml:"double_bit_retirement>retired_count"`
	PendingBlacklist  Value `xml:"pending_blacklist"`
	PendingRetirement Value `xml:"pending_retirement"`
}

// RemappedRows reports row remapping, used from Ampere on. Every field
// is unavailable when the report has "N/A" in place of the element's
// children.
type RemappedRows struct {
	Correctable   Value `xml:"remapped_row_corr"`
	Uncorrectable Value `xml:"remapped_row_unc"`
	Pending       Value `xml:"remapped_row_pending"`
	Failure       Value `xml:"remapped_row_failure"`

	HistogramMax     Value `xml:"row_remapper_histogram>row_remapper_histogram_max"`
	HistogramHigh    Value `xml:"row_remapper_histogram>row_remapper_histogram_high"`
	HistogramPartial Value `xml:"row_remapper_histogram>row_remapper_histogram_partial"`
	HistogramLow     Value `xml:"row_remapper_histogram>row_remapper_histogram_low"`
	HistogramNone    Value `xml:"row_remapper_histogram>row_remapper_histogram_none"`
}

// Temperature holds temperatures and thresholds in degrees C.
// The TLimit fields, in newer reports, are margins relative to a
// limit rather than absolute temperatures.
type Temperature struct {
	GPU             Value `xml:"gpu_temp"`
	Memory          Value `xml:"memory_temp"`
	Target          Value `xml:"gpu_target_temperature"`
	MaxThreshold    Value `xml:"gpu_temp_max_threshold"`
	SlowThreshold   Value `xml:"gpu_temp_slow_threshold"`
	MaxGPUThreshold Value `xml:"gpu_temp_max_gpu_threshold"`
	MaxMemThreshold Value `xml:"gpu_temp_max_mem_threshold"`

	TLimit                Value `xml:"gpu_temp_tlimit"`
	MaxTLimitThreshold    Value `xml:"gpu_temp_max_tlimit_threshold"`
	SlowTLimitThreshold   Value `xml:"gpu_temp_slow_tlimit_threshold"`
	MaxGPUTLimitThreshold Value `xml:"gpu_temp_max_gpu_tlimit_threshold"`
	MaxMemTLimitThreshold Value `xml:"gpu_temp_max_mem_tlimit_threshold"`
}

// Power holds power readings and limits in watts.
//
// Reports name these differently by driver: power_readings has
// power_limit and enforced_power_limit, while gpu_power_readings has
// requested_power_limit and current_power_limit, and in newer reports
// average_power_draw and instant_power_draw instead of power_draw.
type Power struct {
	State      Value
	Management Value // power_readings only

	PowerDraw   Value
	AverageDraw Value
	InstantDraw Value

	// Limit is the enforced limit: enforced_power_limit or
	// current_power_limit.
	Limit Value

	// RequestedLimit is the limit set by software: power_limit or
	// requested_power_limit.
	RequestedLimit Value

	DefaultLimit Value
	MinLimit     Value
	MaxLimit     Value
}

// Draw returns the first available of p.PowerDraw, p.AverageDraw, and
// p.InstantDraw, or p.PowerDraw if none is available.
func (p Power) Draw() Value {
	for _, v := range []Value{p.PowerDraw, p.AverageDraw, p.InstantDraw} {
		if v.Available() {
			return v
		}
	}
	return p.PowerDraw
}

// UnmarshalXML decodes any of the power reading elements.
func (p *Power) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	var x struct {
		State          Value `xml:"power_state"`
		Management     Value `xml:"power_management"`
		PowerDraw      Value `xml:"power_draw"`
		AverageDraw    Value `xml:"average_power_draw"`
		InstantDraw    Value `xml:"instant_power_draw"`
		EnforcedLimit  Value `xml:"enforced_power_limit"`
		CurrentLimit   Value `xml:"current_power_limit"`
		PowerLimit     Value `xml:"power_limit"`
		RequestedLimit Value `xml:"requested_power_limit"`
		DefaultLimit   Value `xml:"default_power_limit"`
		MinLimit       Value `xml:"min_power_limit"`
		MaxLimit       Value `xml:"max_power_limit"`
	}
	if err := d.DecodeElement(&x, &start); err != nil {
		return err
	}
	*p = Power{
		State:          x.State,
		Management:     x.Management,
		PowerDraw:      x.PowerDraw,
		AverageDraw:    x.AverageDraw,
		InstantDraw:    x.InstantDraw,
		Limit:          first(x.CurrentLimit, x.EnforcedLimit),
		RequestedLimit: first(x.RequestedLimit, x.PowerLimit),
		DefaultLimit:   x.DefaultLimit,
		MinLimit:       x.MinLimit,
		MaxLimit:       x.MaxLimit,
	}
	return nil
}

// first returns the first present (non-empty) value.
func first(vs ...Value) Value {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

// Clocks holds clock frequencies in MHz.
type Clocks struct {
	Graphics Value `xml:"graphics_clock"`
	SM       Value `xml:"sm_clock"`
	Memory   Value `xml:"mem_clock"`
	Video    Value `xml:"video_clock"`
}

// Fabric reports NVLink fabric registration, as on GB200 and H100 NVSwitch
// systems. Other GPUs report N/A.
type Fabric struct {
	State       Value `xml:"state"`
	Status      Value `xml:"status"`
	CliqueID    Value `xml:"cliqueId"`
	ClusterUUID Value `xml:"clusterUuid"`

	HealthSummary           Value `xml:"health>summary"`
	Bandwidth               Value `xml:"health>bandwidth"`
	RouteRecoveryInProgress Value `xml:"health>route_recovery_in_progress"`
	RouteUnhealthy          Value `xml:"health>route_unhealthy"`
	AccessTimeoutRecovery   Value `xml:"health>access_timeout_recovery"`
	IncorrectConfiguration  Value `xml:"health>incorrect_configuration"`
	PartitionAssigned       Value `xml:"health>partition_assigned"`
}

// A Process is one process using a GPU.
type Process struct {
	GPUInstanceID     Value  `xml:"gpu_instance_id"`
	ComputeInstanceID Value  `xml:"compute_instance_id"`
	PID               Value  `xml:"pid"`
	Type              Value  `xml:"type"`
	Name              string `xml:"process_name"`
	UsedMemory        Value  `xml:"used_memory"`
}

// An Element is an XML element kept without interpretation.
type Element struct {
	XMLName  xml.Name
	Attrs    []xml.Attr `xml:",any,attr"`
	Text     string     `xml:",chardata"`
	Children []Element  `xml:",any"`
}

// Parse reads an nvidia-smi -q -x report from r.
// It returns an error if the report is not well-formed XML or its root
// element is not nvidia_smi_log. Elements it does not recognize are kept
// in the Other fields.
func Parse(r io.Reader) (*Log, error) {
	d := xml.NewDecoder(r)
	var dtd string
	for {
		tok, err := d.Token()
		if err == io.EOF {
			return nil, errors.New("parse nvidia-smi report: no nvidia_smi_log element")
		}
		if err != nil {
			return nil, fmt.Errorf("parse nvidia-smi report: %w", err)
		}
		switch t := tok.(type) {
		case xml.Directive:
			if s, ok := doctypeSystem(string(t)); ok {
				dtd = s
			}
		case xml.StartElement:
			if t.Name.Local != "nvidia_smi_log" {
				return nil, fmt.Errorf("parse nvidia-smi report: root element is %s, not nvidia_smi_log", t.Name.Local)
			}
			l := new(Log)
			if err := d.DecodeElement(l, &t); err != nil {
				return nil, fmt.Errorf("parse nvidia-smi report: %w", err)
			}
			l.DTD = dtd
			l.DTDVersion = dtdVersion(dtd)
			return l, nil
		}
	}
}

// doctypeSystem returns the quoted system identifier of a DOCTYPE
// directive, such as nvsmi_device_v12.dtd in
// DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd".
func doctypeSystem(dir string) (string, bool) {
	f := strings.Fields(dir)
	if len(f) < 4 || f[0] != "DOCTYPE" || f[2] != "SYSTEM" {
		return "", false
	}
	s := f[3]
	if len(s) < 2 || (s[0] != '"' && s[0] != '\'') || s[len(s)-1] != s[0] {
		return "", false
	}
	return s[1 : len(s)-1], true
}

// dtdVersion returns NN from nvsmi_device_vNN.dtd, or 0.
func dtdVersion(dtd string) int {
	s, ok := strings.CutPrefix(dtd, "nvsmi_device_v")
	if !ok {
		return 0
	}
	s, ok = strings.CutSuffix(s, ".dtd")
	if !ok {
		return 0
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return 0
	}
	return n
}
