package nvidiasmi

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixtureGPU holds values for the first gpu element of a fixture,
// copied from the XML by hand.
type fixtureGPU struct {
	id, product, arch, uuid string
	mig, ecc, temp          Value
	draw, limit             Value // Power.Draw(), Power.Limit
	requested               Value // Power.RequestedLimit
	fbTotal, fbUsed         Value
	gpuIdle                 Value // ClockEventReasons.GPUIdle
	fabricState             Value
	migDevices, processes   int
}

// fixtures lists every file in testdata. Adding a capture means adding
// testdata/NAME.xml and a row here; TestFixturesListed fails otherwise.
var fixtures = []struct {
	file           string
	dtd            string
	dtdVersion     int
	driver, cuda   string
	attached, gpus int
	gpu            fixtureGPU
}{
	{
		file: "a100-sxm4-mig-535.xml", dtd: "nvsmi_device_v12.dtd", dtdVersion: 12,
		driver: "535.54.03", cuda: "12.2", attached: 4, gpus: 1,
		gpu: fixtureGPU{
			id: "00000000:01:00.0", product: "NVIDIA A100-SXM4-80GB", arch: "Ampere",
			uuid: "GPU-513536b6-7d19-9063-b049-1e69664bb298",
			mig:  "Enabled", ecc: "Enabled", temp: "27 C",
			draw: "67.03 W", limit: "500.00 W", requested: "500.00 W",
			fbTotal: "81920 MiB", fbUsed: "50 MiB",
			gpuIdle: "Not Active", fabricState: "N/A",
			migDevices: 4,
		},
	},
	{
		file: "gb200-570.xml", dtd: "nvsmi_device_v12.dtd", dtdVersion: 12,
		driver: "570.148.08", cuda: "12.8", attached: 4, gpus: 4,
		gpu: fixtureGPU{
			id: "00000000:29:00.0", product: "NVIDIA GB200", arch: "Blackwell",
			uuid: "GPU-25e04bdf-6a27-0a2d-f989-a9228a03f373",
			mig:  "Disabled", ecc: "Enabled", temp: "28 C",
			draw: "242.29 W", limit: "1200.00 W", requested: "1200.00 W",
			fbTotal: "189471 MiB", fbUsed: "69580 MiB",
			gpuIdle: "Not Active", fabricState: "Completed",
		},
	},
	{
		file: "gtx-1660-ti-418.xml", dtd: "", dtdVersion: 0,
		driver: "418.43", cuda: "10.1", attached: 1, gpus: 1,
		gpu: fixtureGPU{
			id: "00000000:43:00.0", product: "Graphics Device",
			uuid: "GPU-304a277d-3545-63b8-3a36-dfde3c992989",
			ecc:  "N/A", temp: "40 C",
			draw: "8.93 W", limit: "130.00 W", requested: "130.00 W",
			fbTotal: "5912 MiB", fbUsed: "0 MiB",
			gpuIdle: "Active",
		},
	},
	{
		file: "h100-570.xml", dtd: "nvsmi_device_v12.dtd", dtdVersion: 12,
		driver: "570.86.15", cuda: "12.8", attached: 8, gpus: 8,
		gpu: fixtureGPU{
			id: "00000000:04:00.0", product: "NVIDIA H100 80GB HBM3", arch: "Hopper",
			uuid: "GPU-ab6ab95d-8259-5551-5962-e4a7a710fbae",
			mig:  "Disabled", ecc: "Enabled", temp: "33 C",
			draw: "70.93 W", limit: "700.00 W", requested: "700.00 W",
			fbTotal: "81559 MiB", fbUsed: "1 MiB",
			gpuIdle: "Active", fabricState: "Completed",
		},
	},
	{
		file: "quadro-p4000-440.xml", dtd: "nvsmi_device_v10.dtd", dtdVersion: 10,
		driver: "440.44", cuda: "10.2", attached: 1, gpus: 1,
		gpu: fixtureGPU{
			id: "00000000:08:00.0", product: "Quadro P4000",
			uuid: "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
			ecc:  "N/A", temp: "61 C",
			draw: "30.86 W", limit: "105.00 W", requested: "105.00 W",
			fbTotal: "8119 MiB", fbUsed: "10 MiB",
			gpuIdle: "Not Active",
		},
	},
	{
		file: "rtx-3080-595.xml", dtd: "nvsmi_device_v13.dtd", dtdVersion: 13,
		driver: "595.58.03", cuda: "13.2", attached: 1, gpus: 1,
		gpu: fixtureGPU{
			id: "00000000:0C:00.0", product: "NVIDIA GeForce RTX 3080", arch: "Ampere",
			uuid: "GPU-19d6d965-2acc-f646-00f8-4c76979aabb4",
			mig:  "N/A", ecc: "N/A", temp: "27 C",
			draw: "139.62 W", limit: "320.00 W", requested: "320.00 W",
			fbTotal: "10240 MiB", fbUsed: "9184 MiB",
			gpuIdle: "Not Active", fabricState: "N/A",
		},
	},
	{
		file: "tesla-t4-515.xml", dtd: "nvsmi_device_v11.dtd", dtdVersion: 11,
		driver: "515.105.01", cuda: "11.7", attached: 1, gpus: 1,
		gpu: fixtureGPU{
			id: "00000000:00:1E.0", product: "Tesla T4", arch: "Turing",
			uuid: "GPU-d37e67a5-91dd-3774-a5cb-99096249601a",
			mig:  "N/A", ecc: "Enabled", temp: "40 C",
			draw: "26.78 W", limit: "70.00 W", requested: "70.00 W",
			fbTotal: "15360 MiB", fbUsed: "1032 MiB",
			gpuIdle:   "Not Active",
			processes: 2,
		},
	},
}

func parseFile(t *testing.T, name string) *Log {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	l, err := Parse(f)
	if err != nil {
		t.Fatalf("Parse(%s): %v", name, err)
	}
	return l
}

func TestFixtures(t *testing.T) {
	for _, tt := range fixtures {
		t.Run(tt.file, func(t *testing.T) {
			l := parseFile(t, tt.file)
			if l.DTD != tt.dtd || l.DTDVersion != tt.dtdVersion {
				t.Errorf("DTD = %q, %d; want %q, %d", l.DTD, l.DTDVersion, tt.dtd, tt.dtdVersion)
			}
			if l.DriverVersion != tt.driver || l.CUDAVersion != tt.cuda {
				t.Errorf("versions = %q, %q; want %q, %q", l.DriverVersion, l.CUDAVersion, tt.driver, tt.cuda)
			}
			if l.AttachedGPUs != tt.attached || len(l.GPUs) != tt.gpus {
				t.Fatalf("AttachedGPUs, len(GPUs) = %d, %d; want %d, %d", l.AttachedGPUs, len(l.GPUs), tt.attached, tt.gpus)
			}
			g, w := l.GPUs[0], tt.gpu
			check := func(name string, got, want any) {
				t.Helper()
				if got != want {
					t.Errorf("%s = %q, want %q", name, got, want)
				}
			}
			check("ID", g.ID, w.id)
			check("ProductName", g.ProductName, w.product)
			check("ProductArchitecture", g.ProductArchitecture, w.arch)
			check("UUID", g.UUID, w.uuid)
			check("MIGMode", g.MIGMode, w.mig)
			check("ECCMode", g.ECCMode, w.ecc)
			check("Temperature.GPU", g.Temperature.GPU, w.temp)
			check("Power.Draw()", g.Power.Draw(), w.draw)
			check("Power.Limit", g.Power.Limit, w.limit)
			check("Power.RequestedLimit", g.Power.RequestedLimit, w.requested)
			check("FBMemory.Total", g.FBMemory.Total, w.fbTotal)
			check("FBMemory.Used", g.FBMemory.Used, w.fbUsed)
			check("ClockEventReasons.GPUIdle", g.ClockEventReasons.GPUIdle, w.gpuIdle)
			check("Fabric.State", g.Fabric.State, w.fabricState)
			check("len(MIGDevices)", len(g.MIGDevices), w.migDevices)
			check("len(Processes)", len(g.Processes), w.processes)
		})
	}
}

// TestFixturesListed fails when a testdata file has no row in fixtures,
// so a new capture cannot be added without its expected values.
func TestFixturesListed(t *testing.T) {
	files, err := filepath.Glob("testdata/*.xml")
	if err != nil {
		t.Fatal(err)
	}
	listed := make(map[string]bool)
	for _, tt := range fixtures {
		listed[tt.file] = true
	}
	if len(files) == 0 {
		t.Fatal("no testdata/*.xml files")
	}
	for _, f := range files {
		if !listed[filepath.Base(f)] {
			t.Errorf("%s has no row in fixtures", f)
		}
	}
	if len(files) != len(fixtures) {
		t.Errorf("%d files in testdata, %d rows in fixtures", len(files), len(fixtures))
	}
}

// TestFixturesInvariants checks properties that hold in every report:
// every GPU decodes, the id attribute matches pci_bus_id, renamed
// elements are never left in Other, and Power and ClockEventReasons are
// filled whichever spelling the report used.
func TestFixturesInvariants(t *testing.T) {
	typed := []string{
		"power_readings", "gpu_power_readings", "module_power_readings",
		"clocks_throttle_reasons", "clocks_event_reasons",
		"mig_mode", "mig_devices", "fabric", "temperature", "pci", "processes",
	}
	for _, tt := range fixtures {
		l := parseFile(t, tt.file)
		for i, g := range l.GPUs {
			if g.ID == "" || g.ID != g.PCI.BusID.String() {
				t.Errorf("%s gpu %d: ID %q, PCI.BusID %q", tt.file, i, g.ID, g.PCI.BusID)
			}
			if !g.Power.Limit.Available() {
				t.Errorf("%s gpu %d: Power.Limit = %q, want a reading", tt.file, i, g.Power.Limit)
			}
			if !g.ClockEventReasons.SWPowerCap.Available() {
				t.Errorf("%s gpu %d: ClockEventReasons.SWPowerCap = %q, want a reading", tt.file, i, g.ClockEventReasons.SWPowerCap)
			}
			for _, e := range g.Other {
				for _, name := range typed {
					if e.XMLName.Local == name {
						t.Errorf("%s gpu %d: %s left in Other", tt.file, i, name)
					}
				}
			}
		}
	}
}

func TestPowerLayouts(t *testing.T) {
	tests := []struct {
		name string
		xml  string
		want Power
	}{
		{
			name: "power_readings",
			xml: `<power_readings>
				<power_state>P0</power_state>
				<power_management>Supported</power_management>
				<power_draw>26.78 W</power_draw>
				<power_limit>65.00 W</power_limit>
				<default_power_limit>70.00 W</default_power_limit>
				<enforced_power_limit>60.00 W</enforced_power_limit>
				<min_power_limit>50.00 W</min_power_limit>
				<max_power_limit>75.00 W</max_power_limit>
			</power_readings>`,
			want: Power{State: "P0", Management: "Supported", PowerDraw: "26.78 W",
				Limit: "60.00 W", RequestedLimit: "65.00 W",
				DefaultLimit: "70.00 W", MinLimit: "50.00 W", MaxLimit: "75.00 W"},
		},
		{
			name: "gpu_power_readings",
			xml: `<gpu_power_readings>
				<power_state>P0</power_state>
				<power_draw>70.93 W</power_draw>
				<current_power_limit>650.00 W</current_power_limit>
				<requested_power_limit>600.00 W</requested_power_limit>
				<default_power_limit>700.00 W</default_power_limit>
				<min_power_limit>200.00 W</min_power_limit>
				<max_power_limit>710.00 W</max_power_limit>
			</gpu_power_readings>`,
			want: Power{State: "P0", PowerDraw: "70.93 W",
				Limit: "650.00 W", RequestedLimit: "600.00 W",
				DefaultLimit: "700.00 W", MinLimit: "200.00 W", MaxLimit: "710.00 W"},
		},
		{
			name: "gpu_power_readings average and instant",
			xml: `<gpu_power_readings>
				<average_power_draw>242.29 W</average_power_draw>
				<instant_power_draw>242.67 W</instant_power_draw>
				<current_power_limit>1200.00 W</current_power_limit>
			</gpu_power_readings>`,
			want: Power{AverageDraw: "242.29 W", InstantDraw: "242.67 W", Limit: "1200.00 W"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, err := Parse(strings.NewReader(`<nvidia_smi_log><gpu id="0">` + tt.xml + `</gpu></nvidia_smi_log>`))
			if err != nil {
				t.Fatal(err)
			}
			if got := l.GPUs[0].Power; got != tt.want {
				t.Errorf("Power = %+v\nwant %+v", got, tt.want)
			}
		})
	}
}

func TestPowerDraw(t *testing.T) {
	tests := []struct {
		name string
		p    Power
		want Value
	}{
		{"power_draw", Power{PowerDraw: "1 W", AverageDraw: "2 W", InstantDraw: "3 W"}, "1 W"},
		{"average", Power{AverageDraw: "2 W", InstantDraw: "3 W"}, "2 W"},
		{"average N/A", Power{AverageDraw: "N/A", InstantDraw: "3 W"}, "3 W"},
		{"none available", Power{PowerDraw: "[N/A]", AverageDraw: "N/A"}, "[N/A]"},
		{"empty", Power{}, ""},
	}
	for _, tt := range tests {
		if got := tt.p.Draw(); got != tt.want {
			t.Errorf("%s: Draw() = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestClockEventReasonNames(t *testing.T) {
	for _, prefix := range []string{"clocks_event_reason", "clocks_throttle_reason"} {
		t.Run(prefix, func(t *testing.T) {
			el := prefix + "s"
			xml := `<nvidia_smi_log><gpu id="0"><` + el + `>
				<` + prefix + `_gpu_idle>Active</` + prefix + `_gpu_idle>
				<` + prefix + `_sw_power_cap>Not Active</` + prefix + `_sw_power_cap>
				<` + prefix + `_hw_thermal_slowdown>Active</` + prefix + `_hw_thermal_slowdown>
				<` + prefix + `_future_reason>Active</` + prefix + `_future_reason>
			</` + el + `></gpu></nvidia_smi_log>`
			l, err := Parse(strings.NewReader(xml))
			if err != nil {
				t.Fatal(err)
			}
			r := l.GPUs[0].ClockEventReasons
			if r.GPUIdle != "Active" || r.SWPowerCap != "Not Active" || r.HWThermalSlowdown != "Active" {
				t.Errorf("reasons = %+v", r)
			}
			if r.Other["future_reason"] != "Active" || len(r.Other) != 1 {
				t.Errorf("Other = %v, want map[future_reason:Active]", r.Other)
			}
		})
	}
}

func TestUnavailable(t *testing.T) {
	xml := `<nvidia_smi_log><gpu id="0">
		<fan_speed>[N/A]</fan_speed>
		<serial>N/A</serial>
		<temperature><gpu_temp>[Not Supported]</gpu_temp><memory_temp>Not Supported</memory_temp></temperature>
		<utilization><gpu_util>[Unknown Error]</gpu_util><memory_util>35 %</memory_util></utilization>
		<remapped_rows>N/A</remapped_rows>
	</gpu></nvidia_smi_log>`
	l, err := Parse(strings.NewReader(xml))
	if err != nil {
		t.Fatal(err)
	}
	g := l.GPUs[0]
	for name, v := range map[string]Value{
		"FanSpeed":               g.FanSpeed,
		"Serial":                 g.Serial,
		"Temperature.GPU":        g.Temperature.GPU,
		"Temperature.Memory":     g.Temperature.Memory,
		"Utilization.GPU":        g.Utilization.GPU,
		"RemappedRows.Pending":   g.RemappedRows.Pending,
		"absent Power.PowerDraw": g.Power.PowerDraw,
	} {
		if v.Available() {
			t.Errorf("%s = %q: Available() = true", name, v)
		}
		if n, ok := v.Float(); ok {
			t.Errorf("%s = %q: Float() = %v, true", name, v, n)
		}
	}
	if n, ok := g.Utilization.Memory.Int(); n != 35 || !ok {
		t.Errorf("Utilization.Memory.Int() = %d, %v; want 35, true", n, ok)
	}
}

func TestUnknownElements(t *testing.T) {
	xml := `<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v99.dtd">
<nvidia_smi_log>
	<driver_version>999.1</driver_version>
	<new_top_level>x</new_top_level>
	<gpu id="00000000:01:00.0">
		<product_name>Future GPU</product_name>
		<new_gpu_block attr="a"><inner>42 W</inner></new_gpu_block>
		<temperature><gpu_temp>45 C</gpu_temp><new_sensor>1 C</new_sensor></temperature>
	</gpu>
</nvidia_smi_log>`
	l, err := Parse(strings.NewReader(xml))
	if err != nil {
		t.Fatal(err)
	}
	if l.DTDVersion != 99 {
		t.Errorf("DTDVersion = %d, want 99", l.DTDVersion)
	}
	if len(l.Other) != 1 || l.Other[0].XMLName.Local != "new_top_level" || l.Other[0].Text != "x" {
		t.Errorf("Log.Other = %+v", l.Other)
	}
	g := l.GPUs[0]
	if g.ProductName != "Future GPU" || g.Temperature.GPU != "45 C" {
		t.Errorf("ProductName, Temperature.GPU = %q, %q", g.ProductName, g.Temperature.GPU)
	}
	if len(g.Other) != 1 {
		t.Fatalf("GPU.Other = %+v, want one element", g.Other)
	}
	e := g.Other[0]
	if e.XMLName.Local != "new_gpu_block" || len(e.Attrs) != 1 || len(e.Children) != 1 || e.Children[0].Text != "42 W" {
		t.Errorf("GPU.Other[0] = %+v", e)
	}
}

func TestParseErrors(t *testing.T) {
	tests := []struct {
		name, xml, want string
	}{
		{"empty", "", "no nvidia_smi_log element"},
		{"wrong root", "<other/>", "root element is other"},
		{"truncated", "<nvidia_smi_log><gpu id=\"0\"><product_name>A", "unexpected EOF"},
		{"mismatched", "<nvidia_smi_log><gpu></nvidia_smi_log>", "element <gpu> closed by </nvidia_smi_log>"},
		{"bad attached_gpus", "<nvidia_smi_log><attached_gpus>many</attached_gpus></nvidia_smi_log>", "invalid syntax"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, err := Parse(strings.NewReader(tt.xml))
			if err == nil {
				t.Fatalf("Parse succeeded: %+v", l)
			}
			if !strings.Contains(err.Error(), tt.want) || !strings.HasPrefix(err.Error(), "parse nvidia-smi report: ") {
				t.Errorf("err = %q, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestDTDVersion(t *testing.T) {
	tests := []struct {
		dir  string
		dtd  string
		want int
	}{
		{`DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd"`, "nvsmi_device_v12.dtd", 12},
		{`DOCTYPE nvidia_smi_log SYSTEM 'nvsmi_device_v9.dtd'`, "nvsmi_device_v9.dtd", 9},
		{`DOCTYPE nvidia_smi_log SYSTEM "other.dtd"`, "other.dtd", 0},
		{`DOCTYPE nvidia_smi_log`, "", 0},
	}
	for _, tt := range tests {
		dtd, _ := doctypeSystem(tt.dir)
		if dtd != tt.dtd || dtdVersion(dtd) != tt.want {
			t.Errorf("%s: %q, %d; want %q, %d", tt.dir, dtd, dtdVersion(dtd), tt.dtd, tt.want)
		}
	}
}

func TestMIGDevices(t *testing.T) {
	l := parseFile(t, "a100-sxm4-mig-535.xml")
	// Values from the second mig_device in the fixture.
	d := l.GPUs[0].MIGDevices[1]
	if d.Index != "1" || d.GPUInstanceID != "4" || d.ComputeInstanceID != "0" ||
		d.MultiprocessorCount != "14" || d.FBMemory.Total != "19968 MiB" || d.BAR1Memory.Total != "32767 MiB" {
		t.Errorf("MIGDevices[1] = %+v", d)
	}
}

func TestFabric(t *testing.T) {
	l := parseFile(t, "gb200-570.xml")
	g := l.GPUs[0]
	f := g.Fabric
	if f.Status != "Success" || f.CliqueID != "2" || f.ClusterUUID != "889d2be1-4bbf-69e1-d3f2-97c465ee31aa" ||
		f.Bandwidth != "Full" || f.RouteUnhealthy != "True" {
		t.Errorf("Fabric = %+v", f)
	}
	if g.Platform.SlotNumber != "15" || g.Platform.TrayIndex != "5" || g.Platform.GPUFabricGUID != "0x983400ac35e1fa7e" {
		t.Errorf("Platform = %+v", g.Platform)
	}
	if g.ModulePower.Draw() != "611.27 W" || g.ModulePower.Limit != "2500.00 W" || g.MemoryPower.AverageDraw != "28.96 W" {
		t.Errorf("ModulePower, MemoryPower = %+v, %+v", g.ModulePower, g.MemoryPower)
	}
	if g.ECCErrors.Aggregate.SRAMThresholdExceeded != "No" || g.RemappedRows.HistogramMax != "3968 bank(s)" {
		t.Errorf("ECCErrors.Aggregate, RemappedRows = %+v, %+v", g.ECCErrors.Aggregate, g.RemappedRows)
	}
}

func TestProcesses(t *testing.T) {
	l := parseFile(t, "tesla-t4-515.xml")
	p := l.GPUs[0].Processes
	if len(p) != 2 {
		t.Fatalf("len(Processes) = %d, want 2", len(p))
	}
	want := Process{GPUInstanceID: "N/A", ComputeInstanceID: "N/A", PID: "5762", Type: "C", Name: "python", UsedMemory: "1005 MiB"}
	if p[1] != want {
		t.Errorf("Processes[1] = %+v, want %+v", p[1], want)
	}
	if l.GPUs[0].RetiredPages.SingleBitCount != "0" || l.GPUs[0].RetiredPages.PendingRetirement != "No" {
		t.Errorf("RetiredPages = %+v", l.GPUs[0].RetiredPages)
	}
}

func TestLegacyECC(t *testing.T) {
	l := parseFile(t, "quadro-p4000-440.xml")
	e := l.GPUs[0].ECCErrors
	if e.Volatile.SingleBit.Total != "N/A" || e.Aggregate.DoubleBit.DeviceMemory != "N/A" {
		t.Errorf("ECCErrors = %+v", e)
	}
}
