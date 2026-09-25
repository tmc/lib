package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"strings"
)

// Indents of the snapshot's device fields, processes, and process fields.
const (
	devIndent   = "  "
	fieldIndent = "   "
	procIndent  = "     "
	pfldIndent  = "       "
)

// writeSnapshot writes devs as nvtop's print_snapshot does, byte for byte:
// a JSON array of devices whose values are strings with units, or null.
// With hideProcs it leaves out each device's process list.
func writeSnapshot(w io.Writer, devs []*device, fahrenheit, hideProcs bool) error {
	b := snapshotWriter{bufio.NewWriter(w)}
	b.WriteString("[\n")
	for i, d := range devs {
		fmt.Fprintf(b, "%s{\n", devIndent)
		if d.host != "" {
			b.field(fieldIndent, "host", some(escape(d.host)), false)
		}
		b.field(fieldIndent, "device_name", d.name, false)
		b.field(fieldIndent, "device_architecture", d.arch, false)
		b.field(fieldIndent, "memory_type", opt[string]{}, false) // NVML does not say
		b.field(fieldIndent, "gpu_clock", unit(d.gpuClock, "MHz"), false)
		b.field(fieldIndent, "mem_clock", unit(d.memClock, "MHz"), false)
		temp := opt[string]{}
		if d.temp.ok {
			temp = some(fmt.Sprintf("%dC", d.temp.v))
			if fahrenheit {
				temp = some(fmt.Sprintf("%dF", toFahrenheit(d.temp.v)))
			}
		}
		b.field(fieldIndent, "temp", temp, false)
		fan := opt[string]{}
		if d.fan.ok {
			fan = some(fmt.Sprintf("%d%%", min(d.fan.v, 100)))
		}
		b.field(fieldIndent, "fan_speed", fan, false)
		power := opt[string]{}
		if d.power.ok {
			power = some(fmt.Sprintf("%dW", d.power.v/1000))
		}
		b.field(fieldIndent, "power_draw", power, false)
		b.field(fieldIndent, "gpu_util", pct(d.gpuUtil), false)
		b.field(fieldIndent, "encode", pct(d.encUtil), false)
		b.field(fieldIndent, "decode", pct(d.decUtil), false)
		b.field(fieldIndent, "mem_util", pct(d.memUtil), false)
		b.field(fieldIndent, "mem_total", num(d.memTotal), false)
		b.field(fieldIndent, "mem_used", num(d.memUsed), false)
		// The field after mem_free, if any, writes the comma.
		b.value(fieldIndent, "mem_free", num(d.memFree))
		if hideProcs {
			b.WriteString("\n")
		} else {
			b.processes(d.procs)
		}
		if i == len(devs)-1 {
			fmt.Fprintf(b, "%s}\n", devIndent)
		} else {
			fmt.Fprintf(b, "%s},\n", devIndent)
		}
	}
	b.WriteString("]\n")
	return b.Flush()
}

type snapshotWriter struct{ *bufio.Writer }

// value writes "key": v, with v quoted, or null.
func (b snapshotWriter) value(indent, key string, v opt[string]) {
	fmt.Fprintf(b, "%s\"%s\": ", indent, key)
	if v.ok {
		fmt.Fprintf(b, "\"%s\"", v.v)
	} else {
		b.WriteString("null")
	}
}

// field writes a value on its own line, followed by a comma unless last.
func (b snapshotWriter) field(indent, key string, v opt[string], last bool) {
	b.value(indent, key, v)
	if !last {
		b.WriteString(",")
	}
	b.WriteString("\n")
}

// processes writes a device's "processes" field, preceded by the comma
// that ends the field before it.
func (b snapshotWriter) processes(procs []process) {
	fmt.Fprintf(b, ",\n%s\"processes\" : [\n", fieldIndent)
	for j, p := range procs {
		fmt.Fprintf(b, "%s{\n", procIndent)
		b.field(pfldIndent, "pid", some(fmt.Sprint(p.pid)), false)
		cmdline := p.cmdline
		if cmdline.ok {
			cmdline.v = escape(cmdline.v)
		}
		b.field(pfldIndent, "cmdline", cmdline, false)
		b.field(pfldIndent, "kind", some(p.kind.json()), false)
		b.field(pfldIndent, "user", p.user, false)
		b.field(pfldIndent, "gpu_usage", pct(p.gpuUtil), false)
		b.field(pfldIndent, "gpu_mem_bytes_alloc", num(p.gpuMem), false)
		b.field(pfldIndent, "gpu_mem_usage", pct(p.memPct), false)
		b.field(pfldIndent, "encode", pct(p.encUtil), false)
		b.field(pfldIndent, "decode", pct(p.decUtil), true)
		fmt.Fprintf(b, "%s}", procIndent)
		if j != len(procs)-1 {
			b.WriteString(",")
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(b, "%s]\n", fieldIndent)
}

func pct(v opt[uint32]) opt[string] {
	if !v.ok {
		return opt[string]{}
	}
	return some(fmt.Sprintf("%d%%", v.v))
}

func num(v opt[uint64]) opt[string] {
	if !v.ok {
		return opt[string]{}
	}
	return some(fmt.Sprint(v.v))
}

func unit(v opt[uint32], u string) opt[string] {
	if !v.ok {
		return opt[string]{}
	}
	return some(fmt.Sprintf("%d%s", v.v, u))
}

func (k processKind) json() string {
	switch k {
	case graphic:
		return "graphic"
	case compute:
		return "compute"
	}
	return "graphic & compute"
}

// toFahrenheit converts °C to °F, rounding to nearest even as nvtop's
// nearbyint does.
func toFahrenheit(c uint32) uint32 {
	return uint32(32 + math.RoundToEven(float64(c)*1.8))
}

// escape escapes s for a JSON string the way nvtop does: the named control
// characters, backslash, and double quote. Other bytes pass through.
var escape = strings.NewReplacer(
	"\n", `\n`,
	"\b", `\b`,
	"\f", `\f`,
	"\r", `\r`,
	"\t", `\t`,
	`\`, `\\`,
	`"`, `\"`,
).Replace
