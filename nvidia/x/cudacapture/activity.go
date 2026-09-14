package cudacapture

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/google/pprof/profile"
)

// ActivityRecord represents a parsed GPU activity record (kernel execution or memory copy)
// from a CUPTI journal or execution capture.
type ActivityRecord struct {
	Process             uint64    `json:"process,omitempty"`
	Subtype             string    `json:"subtype"`
	Name                string    `json:"name,omitempty"`
	Valid               bool      `json:"valid"`
	Start               uint64    `json:"start"`
	End                 uint64    `json:"end"`
	Bytes               uint64    `json:"bytes,omitempty"`
	Queued              uint64    `json:"queued,omitempty"`
	Submitted           uint64    `json:"submitted,omitempty"`
	CopyKind            uint8     `json:"copy_kind,omitempty"`
	Device              uint32    `json:"device"`
	Context             uint32    `json:"context,omitempty"`
	Stream              uint32    `json:"stream,omitempty"`
	Correlation         uint32    `json:"correlation,omitempty"`
	RuntimeCorrelation  uint32    `json:"runtime_correlation,omitempty"`
	GridDim             [3]uint32 `json:"grid,omitempty"`
	BlockDim            [3]uint32 `json:"block,omitempty"`
	RegistersPerThread  uint32    `json:"registers_per_thread,omitempty"`
	DynamicSharedMemory uint32    `json:"dynamic_shared_memory,omitempty"`
}

// UnmarshalJSON unmarshals an ActivityRecord from JSON. It supports array
// dimensions (grid, block, grid_dim, block_dim) and coordinate fields
// (grid_x, grid_y, grid_z, block_x, block_y, block_z), as well as label shorthand
// aliases (registers, smem).
func (a *ActivityRecord) UnmarshalJSON(data []byte) error {
	type Alias ActivityRecord
	var raw struct {
		Alias
		GridDimRaw  *[3]uint32 `json:"grid_dim"`
		BlockDimRaw *[3]uint32 `json:"block_dim"`
		GridX       uint32     `json:"grid_x"`
		GridY       uint32     `json:"grid_y"`
		GridZ       uint32     `json:"grid_z"`
		BlockX      uint32     `json:"block_x"`
		BlockY      uint32     `json:"block_y"`
		BlockZ      uint32     `json:"block_z"`
		Registers   uint32     `json:"registers"`
		Smem        uint32     `json:"smem"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*a = ActivityRecord(raw.Alias)
	if raw.GridDimRaw != nil {
		a.GridDim = *raw.GridDimRaw
	}
	if a.GridDim == [3]uint32{0, 0, 0} && (raw.GridX != 0 || raw.GridY != 0 || raw.GridZ != 0) {
		a.GridDim = [3]uint32{raw.GridX, raw.GridY, raw.GridZ}
	}
	if raw.BlockDimRaw != nil {
		a.BlockDim = *raw.BlockDimRaw
	}
	if a.BlockDim == [3]uint32{0, 0, 0} && (raw.BlockX != 0 || raw.BlockY != 0 || raw.BlockZ != 0) {
		a.BlockDim = [3]uint32{raw.BlockX, raw.BlockY, raw.BlockZ}
	}
	if a.RegistersPerThread == 0 && raw.Registers != 0 {
		a.RegistersPerThread = raw.Registers
	}
	if a.DynamicSharedMemory == 0 && raw.Smem != 0 {
		a.DynamicSharedMemory = raw.Smem
	}
	return nil
}

// MarshalJSON marshals an ActivityRecord, omitting launch geometry and register metrics
// for non-kernel activities or when zero.
func (a ActivityRecord) MarshalJSON() ([]byte, error) {
	type Alias ActivityRecord
	if a.Subtype != "kernel" && a.GridDim == [3]uint32{0, 0, 0} && a.BlockDim == [3]uint32{0, 0, 0} && a.RegistersPerThread == 0 && a.DynamicSharedMemory == 0 {
		var raw struct {
			Alias
			GridDim             any `json:"grid,omitempty"`
			BlockDim            any `json:"block,omitempty"`
			RegistersPerThread  any `json:"registers_per_thread,omitempty"`
			DynamicSharedMemory any `json:"dynamic_shared_memory,omitempty"`
		}
		raw.Alias = Alias(a)
		return json.Marshal(raw)
	}
	return json.Marshal(Alias(a))
}

// AttachKernelLabels adds kernel launch geometry and register allocation labels
// ('grid', 'block', 'registers', 'smem') to the provided label map.
// If labels is nil or rec is not a kernel, AttachKernelLabels does nothing.
func AttachKernelLabels(labels map[string][]string, rec ActivityRecord) {
	if labels == nil || rec.Subtype != "kernel" {
		return
	}
	labels["grid"] = []string{fmt.Sprintf("%d,%d,%d", rec.GridDim[0], rec.GridDim[1], rec.GridDim[2])}
	labels["block"] = []string{fmt.Sprintf("%d,%d,%d", rec.BlockDim[0], rec.BlockDim[1], rec.BlockDim[2])}
	labels["registers"] = []string{strconv.FormatUint(uint64(rec.RegistersPerThread), 10)}
	labels["smem"] = []string{strconv.FormatUint(uint64(rec.DynamicSharedMemory), 10)}
}

// BuildActivityProfile converts a slice of ActivityRecord entries into a pprof Profile.
// Kernel samples are annotated with launch geometry and register labels
// ('grid', 'block', 'registers', 'smem').
func BuildActivityProfile(session string, targetPID uint64, activities []ActivityRecord) *profile.Profile {
	p := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "kernel_time", Unit: "nanoseconds"},
			{Type: "copy_time", Unit: "nanoseconds"},
			{Type: "copy_bytes", Unit: "bytes"},
			{Type: "activity", Unit: "count"},
		},
		// Without this pprof opens the last sample type, "activity", which is
		// a launch count and not a duration. Every consumer would otherwise
		// have to remember -sample_index=0 to see device time.
		DefaultSampleType: "kernel_time",
	}
	functions := make(map[string]*profile.Function)
	var nextFunctionID uint64 = 1
	var nextLocationID uint64 = 1
	for _, activity := range activities {
		if !activity.Valid {
			continue
		}
		if activity.End < activity.Start || activity.End-activity.Start > uint64(^uint64(0)>>1) {
			continue
		}
		value := []int64{0, 0, 0, 1}
		if activity.Subtype == "kernel" {
			value[0] = int64(activity.End - activity.Start)
		} else if activity.Subtype == "memcpy" {
			value[1] = int64(activity.End - activity.Start)
			value[2] = int64(activity.Bytes)
		}
		labels := map[string][]string{
			"subtype":             {activity.Subtype},
			"name":                {activity.Name},
			"session":             {session},
			"pid":                 {strconv.FormatUint(targetPID, 10)},
			"device":              {strconv.FormatUint(uint64(activity.Device), 10)},
			"context":             {strconv.FormatUint(uint64(activity.Context), 10)},
			"stream":              {strconv.FormatUint(uint64(activity.Stream), 10)},
			"correlation":         {strconv.FormatUint(uint64(activity.Correlation), 10)},
			"runtime_correlation": {strconv.FormatUint(uint64(activity.RuntimeCorrelation), 10)},
			"queued":              {strconv.FormatUint(activity.Queued, 10)},
			"submitted":           {strconv.FormatUint(activity.Submitted, 10)},
			"copy_kind":           {strconv.FormatUint(uint64(activity.CopyKind), 10)},
		}
		if activity.Process != 0 {
			labels["pid"] = []string{strconv.FormatUint(activity.Process, 10)}
		}
		if activity.Subtype == "kernel" {
			AttachKernelLabels(labels, activity)
		}
		name := activity.Name
		if activity.Subtype == "memcpy" && name == "" {
			name = "cudaMemcpy"
		}
		fn := functions[name]
		if fn == nil {
			fn = &profile.Function{ID: nextFunctionID, Name: name, SystemName: name}
			nextFunctionID++
			functions[name] = fn
			p.Function = append(p.Function, fn)
		}
		loc := &profile.Location{ID: nextLocationID, Address: 0, Line: []profile.Line{{Function: fn}}}
		nextLocationID++
		p.Location = append(p.Location, loc)
		p.Sample = append(p.Sample, &profile.Sample{Value: value, Label: labels, Location: []*profile.Location{loc}})
	}
	return p
}
