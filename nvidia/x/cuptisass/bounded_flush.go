package cuptisass

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
)

// FlushLimits bounds the storage used by FlushBounded. All fields must be positive.
// MaxInstances bounds instances per record. MaxBytes includes raw and decoded
// storage, array headers, and a reserved name copy for every record.
// MaxFunctionNameBytes includes the terminating NUL byte.
type FlushLimits struct {
	MaxRecords           uint64
	MaxInstances         uint64
	MaxBytes             uint64
	MaxFunctionNameBytes uint64
}

// FlushBounded retrieves records within limits and clears CUPTI's buffers.
// Native errors or changed counts return no records. CUPTI remains responsible
// for respecting the capacities passed to its flush API.
func (s *Session) FlushBounded(limits FlushLimits) ([]Record, error) {
	return s.flushData(&limits, nativeFlushAPI())
}

type flushAPI struct {
	properties func(*cupti.SassMetricsGetDataPropertiesParams) error
	flush      func(*cupti.SassMetricsFlushDataParams, []cupti.SassMetricsData, [][]cupti.SassMetricsInstanceValue) error
	free       func(uintptr)
	name       func(uintptr, uint64) (string, error)
}

func nativeFlushAPI() flushAPI {
	return flushAPI{
		properties: cupti.SassMetricsGetDataProperties,
		flush: func(p *cupti.SassMetricsFlushDataParams, _ []cupti.SassMetricsData, _ [][]cupti.SassMetricsInstanceValue) error {
			return cupti.SassMetricsFlushData(p)
		},
		free: nativeFree,
		name: boundedFunctionName,
	}
}

func boundedFunctionName(ptr uintptr, limit uint64) (string, error) {
	if ptr == 0 {
		return "", nil
	}
	// Native allocations are owned by CUPTI and stay live until the flush cleanup.
	base := *(*unsafe.Pointer)(unsafe.Pointer(&ptr))
	for i := uint64(0); i < limit; i++ {
		if *(*byte)(unsafe.Add(base, uintptr(i))) == 0 {
			return string(unsafe.Slice((*byte)(base), int(i))), nil
		}
	}
	return "", errors.New("cuptisass: function name exceeds limit")
}

func flushStorage(records, instances uint64, limits FlushLimits) error {
	if limits.MaxRecords == 0 || limits.MaxInstances == 0 || limits.MaxBytes == 0 || limits.MaxFunctionNameBytes == 0 {
		return errors.New("cuptisass: flush limits must be positive")
	}
	maxInt := uint64(^uint(0) >> 1)
	if records > limits.MaxRecords || instances > limits.MaxInstances || records > maxInt || instances > maxInt || limits.MaxFunctionNameBytes > maxInt {
		return errors.New("cuptisass: flush capacity exceeds limit")
	}
	if limits.MaxBytes > maxInt {
		limits.MaxBytes = maxInt
	}
	total := uint64(0)
	add := func(count, size uint64) bool {
		if size != 0 && count > (limits.MaxBytes-total)/size {
			return false
		}
		total += count * size
		return true
	}
	if !add(3, uint64(unsafe.Sizeof([]Record{}))) || !add(records, uint64(unsafe.Sizeof(cupti.SassMetricsData{}))) ||
		!add(records, uint64(unsafe.Sizeof([]cupti.SassMetricsInstanceValue{}))) ||
		!add(records, uint64(unsafe.Sizeof(Record{}))) ||
		!add(records, limits.MaxFunctionNameBytes) {
		return errors.New("cuptisass: flush storage exceeds byte limit")
	}
	// Check multiplication before forming the aggregate instance count.
	if instances != 0 && records > ^uint64(0)/instances {
		return errors.New("cuptisass: flush instance count overflows")
	}
	count := records * instances
	if !add(count, uint64(unsafe.Sizeof(cupti.SassMetricsInstanceValue{}))) || !add(count, uint64(unsafe.Sizeof(InstanceValue{}))) {
		return errors.New("cuptisass: flush storage exceeds byte limit")
	}
	return nil
}

func (s *Session) flushData(limits *FlushLimits, api flushAPI) ([]Record, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.closing {
		return nil, errors.New("cuptisass: session is closed")
	}
	if limits != nil {
		if err := flushStorage(0, 0, *limits); err != nil {
			return nil, err
		}
	}
	var props cupti.SassMetricsGetDataPropertiesParams
	props.StructSize = uint(unsafe.Offsetof(props.NumOfInstances) + unsafe.Sizeof(props.NumOfInstances))
	props.Ctx = driver.Context(s.ctx)
	if err := api.properties(&props); err != nil {
		return nil, fmt.Errorf("cuptisass: get data properties: %w", err)
	}
	n, m := props.NumOfPatchedInstructionRecords, props.NumOfInstances
	if limits != nil {
		if err := flushStorage(uint64(n), uint64(m), *limits); err != nil {
			return nil, err
		}
	}
	if limits != nil && n > 0 && m == 0 {
		return nil, errors.New("cuptisass: nonempty records have no instances")
	}
	if n == 0 || m == 0 {
		return nil, nil
	}
	raw := make([]cupti.SassMetricsData, n)
	values := make([][]cupti.SassMetricsInstanceValue, n)
	var pin runtime.Pinner
	pin.Pin(&raw[0])
	defer pin.Unpin()
	// Scan every allocated slot even when CUPTI returns an error or smaller counts.
	defer func() {
		for i := range raw {
			if raw[i].FunctionName != 0 {
				api.free(raw[i].FunctionName)
			}
		}
	}()
	for i := range raw {
		values[i] = make([]cupti.SassMetricsInstanceValue, m)
		pin.Pin(&values[i][0])
		raw[i].StructSize = uint(unsafe.Sizeof(cupti.SassMetricsData{}))
		raw[i].PInstanceValues = uintptr(unsafe.Pointer(&values[i][0]))
	}
	var p cupti.SassMetricsFlushDataParams
	p.StructSize = uint(unsafe.Offsetof(p.NumOfInstances) + unsafe.Sizeof(p.NumOfInstances))
	p.Ctx = driver.Context(s.ctx)
	p.NumOfPatchedInstructionRecords, p.NumOfInstances = n, m
	p.PMetricsData = uintptr(unsafe.Pointer(&raw[0]))
	if err := api.flush(&p, raw, values); err != nil {
		return nil, fmt.Errorf("cuptisass: flush data: %w", err)
	}
	if p.NumOfPatchedInstructionRecords > n || p.NumOfInstances > m || p.PMetricsData != uintptr(unsafe.Pointer(&raw[0])) {
		return nil, errors.New("cuptisass: flush returned invalid capacities")
	}
	if limits != nil && (p.NumOfPatchedInstructionRecords != n || p.NumOfInstances != m) {
		return nil, errors.New("cuptisass: flush properties changed")
	}
	records := make([]Record, n)
	for i := range records {
		if raw[i].PInstanceValues != uintptr(unsafe.Pointer(&values[i][0])) {
			return nil, errors.New("cuptisass: flush changed instance storage")
		}
		var name string
		var err error
		if limits == nil {
			name = readCString(raw[i].FunctionName)
		} else {
			name, err = api.name(raw[i].FunctionName, limits.MaxFunctionNameBytes)
		}
		if err != nil {
			return nil, err
		}
		records[i] = Record{CubinCRC: raw[i].CubinCrc, FunctionIndex: raw[i].FunctionIndex, FunctionName: name, PCOffset: raw[i].PcOffset, Values: make([]InstanceValue, m)}
		for j := range records[i].Values {
			records[i].Values[j] = InstanceValue{MetricID: values[i][j].MetricId, Value: values[i][j].Value}
		}
	}
	return records, nil
}
