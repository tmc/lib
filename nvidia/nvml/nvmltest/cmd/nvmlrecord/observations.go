package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/nvml"
)

func (r *receipt) add(method string, args, value any, err error) {
	v := observation{Method: method, Args: args, Value: value}
	if err != nil {
		v.Error = err.Error()
		v.Unsupported = nvml.IsUnsupported(err)
		var code nvml.Result
		if errors.As(err, &code) {
			v.Code = &code
		}
	}
	r.Calls = append(r.Calls, v)
}

func (r *receipt) unmeasured(method string, args any, reason string) {
	r.Calls = append(r.Calls, observation{Method: method, Args: args, Unmeasured: reason})
}

func cleanupObservation(r *receipt, method string, args any, close func() error) error {
	err := close()
	r.add(method, args, nil, err)
	if err != nil {
		return fmt.Errorf("%s: %w", method, err)
	}
	return nil
}

// An early identity/initialization failure must not erase the selected query
// population. These rows declare absence of measurement, never a zero result.
func completeUnmeasured(r *receipt, cause error) {
	reason := "collection stopped before query"
	if cause != nil {
		reason += ": " + cause.Error()
	}
	counts := make(map[string]int)
	for _, o := range r.Calls {
		counts[o.Method]++
	}
	for _, method := range []string{"MigMode", "GpmSupported", "ProcessesUtilization", "FabricInfo", "FieldValuesFor", "GpuInstanceProfileInfo", "GpuInstanceList", "RemappedRows", "RepairStatus", "CurrentClocksEventReasons", "Samples", "GpmMetrics", "GpuInstanceId", "GpmMigSampleGet"} {
		if counts[method] == 0 {
			r.unmeasured(method, nil, reason)
		}
	}
	for _, method := range []string{"NewGpmSample", "GpmSample.Get"} {
		for i := counts[method]; i < 2; i++ {
			r.unmeasured(method, i+1, reason)
		}
	}
}

type gpmQueries struct {
	migUnavailable string
	allocate       func() (nvml.GpmSample, error)
	get            func(nvml.GpmSample) error
	metrics        func(nvml.GpmSample, nvml.GpmSample) ([]nvml.GpmMetric, error)
	instanceID     func() (uint32, error)
	migGet         func(uint32, nvml.GpmSample) error
	close          func(*nvml.GpmSample) error
	pause          func()
}

func collectGPM(r *receipt, supported bool, supportErr error, mig bool, q gpmQueries) (err error) {
	ids := []nvml.GpmMetricId{nvml.GpmMetricSmUtil, nvml.GpmMetricDramBwUtil}
	reason := "GPM support was not established"
	if supportErr != nil {
		reason += ": " + supportErr.Error()
	}
	var samples [2]nvml.GpmSample
	defer func() {
		for i := len(samples) - 1; i >= 0; i-- {
			if samples[i] != 0 {
				s := &samples[i]
				err = errors.Join(err, cleanupObservation(r, "GpmSample.Close", i+1, func() error { return q.close(s) }))
			}
		}
	}()
	allocated := supported && supportErr == nil
	for i := range samples {
		if !allocated {
			r.unmeasured("NewGpmSample", i+1, reason)
			continue
		}
		s, e := q.allocate()
		samples[i] = s // Nonzero owners remain ours even when allocation reports error.
		r.add("NewGpmSample", i+1, s, e)
		if e != nil || s == 0 {
			allocated = false
			reason = fmt.Sprintf("sample %d allocation unavailable", i+1)
			if e != nil {
				reason += ": " + e.Error()
			}
		}
	}
	getOK := allocated
	for i, s := range samples {
		if !allocated {
			r.unmeasured("GpmSample.Get", i+1, reason)
			continue
		}
		if i == 1 {
			q.pause()
		}
		e := q.get(s)
		r.add("GpmSample.Get", i+1, nil, e)
		if e != nil {
			getOK = false
			reason = fmt.Sprintf("sample %d get failed: %v", i+1, e)
		}
	}
	if getOK {
		values, e := q.metrics(samples[0], samples[1])
		r.add("GpmMetrics", ids, values, e)
	} else {
		r.unmeasured("GpmMetrics", ids, reason)
	}
	if !allocated {
		r.unmeasured("GpuInstanceId", nil, reason)
		r.unmeasured("GpmMigSampleGet", nil, reason)
	} else if !mig {
		reason := q.migUnavailable
		if reason == "" {
			reason = "MIG is disabled or unavailable"
		}
		r.unmeasured("GpuInstanceId", nil, reason)
		r.unmeasured("GpmMigSampleGet", nil, reason)
	} else {
		id, e := q.instanceID()
		r.add("GpuInstanceId", nil, id, e)
		if e != nil {
			r.unmeasured("GpmMigSampleGet", nil, "instance ID query failed: "+e.Error())
		} else {
			r.add("GpmMigSampleGet", id, nil, q.migGet(id, samples[0]))
		}
	}
	return nil
}
