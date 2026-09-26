package sizetrunc

import (
	"errors"
	"math"

	"github.com/tmc/lib/nvidia/cutensor"
	"github.com/tmc/lib/nvidia/driver"
)

func unchecked(h cutensor.Handle, op cutensor.OperationDescriptor, pref cutensor.PlanPreference) (driver.DevicePtr, error) {
	ws, err := h.EstimateWorkspaceSize(op, pref)
	if err != nil {
		return 0, err
	}
	return driver.MemAlloc[byte](int(ws)) // want `ws from EstimateWorkspaceSize is converted to int without a bounds check; a size_t that does not fit is silently truncated`
}

func checked(h cutensor.Handle, op cutensor.OperationDescriptor, pref cutensor.PlanPreference) (driver.DevicePtr, error) {
	ws, err := h.EstimateWorkspaceSize(op, pref)
	if err != nil {
		return 0, err
	}
	if ws > math.MaxInt32 {
		return 0, errors.New("workspace too large")
	}
	return driver.MemAlloc[byte](int(ws))
}

func wide(h cutensor.Handle, op cutensor.OperationDescriptor, pref cutensor.PlanPreference) (driver.DevicePtr, error) {
	ws, err := h.EstimateWorkspaceSize(op, pref)
	if err != nil {
		return 0, err
	}
	return driver.MemAllocBytes(uint(ws))
}
