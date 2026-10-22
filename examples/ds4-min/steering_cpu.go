package main

import (
	"errors"
	"math"
)

// projectCPU applies the pinned serial float32 CPU projection to each row.
// The vectors are immutable; scales and history belong to the calling State.
func (v *steeringVectors) projectCPU(x []float32, layer, rows int, scale float32) error {
	if err := steeringScale(scale); err != nil {
		return err
	}
	if scale == 0 || rows == 0 {
		return nil
	}
	if v == nil || x == nil {
		return errors.New("invalid CPU steering projection")
	}
	if layer < 0 || layer >= v.layers || rows < 0 || v.width <= 0 || v.layers > math.MaxInt/v.width || rows > len(x)/v.width || len(x) != rows*v.width || len(v.values) != v.layers*v.width {
		return errors.New("invalid CPU steering projection")
	}
	direction := v.values[layer*v.width : (layer+1)*v.width]
	for row := range rows {
		if err := steeringProject(x[row*v.width:(row+1)*v.width], direction, scale); err != nil {
			return err
		}
	}
	return nil
}
