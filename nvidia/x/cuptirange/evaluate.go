package cuptirange

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// Evaluate evaluates one range from a completed, decoded counter data image.
// The image must come from trusted CUPTI collection using this plan's chip and
// configuration; this method is not a parser for untrusted binary data. Values
// are returned in the metric order supplied to NewPlan. It neither collects
// counters nor establishes whether all required replay passes were performed.
func (p *Plan) Evaluate(image []byte, rangeIndex uint) ([]float64, error) {
	if p == nil {
		return nil, errors.New("cuptirange: nil plan")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.host == 0 || p.names == nil {
		return nil, errors.New("cuptirange: closed plan")
	}
	if len(image) == 0 {
		return nil, errors.New("cuptirange: empty counter image")
	}
	data := append([]byte(nil), image...)
	values := make([]float64, len(p.names.pointers))
	for i := range values {
		values[i] = math.NaN()
	}
	var pin runtime.Pinner
	pin.Pin(&data[0])
	pin.Pin(&values[0])
	defer pin.Unpin()
	params := cupti.ProfilerHostEvaluateToGpuValuesParams{
		PHostObject: p.host, PCounterDataImage: uintptr(unsafe.Pointer(&data[0])),
		CounterDataImageSize: uint(len(data)), RangeIndex: rangeIndex,
		PpMetricNames: uintptr(unsafe.Pointer(&p.names.pointers[0])), NumMetrics: uint(len(values)),
		PMetricValues: uintptr(unsafe.Pointer(&values[0])),
	}
	params.StructSize = uint(unsafe.Offsetof(params.PMetricValues) + unsafe.Sizeof(params.PMetricValues))
	if err := p.api.evaluate(&params); err != nil {
		return nil, fmt.Errorf("cuptirange: evaluate: %w", err)
	}
	for i, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("cuptirange: metric %d has no finite value", i)
		}
	}
	return values, nil
}
