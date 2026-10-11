package nvtelemetry

import (
	"math"
	"testing"
)

func TestEnergyNumericAdmission(t *testing.T) {
	for _, value := range []float64{-1, math.NaN(), math.Inf(1), math.Ldexp(1, 64)} {
		p := []SamplePoint{{Timestamp: 0, Value: value, Available: true}, {Timestamp: 1e9, Value: value, Available: true}}
		if got := IntegratePowerToEnergy("test", "GPU", p, 0, 1e9, WindowOptions{}); got.Valid {
			t.Errorf("value %v admitted", value)
		}
	}
	p := []SamplePoint{{Timestamp: 0, Value: 1000, Available: true}, {Timestamp: 1e9, Value: 1000, Available: true}}
	if got := IntegratePowerToEnergy("test", "GPU", p, 0, 1e9, WindowOptions{}); !got.Valid || got.EnergyConsumedMJ != 1000 {
		t.Fatalf("valid power: %+v", got)
	}
	if got := computeCumulativeEnergy("test", "GPU", p, 0, 1e9, WindowOptions{}); got.Valid {
		t.Fatal("float-only counter admitted")
	}
	p[0].IsInt = true
	p[1].IsInt = true
	p[1].IntValue = 1000
	if got := computeCumulativeEnergy("test", "GPU", p, 0, 1e9, WindowOptions{}); !got.Valid || got.EnergyConsumedMJ != 1000 {
		t.Fatalf("integer counter: %+v", got)
	}
}
