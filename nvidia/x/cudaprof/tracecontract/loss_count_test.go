package tracecontract

import "testing"

func TestLossCountZeroTotal(t *testing.T) {
	for _, tt := range []struct {
		name  string
		loss  Loss
		valid bool
	}{
		{"measured zero", Loss{}, true},
		{"impossible zero denominator", Loss{Dropped: 1, Total: 0}, false},
		{"impossible positive denominator", Loss{Dropped: 2, Total: 1}, false},
		{"all dropped", Loss{Dropped: 1, Total: 1}, true},
		{"unknown denominator", Loss{Dropped: 1, Unknown: true}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := validReceiptFixture()
			r.LostRecords = map[string]Loss{"activity": tt.loss}
			_, err := r.JSON()
			if (err == nil) != tt.valid {
				t.Fatalf("loss %+v error %v", tt.loss, err)
			}
		})
	}
}
