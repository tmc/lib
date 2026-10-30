package nvidiasmi

import (
	"slices"
	"strconv"
	"testing"
)

func TestValue(t *testing.T) {
	tests := []struct {
		v         Value
		available bool
		f         float64
		fok       bool
		n         int64
		nok       bool
		unit      string
	}{
		{"45 C", true, 45, true, 45, true, "C"},
		{"-5 C", true, -5, true, -5, true, "C"},
		{"70.12 W", true, 70.12, true, 0, false, "W"},
		{"1024 MiB", true, 1024, true, 1024, true, "MiB"},
		{"35 %", true, 35, true, 35, true, "%"},
		{"16x", true, 16, true, 16, true, "x"},
		{"0x100", true, 0, true, 256, true, "x100"},
		{"  27 C\n", true, 27, true, 27, true, "C"},
		{"3968 bank(s)", true, 3968, true, 3968, true, "bank(s)"},
		{"P0", true, 0, false, 0, false, ""},
		{"Enabled", true, 0, false, 0, false, ""},
		{"", false, 0, false, 0, false, ""},
		{"N/A", false, 0, false, 0, false, ""},
		{"[N/A]", false, 0, false, 0, false, ""},
		{"Not Supported", false, 0, false, 0, false, ""},
		{"[Not Supported]", false, 0, false, 0, false, ""},
		{"Unknown Error", false, 0, false, 0, false, ""},
		{"[Insufficient Permissions]", false, 0, false, 0, false, ""},
		{"Requested functionality has been deprecated", false, 0, false, 0, false, ""},
		{"\n\t\t\tNone\n\t\t", true, 0, false, 0, false, ""},
	}
	for _, tt := range tests {
		if got := tt.v.Available(); got != tt.available {
			t.Errorf("%q.Available() = %v, want %v", tt.v, got, tt.available)
		}
		if f, ok := tt.v.Float(); f != tt.f || ok != tt.fok {
			t.Errorf("%q.Float() = %v, %v; want %v, %v", tt.v, f, ok, tt.f, tt.fok)
		}
		if n, ok := tt.v.Int(); n != tt.n || ok != tt.nok {
			t.Errorf("%q.Int() = %v, %v; want %v, %v", tt.v, n, ok, tt.n, tt.nok)
		}
		if u := tt.v.Unit(); u != tt.unit {
			t.Errorf("%q.Unit() = %q, want %q", tt.v, u, tt.unit)
		}
	}
}

func TestValueBytes(t *testing.T) {
	tests := []struct {
		v  Value
		n  int64
		ok bool
	}{
		{"1024 MiB", 1024 << 20, true},
		{"0 MiB", 0, true},
		{"2 GiB", 2 << 30, true},
		{"4000 KB/s", 0, false},
		{"1.5 MiB", 0, false},
		{"N/A", 0, false},
		{"12", 0, false},
		{"9999999999999 TiB", 0, false},
	}
	for _, tt := range tests {
		if n, ok := tt.v.Bytes(); n != tt.n || ok != tt.ok {
			t.Errorf("%q.Bytes() = %d, %v; want %d, %v", tt.v, n, ok, tt.n, tt.ok)
		}
	}
}

func TestValueBool(t *testing.T) {
	tests := []struct {
		v     Value
		b, ok bool
	}{
		{"Enabled", true, true},
		{"disabled", false, true},
		{"Yes", true, true},
		{"No", false, true},
		{"Active", true, true},
		{"Not Active", false, true},
		{"True", true, true},
		{"False", false, true},
		{"N/A", false, false},
		{"Pass-Through", false, false},
	}
	for _, tt := range tests {
		if b, ok := tt.v.Bool(); b != tt.b || ok != tt.ok {
			t.Errorf("%q.Bool() = %v, %v; want %v, %v", tt.v, b, ok, tt.b, tt.ok)
		}
	}
}

func TestValueIntList(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		v    Value
		want []int
		ok   bool
	}{
		{"0", []int{0}, true},
		{Value(strconv.Itoa(maxInt)), []int{maxInt}, true},
		{Value(strconv.Itoa(maxInt-1) + "-" + strconv.Itoa(maxInt)), []int{maxInt - 1, maxInt}, true},
		{" 3-5 ", []int{3, 4, 5}, true},
		{"0,5-7", []int{0, 5, 6, 7}, true},
		{"0,2,4", []int{0, 2, 4}, true},
		{"0-65535", nil, true},
		{"N/A", nil, false},
		{"", nil, false},
		{"0-65536", nil, false},
		{"0-40000,0-40000", nil, false},
		{"5-3", nil, false},
		{"-1", nil, false},
		{"+1", nil, false},
		{"1-+2", nil, false},
		{"1,", nil, false},
		{"1-", nil, false},
		{"a", nil, false},
		{"1 2", nil, false},
	}
	for _, tt := range tests {
		got, ok := tt.v.IntList()
		if ok != tt.ok || (tt.want != nil && !slices.Equal(got, tt.want)) {
			t.Errorf("Value(%q).IntList() = %v, %v, want %v, %v", tt.v, got, ok, tt.want, tt.ok)
		}
		if tt.v == "0-65535" && len(got) != 65536 {
			t.Errorf("Value(%q).IntList() has %d elements", tt.v, len(got))
		}
	}
}
