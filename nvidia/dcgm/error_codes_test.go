package dcgm

import (
	"testing"
	"unsafe"
)

func TestErrorCodeNames(t *testing.T) {
	for _, tt := range []struct {
		code ErrorCode
		want string
	}{
		{0, "DCGM_FR_OK"},
		{101, "DCGM_FR_XID_ERROR"},
		{109, "DCGM_FR_SXID_ERROR"},
		{133, "DCGM_FR_ERROR_SENTINEL"},
		{10, "DCGM_FR_CLOCKS_EVENT_THERMAL"},
		{12, "DCGM_FR_CLOCKS_EVENT_POWER"},
		{43, "DCGM_FR_CLOCKS_EVENT_VIOLATION"},
		{134, "ErrorCode(134)"},
		{1<<32 - 1, "ErrorCode(4294967295)"},
	} {
		if got := tt.code.String(); got != tt.want {
			t.Errorf("ErrorCode(%d).String() = %q, want %q", uint32(tt.code), got, tt.want)
		}
	}
	if unsafe.Sizeof(ErrorCode(0)) != 4 {
		t.Fatal("diagnostic code is not uint32-sized")
	}
	// These assignments intentionally require the raw ABI fields to stay uint32.
	var raw uint32
	raw = DiagErrorDetail{}.Code
	raw = DiagErrorDetailV2{}.Code
	raw = DiagErrorV1{}.Code
	if ErrorCode(raw) != 0 {
		t.Fatal("zero diagnostic code changed")
	}
}

func TestErrorCodeAliases(t *testing.T) {
	for _, tt := range []struct {
		alias, current ErrorCode
		want           string
	}{
		{ErrorCodeClockThrottleThermal, ErrorCodeClocksEventThermal, "DCGM_FR_CLOCKS_EVENT_THERMAL"},
		{ErrorCodeClockThrottlePower, ErrorCodeClocksEventPower, "DCGM_FR_CLOCKS_EVENT_POWER"},
		{ErrorCodeThrottlingViolation, ErrorCodeClocksEventViolation, "DCGM_FR_CLOCKS_EVENT_VIOLATION"},
	} {
		if tt.alias != tt.current || tt.alias.String() != tt.want {
			t.Errorf("alias %d differs from current %d or name %q", uint32(tt.alias), uint32(tt.current), tt.want)
		}
	}
}
