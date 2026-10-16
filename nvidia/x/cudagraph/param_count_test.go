package cudagraph

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestCountParams(t *testing.T) {
	missing := &driver.SymbolError{Symbol: "cuFuncGetParamCount"}
	unbound := &driver.SymbolError{Symbol: "cuFuncGetParamCount", Err: errors.New("bind failed")}
	invalid := &driver.CallError{Func: "cuFuncGetParamInfo", Result: driver.ErrInvalidValue}
	handle := &driver.CallError{Func: "cuFuncGetParamInfo", Result: driver.ErrInvalidHandle}
	// infoUpTo accepts indices below n and rejects the rest as CUDA does.
	infoUpTo := func(n uint) func(driver.Function, uint) (uint, uint, error) {
		return func(_ driver.Function, i uint) (uint, uint, error) {
			if i >= n {
				return 0, 0, invalid
			}
			return 8 * i, 8, nil
		}
	}
	tests := []struct {
		name    string
		count   func(driver.Function) (uint, error)
		info    func(driver.Function, uint) (uint, uint, error)
		want    uint
		wantErr error
	}{
		{"native count", func(driver.Function) (uint, error) { return 3, nil }, infoUpTo(5), 3, nil},
		{"native error", func(driver.Function) (uint, error) { return 0, handle }, infoUpTo(3), 0, handle},
		{"unbound symbol", func(driver.Function) (uint, error) { return 0, unbound }, infoUpTo(3), 0, unbound},
		{"probe three", func(driver.Function) (uint, error) { return 0, missing }, infoUpTo(3), 3, nil},
		{"probe zero", func(driver.Function) (uint, error) { return 0, missing }, infoUpTo(0), 0, nil},
		{"probe limit", func(driver.Function) (uint, error) { return 0, missing }, infoUpTo(128), 128, nil},
		{"probe over limit", func(driver.Function) (uint, error) { return 0, missing }, infoUpTo(129), 0, errors.New("kernel parameter count exceeds 128")},
		{"probe other error", func(driver.Function) (uint, error) { return 0, missing }, func(driver.Function, uint) (uint, uint, error) { return 0, 0, handle }, 0, handle},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := countParams(1, tt.count, tt.info)
			switch {
			case tt.wantErr == nil && err != nil:
				t.Fatalf("countParams: %v", err)
			case tt.wantErr != nil && err == nil:
				t.Fatalf("countParams = %d, want error %v", got, tt.wantErr)
			case tt.wantErr != nil && !errors.Is(err, tt.wantErr) && err.Error() != tt.wantErr.Error():
				t.Fatalf("countParams error = %v, want %v", err, tt.wantErr)
			}
			if got != tt.want {
				t.Errorf("countParams = %d, want %d", got, tt.want)
			}
		})
	}
}
