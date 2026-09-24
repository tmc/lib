package cudaerrors_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func Example() {
	err := fmt.Errorf("cudacluster: no feature record for sm_98a: %w", cudaerrors.ErrHardwarePending)
	switch {
	case errors.Is(err, cudaerrors.ErrUnsupported):
		fmt.Println("refused")
	case errors.Is(err, cudaerrors.ErrHardwarePending):
		fmt.Println("not yet measured")
	}
	// Output: not yet measured
}
