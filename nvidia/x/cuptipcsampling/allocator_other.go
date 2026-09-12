//go:build !linux

package cuptipcsampling

import "errors"

func samplingNameAllocator() (func(uintptr), error) {
	return nil, errors.New("cuptipcsampling: native sampling requires Linux")
}
