package cuptipcsampling

import (
	"errors"
	"unsafe"
)

// readSamplingName reads only a native-provided, live allocation. A byte bound
// limits scanning, not address validity; callers must validate native output
// and retain allocation ownership before calling it. The terminator counts
// toward maxBytes. The returned string is an owned copy.
func readSamplingName(ptr uintptr, maxBytes int) (string, error) {
	if ptr == 0 || maxBytes < 1 || uintptr(maxBytes-1) > ^uintptr(0)-ptr {
		return "", errors.New("cuptipcsampling: invalid native string bounds")
	}
	var data []byte
	p := *(*unsafe.Pointer)(unsafe.Pointer(&ptr))
	for i := 0; i < maxBytes; i++ {
		v := *(*byte)(unsafe.Add(p, i))
		if v == 0 {
			return string(data), nil
		}
		data = append(data, v)
	}
	return "", errors.New("cuptipcsampling: unterminated native string within limit")
}
