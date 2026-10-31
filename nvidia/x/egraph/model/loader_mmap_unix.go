//go:build darwin || linux

package model

import (
	"fmt"
	"os"
	"syscall"
)

func mapFile(name string) ([]byte, func() error, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if st.Size() == 0 {
		return nil, nil, fmt.Errorf("%s is empty", name)
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, fmt.Errorf("mmap %s: %w", name, err)
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
