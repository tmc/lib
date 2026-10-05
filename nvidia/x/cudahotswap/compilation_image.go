package cudahotswap

import "fmt"

// CUBIN returns a copy of the native image compiled by this compilation's
// loader. It rejects a changed or non-native compilation and an absent image.
// The image does not establish that Load succeeded or that a kernel executed.
func (c *Compilation) CUBIN() ([]byte, error) {
	if c == nil || c.loader == nil {
		return nil, fmt.Errorf("compilation image: missing native loader")
	}
	if err := c.loader.validateCompilation(c); err != nil {
		return nil, fmt.Errorf("compilation image: %w", err)
	}
	image := c.native.CUBIN()
	if len(image) == 0 {
		return nil, fmt.Errorf("compilation image: native image is absent")
	}
	return image, nil
}

func (l *NVRTCLoader) validateCompilation(c *Compilation) error {
	if l == nil || c == nil || c.loader != l || c.native == nil || c.PTX != c.native.PTX() || c.Key != c.native.Key() || c.Entry != c.entry {
		return fmt.Errorf("invalid or changed native compilation")
	}
	return nil
}
