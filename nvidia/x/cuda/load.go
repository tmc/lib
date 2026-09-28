package cuda

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

// LoadModule loads image into dev's primary context. The image is a cubin,
// a fatbin, or PTX; the driver picks the best code in a fatbin for dev and
// compiles PTX if nothing else matches. The caller must close the returned
// module.
func LoadModule(dev *Device, image []byte) (*Module, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("cuda: empty module image")
	}
	var raw driver.Module
	err := dev.WithContext(func() error {
		var err error
		raw, err = driver.ModuleLoadData(image)
		return err
	})
	if err != nil {
		if raw != 0 {
			err = errors.Join(err, unloadIn(dev, raw))
		}
		return nil, fmt.Errorf("cuda: load module: %w", err)
	}
	m, err := OwnModule(dev, raw)
	if err != nil {
		return nil, errors.Join(err, unloadIn(dev, raw))
	}
	return m, nil
}

func unloadIn(dev *Device, raw driver.Module) error {
	if err := dev.WithContext(raw.Close); err != nil {
		return fmt.Errorf("cuda: unload module: %w", err)
	}
	return nil
}
