package main

import "errors"

// runMetalStreamingRows stages a layer's complete selected set after all route
// decisions, and releases it only after every row consumer. Failure leaves the
// lease with the caller; it must drain it before releasing scratch or the model.
func runMetalStreamingRows(layers, rows int, front func(int, int) error, stage func(int) error, consume func(int, int) error, release func(int) error) error {
	if layers <= 0 || rows <= 0 || front == nil || stage == nil || consume == nil || release == nil {
		return errors.New("invalid streaming prefill schedule")
	}
	for layer := range layers {
		for row := range rows {
			if err := front(layer, row); err != nil {
				return err
			}
		}
		if err := stage(layer); err != nil {
			return err
		}
		for row := range rows {
			if err := consume(layer, row); err != nil {
				return err
			}
		}
		if err := release(layer); err != nil {
			return err
		}
	}
	return nil
}
