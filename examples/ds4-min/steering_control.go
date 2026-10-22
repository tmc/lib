package main

import "errors"

// changeSteeringFFN publishes a new control only after device synchronization
// and invalidation of any graph that captured the old launch parameters.
func changeSteeringFFN(current, next float32, synchronize, invalidate func() error, publish func(float32)) error {
	if err := steeringScale(next); err != nil {
		return err
	}
	if synchronize == nil || invalidate == nil || publish == nil {
		return errors.New("invalid steering control transition")
	}
	if err := synchronize(); err != nil {
		return err
	}
	if current != next {
		if err := invalidate(); err != nil {
			return err
		}
	}
	publish(next)
	return nil
}
