package main

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// legacyMTPFrontier binds a derived preparation to an exact request and its
// monotonic lifecycle epoch. Replayable feature generations are not epochs.
type legacyMTPFrontier struct {
	request    *legacyMTPRequestKey
	epoch      uint64
	position   int
	historySHA string
}

// A nonzero-size key has distinct pointer identity for each coordinator.
type legacyMTPRequestKey struct{ marker byte }

func validateLegacyMTPIdentity(model, quant, expectedModel, expectedQuant [32]byte) error {
	if expectedModel == ([32]byte{}) || expectedQuant == ([32]byte{}) || model != expectedModel || quant != expectedQuant {
		return fmt.Errorf("legacy mtp binding: target manifest identity mismatch")
	}
	return nil
}

func newLegacyMTPFrontier(key *legacyMTPRequestKey, epoch uint64, position int, history []int) (legacyMTPFrontier, error) {
	if key == nil || position <= 0 || len(history) != position {
		return legacyMTPFrontier{}, fmt.Errorf("legacy mtp frontier: request and committed history required")
	}
	for _, token := range history {
		if token < 0 || token >= Flash.Vocab {
			return legacyMTPFrontier{}, fmt.Errorf("legacy mtp frontier: token outside target vocabulary")
		}
	}
	hash, err := dsv4.SpeculativeHistorySHA256(history)
	if err != nil {
		return legacyMTPFrontier{}, err
	}
	return legacyMTPFrontier{key, epoch, position, hash}, nil
}

func (f legacyMTPFrontier) validate(current legacyMTPFrontier) error {
	if f.request == nil || f.request != current.request || f.epoch != current.epoch || f.position != current.position || len(f.historySHA) != 64 || f.historySHA != current.historySHA {
		return fmt.Errorf("legacy mtp frontier: stale request preparation")
	}
	return nil
}
