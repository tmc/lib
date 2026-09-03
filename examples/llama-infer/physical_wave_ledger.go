package main

import (
	"errors"
	"slices"
	"sync"
)

type physicalWave struct {
	Sequence       uint64
	Kind           string
	Requests       []string
	Dispatches     uint64
	TierReadBytes  map[string]uint64
	TierWriteBytes map[string]uint64
}

type physicalLedgerSnapshot struct {
	ledger   *physicalWaveLedger
	sequence uint64
}

// physicalWaveLedger records completed device work once, at wave scope. Shared
// dispatches belong to the wave and are never divided among its requests.
type physicalWaveLedger struct {
	mu       sync.Mutex
	sequence uint64
	waves    []physicalWave
}

func (l *physicalWaveLedger) Snapshot() physicalLedgerSnapshot {
	l.mu.Lock()
	defer l.mu.Unlock()
	return physicalLedgerSnapshot{ledger: l, sequence: l.sequence}
}

func (l *physicalWaveLedger) Record(kind string, requests []string, dispatches uint64, read, write map[string]uint64) error {
	if kind == "" || len(requests) == 0 || dispatches == 0 && sumBytes(read)+sumBytes(write) == 0 {
		return errors.New("physical ledger: incomplete wave")
	}
	members := append([]string(nil), requests...)
	slices.Sort(members)
	if slices.Contains(members, "") {
		return errors.New("physical ledger: empty request id")
	}
	for i := 1; i < len(members); i++ {
		if members[i] == members[i-1] {
			return errors.New("physical ledger: duplicate request id")
		}
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.sequence++
	l.waves = append(l.waves, physicalWave{Sequence: l.sequence, Kind: kind, Requests: members, Dispatches: dispatches, TierReadBytes: cloneByteMap(read), TierWriteBytes: cloneByteMap(write)})
	return nil
}

func (l *physicalWaveLedger) Since(before physicalLedgerSnapshot) ([]physicalWave, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if before.ledger != l || before.sequence > l.sequence {
		return nil, errors.New("physical ledger: snapshot is from another generation")
	}
	i := len(l.waves)
	for i > 0 && l.waves[i-1].Sequence > before.sequence {
		i--
	}
	out := make([]physicalWave, len(l.waves)-i)
	for j := range out {
		out[j] = l.waves[i+j]
		out[j].Requests = append([]string(nil), out[j].Requests...)
		out[j].TierReadBytes = cloneByteMap(out[j].TierReadBytes)
		out[j].TierWriteBytes = cloneByteMap(out[j].TierWriteBytes)
	}
	return out, nil
}

func cloneByteMap(in map[string]uint64) map[string]uint64 {
	out := make(map[string]uint64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func sumBytes(m map[string]uint64) uint64 {
	var n uint64
	for _, v := range m {
		n += v
	}
	return n
}
