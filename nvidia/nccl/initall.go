package nccl

import "fmt"

// InitAll creates one communicator for each device ordinal. Devices must be
// nonempty and distinct. The output has one slot per rank.
// On a native error, the returned slice preserves any partially written handles.
// The caller must not use those handles for communication.
func InitAll(devices []int32) ([]Comm, error) {
	return initAll(devices, CommInitAll)
}

func initAll(devices []int32, call func(*Comm, int32, *int32) error) ([]Comm, error) {
	if len(devices) == 0 || uint64(len(devices)) > uint64(1<<31-1) {
		return nil, fmt.Errorf("nccl: invalid device count %d", len(devices))
	}
	seen := make(map[int32]bool, len(devices))
	for _, device := range devices {
		if device < 0 || seen[device] {
			return nil, fmt.Errorf("nccl: invalid or repeated device %d", device)
		}
		seen[device] = true
	}
	comms := make([]Comm, len(devices))
	return comms, call(&comms[0], int32(len(devices)), &devices[0])
}
