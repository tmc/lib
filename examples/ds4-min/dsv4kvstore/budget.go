package dsv4kvstore

import "errors"

const (
	// UpstreamKVCFixedBytes is the fixed KVC header followed by the u32 text
	// length. The text, opaque payload, and optional trailer follow it.
	UpstreamKVCFixedBytes uint64 = upstreamKVCHeaderSize + 4

	// BudgetSafetyPercent is the headroom reserved by the upstream policy
	// when deciding whether a staged file fits the disk budget.
	BudgetSafetyPercent uint64 = 1
)

var ErrBudgetOverflow = errors.New("dsv4kvstore: KVC size overflows uint64")

// KVCSize is the complete serialized size of an upstream KVC entry. It does
// not inspect or allocate the text, payload, or trailer.
func KVCSize(textBytes, payloadBytes, trailerBytes uint64) (uint64, error) {
	total := UpstreamKVCFixedBytes
	for _, n := range []uint64{textBytes, payloadBytes, trailerBytes} {
		if n > ^uint64(0)-total {
			return 0, ErrBudgetOverflow
		}
		total += n
	}
	return total, nil
}

// KVCBudgetRequired returns the budget reservation for a serialized file.
// The one-percent safety margin is rounded up, matching the upstream
// kv_cache_budget_required policy.
func KVCBudgetRequired(fileBytes uint64) (uint64, error) {
	slack := fileBytes / 100
	if fileBytes%100 != 0 {
		slack++
	}
	if slack > ^uint64(0)-fileBytes {
		return 0, ErrBudgetOverflow
	}
	return fileBytes + slack, nil
}

// KVCFitsBudget reports whether text, payload, and trailer bytes fit in a
// budget after the upstream one-percent safety reservation. A zero budget is
// treated as unlimited, like the disabled cache policy.
func KVCFitsBudget(budgetBytes, textBytes, payloadBytes, trailerBytes uint64) (bool, uint64, uint64, error) {
	fileBytes, err := KVCSize(textBytes, payloadBytes, trailerBytes)
	if err != nil {
		return false, 0, 0, err
	}
	required, err := KVCBudgetRequired(fileBytes)
	if err != nil {
		return false, 0, 0, err
	}
	return budgetBytes == 0 || required <= budgetBytes, fileBytes, required, nil
}
