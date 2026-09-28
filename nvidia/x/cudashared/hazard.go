package cudashared

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// ExecutionScope identifies the concurrency level at which memory operations interact.
type ExecutionScope uint8

const (
	ScopeSubwarp ExecutionScope = iota + 1
	ScopeWarp
	ScopeCTA
	ScopeCluster
	ScopeDevice
	ScopeHostDevice
	ScopeMultiGPU
)

// String implements [fmt.Stringer].
func (s ExecutionScope) String() string {
	switch s {
	case ScopeSubwarp:
		return "subwarp"
	case ScopeWarp:
		return "warp"
	case ScopeCTA:
		return "cta"
	case ScopeCluster:
		return "cluster"
	case ScopeDevice:
		return "device"
	case ScopeHostDevice:
		return "hostdevice"
	case ScopeMultiGPU:
		return "multigpu"
	default:
		return fmt.Sprintf("scope(%d)", uint8(s))
	}
}

// MemorySpace identifies the physical or virtual GPU memory space.
type MemorySpace uint8

const (
	MemoryShared MemorySpace = iota + 1
	MemoryDSMEM
	MemoryGlobal
	MemoryUnified
)

// String implements [fmt.Stringer].
func (m MemorySpace) String() string {
	switch m {
	case MemoryShared:
		return "shared"
	case MemoryDSMEM:
		return "dsmem"
	case MemoryGlobal:
		return "global"
	case MemoryUnified:
		return "unified"
	default:
		return fmt.Sprintf("space(%d)", uint8(m))
	}
}

// HazardType classifies memory access conflicts.
type HazardType uint8

const (
	HazardNone HazardType = iota
	HazardReadWrite
	HazardWriteWrite
	HazardWriteRead
)

// String implements [fmt.Stringer].
func (h HazardType) String() string {
	switch h {
	case HazardNone:
		return "none"
	case HazardReadWrite:
		return "read-write"
	case HazardWriteWrite:
		return "write-write"
	case HazardWriteRead:
		return "write-read"
	default:
		return fmt.Sprintf("hazard(%d)", uint8(h))
	}
}

// MemoryAccess describes an individual read or write access. AgentID
// distinguishes independent agents such as kernels, and StreamID records the
// stream that launched a kernel; thread, block and cluster identify execution
// coordinates within the agent.
type MemoryAccess struct {
	Scope     ExecutionScope
	AgentID   int
	StreamID  int
	ThreadID  int
	BlockID   int
	ClusterID int
	Space     MemorySpace
	Offset    uint64
	SizeBytes uint32
	IsWrite   bool
	Stage     int
}

// HazardRecord records an evaluated interaction between two concurrent memory accesses.
type HazardRecord struct {
	AccessA       MemoryAccess
	AccessB       MemoryAccess
	Type          HazardType
	WordConflict  bool
	ByteConflict  bool
	IsRace        bool
	ScopeConflict bool
	Description   string
}

// ByteRangeOverlap reports whether two byte intervals [aOffset, aOffset+aSize) and
// [bOffset, bOffset+bSize) intersect.
func ByteRangeOverlap(aOffset uint64, aSize uint32, bOffset uint64, bSize uint32) bool {
	if aSize == 0 || bSize == 0 {
		return false
	}
	return aOffset < bOffset+uint64(bSize) && bOffset < aOffset+uint64(aSize)
}

// WordOverlap reports whether two memory accesses touch the same aligned 32-bit word.
func WordOverlap(aOffset uint64, aSize uint32, bOffset uint64, bSize uint32) bool {
	if aSize == 0 || bSize == 0 {
		return false
	}
	aStartWord := aOffset &^ 3
	aEndWord := (aOffset + uint64(aSize) - 1) &^ 3
	bStartWord := bOffset &^ 3
	bEndWord := (bOffset + uint64(bSize) - 1) &^ 3
	return aStartWord <= bEndWord && bStartWord <= aEndWord
}

// CheckClusterHardware checks whether the target architecture supports thread block clusters and DSMEM.
// Compute capability 9.0+ (sm_90+) is required. For architectures below sm_90 (e.g. sm_80), it returns
// cudaerrors.ErrHardwarePending rather than simulating device execution.
func CheckClusterHardware(arch string) error {
	arch = strings.ToLower(strings.TrimSpace(arch))
	if arch == "" {
		return fmt.Errorf("check cluster hardware: empty architecture: %w", cudaerrors.ErrHardwarePending)
	}
	raw := strings.TrimPrefix(arch, "sm_")
	// Trim trailing feature suffix like 'a' in sm_90a or sm_100a.
	raw = strings.TrimRight(raw, "abcdefghijklmnopqrstuvwxyz")
	num, err := strconv.Atoi(raw)
	if err != nil {
		return fmt.Errorf("check cluster hardware on %s: unrecognized architecture format: %w", arch, cudaerrors.ErrUnsupported)
	}
	// Compute capability major version: sm_80 -> 8, sm_90 -> 9, sm_100 -> 10, sm_121 -> 12.
	major := num / 10
	if major < 9 {
		return fmt.Errorf("cluster/DSMEM hardware execution on %s (compute capability %d.%d < 9.0): %w",
			arch, major, num%10, cudaerrors.ErrHardwarePending)
	}
	return nil
}

// HazardMatrix tracks concurrent memory operations and classifies hazards across execution scopes.
type HazardMatrix struct {
	TargetArch  string
	ClusterSize int
	accesses    []MemoryAccess
}

// NewHazardMatrix initializes a hazard matrix for a given target GPU architecture.
func NewHazardMatrix(targetArch string) *HazardMatrix {
	return &HazardMatrix{
		TargetArch:  targetArch,
		ClusterSize: 1,
		accesses:    make([]MemoryAccess, 0),
	}
}

// AddAccess registers a memory access into the matrix.
func (m *HazardMatrix) AddAccess(acc MemoryAccess) error {
	if m == nil {
		return fmt.Errorf("nil hazard matrix")
	}
	if acc.SizeBytes == 0 {
		return fmt.Errorf("memory access size must be positive")
	}
	if acc.Scope < ScopeSubwarp || acc.Scope > ScopeMultiGPU {
		return fmt.Errorf("invalid execution scope %d", acc.Scope)
	}
	if acc.Space < MemoryShared || acc.Space > MemoryUnified {
		return fmt.Errorf("invalid memory space %d", acc.Space)
	}
	m.accesses = append(m.accesses, acc)
	return nil
}

// Evaluate inspects all pairs of recorded accesses and classifies race conditions versus false positives.
func (m *HazardMatrix) Evaluate() ([]HazardRecord, error) {
	if m == nil {
		return nil, fmt.Errorf("nil hazard matrix")
	}

	// Check if any access relies on cluster/DSMEM hardware support.
	for _, acc := range m.accesses {
		if acc.Space == MemoryDSMEM || acc.Scope == ScopeCluster {
			if err := CheckClusterHardware(m.TargetArch); err != nil {
				return nil, err
			}
			break
		}
	}

	records := make([]HazardRecord, 0)
	n := len(m.accesses)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a := m.accesses[i]
			b := m.accesses[j]

			// Must interact within compatible memory spaces.
			if a.Space != b.Space {
				// Global and Unified can alias if mapped.
				compatible := (a.Space == MemoryGlobal && b.Space == MemoryUnified) ||
					(a.Space == MemoryUnified && b.Space == MemoryGlobal)
				if !compatible {
					continue
				}
			}

			// Read-Read is never a data race.
			if !a.IsWrite && !b.IsWrite {
				continue
			}

			// Classify hazard type.
			var hz HazardType
			if a.IsWrite && b.IsWrite {
				hz = HazardWriteWrite
			} else if a.IsWrite && !b.IsWrite {
				hz = HazardWriteRead
			} else {
				hz = HazardReadWrite
			}

			wordConflict := WordOverlap(a.Offset, a.SizeBytes, b.Offset, b.SizeBytes)
			byteConflict := ByteRangeOverlap(a.Offset, a.SizeBytes, b.Offset, b.SizeBytes)

			// If they don't even touch the same word, no interaction.
			if !wordConflict && !byteConflict {
				continue
			}

			// Check scope concurrence (is there missing synchronization?).
			scopeConflict := false
			sameThread := a.AgentID == b.AgentID && a.ThreadID == b.ThreadID && a.BlockID == b.BlockID && a.ClusterID == b.ClusterID

			if sameThread {
				// Same thread has sequential execution order within stage.
				scopeConflict = false
			} else if a.Stage == b.Stage {
				// Accesses in the same stage by different threads/scopes without barrier.
				if a.Scope == ScopeSubwarp || b.Scope == ScopeSubwarp {
					// Divergent subwarp execution within same warp.
					scopeConflict = true
				} else if a.Scope == ScopeCTA || b.Scope == ScopeCTA {
					// Threads in same CTA without __syncthreads().
					scopeConflict = true
				} else if a.Scope == ScopeCluster || b.Scope == ScopeCluster || a.Space == MemoryDSMEM {
					// Same-CTA DSMEM aliases are still concurrent thread accesses;
					// cross-CTA DSMEM requires cluster synchronization.
					scopeConflict = a.ClusterID != b.ClusterID || a.BlockID != b.BlockID || a.Space == MemoryDSMEM
				} else if a.Scope == ScopeHostDevice || b.Scope == ScopeHostDevice {
					// Host and device without cudaDeviceSynchronize.
					scopeConflict = true
				} else {
					scopeConflict = true
				}
			}

			// A true data race requires byte-level conflict AND unsynchronized scope conflict.
			// If accesses touch different bytes in the same 32-bit word, byteConflict is false
			// and IsRace is false (eliminating the byte-independence false positive!).
			isRace := byteConflict && scopeConflict

			var desc string
			if isRace {
				desc = fmt.Sprintf("race: concurrent %s hazard on byte range [0x%x, 0x%x) across %s and %s",
					hz, a.Offset, a.Offset+uint64(a.SizeBytes), a.Scope, b.Scope)
			} else if wordConflict && !byteConflict {
				desc = fmt.Sprintf("false-positive filtered: 32-bit word collision on word [0x%x, 0x%x) but disjoint byte ranges ([0x%x..0x%x) vs [0x%x..0x%x))",
					a.Offset&^3, (a.Offset&^3)+4, a.Offset, a.Offset+uint64(a.SizeBytes), b.Offset, b.Offset+uint64(b.SizeBytes))
			} else {
				desc = fmt.Sprintf("synchronized or ordered access: %s", hz)
			}

			records = append(records, HazardRecord{
				AccessA:       a,
				AccessB:       b,
				Type:          hz,
				WordConflict:  wordConflict,
				ByteConflict:  byteConflict,
				IsRace:        isRace,
				ScopeConflict: scopeConflict,
				Description:   desc,
			})
		}
	}

	return records, nil
}

// HasRace reports whether any true data race was identified across all evaluated accesses.
func (m *HazardMatrix) HasRace() (bool, error) {
	records, err := m.Evaluate()
	if err != nil {
		return false, err
	}
	for _, r := range records {
		if r.IsRace {
			return true, nil
		}
	}
	return false, nil
}
