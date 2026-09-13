package cudapipeline

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// ValidatePTX validates all kernel definitions in the given PTX module, reporting
// any pipeline or mbarrier protocol violations.
func ValidatePTX(mod *cudaptx.Module, opts ...Option) ([]ValidationError, error) {
	if mod == nil {
		return nil, fmt.Errorf("cudapipeline: nil module")
	}
	tracker := NewBarrierTracker(opts...)
	return tracker.Validate(mod)
}

// ValidateKernel validates a single PTX kernel definition, reporting any pipeline
// or mbarrier protocol violations.
func ValidateKernel(k *cudaptx.Kernel, opts ...Option) ([]ValidationError, error) {
	if k == nil {
		return nil, fmt.Errorf("cudapipeline: nil kernel")
	}
	tracker := NewBarrierTracker(opts...)
	return tracker.ValidateKernel(k)
}

// Validate audits all kernel definitions in mod and reports any pipeline or mbarrier
// protocol violations.
func (t *BarrierTracker) Validate(mod *cudaptx.Module) ([]ValidationError, error) {
	if mod == nil {
		return nil, fmt.Errorf("cudapipeline: nil module")
	}
	var violations []ValidationError
	for _, k := range mod.Kernels {
		if !k.Body {
			continue
		}
		errs, err := t.ValidateKernel(k)
		if err != nil {
			return nil, err
		}
		violations = append(violations, errs...)
	}
	return violations, nil
}

// ValidateKernel audits a single kernel instruction sequence and reports any
// pipeline or mbarrier protocol violations.
func (t *BarrierTracker) ValidateKernel(k *cudaptx.Kernel) ([]ValidationError, error) {
	if k == nil {
		return nil, fmt.Errorf("cudapipeline: nil kernel")
	}
	t.Reset()
	var violations []ValidationError

	for _, inst := range k.Instructions {
		errs := t.processInstruction(inst)
		violations = append(violations, errs...)
	}

	return violations, nil
}

func (t *BarrierTracker) processInstruction(inst cudaptx.Instruction) []ValidationError {
	var violations []ValidationError
	operands := cudaptx.SplitOperands(inst.Operands)

	// Register tracking
	if strings.HasPrefix(inst.Opcode, "mov.") && len(operands) >= 2 {
		dst := operands[0]
		src := cleanAddress(operands[1])
		if val, err := strconv.ParseInt(src, 0, 64); err == nil {
			t.regs[dst] = val
		} else {
			t.regSyms[dst] = src
		}
		return nil
	}

	if strings.HasPrefix(inst.Opcode, "ld.param.") && len(operands) >= 2 {
		dst := operands[0]
		src := cleanAddress(operands[1])
		t.regSyms[dst] = src
		return nil
	}

	// tcgen05.alloc
	if strings.HasPrefix(inst.Opcode, "tcgen05.alloc") {
		if t.targetArch != "" {
			arch, err := ArchProfileFor(t.targetArch)
			if err == nil && !arch.SupportsTMEM {
				violations = append(violations, ValidationError{
					Line:      inst.Line,
					Opcode:    inst.Opcode,
					Violation: ErrUnsupportedTMEM,
					Message:   fmt.Sprintf("architecture %s does not support tcgen05/TMEM instructions", t.targetArch),
				})
			}
		}
		if len(operands) >= 2 {
			if cols, ok := t.resolveInt(operands[1]); ok {
				if cols < 32 || cols > 512 || cols%32 != 0 {
					violations = append(violations, ValidationError{
						Line:      inst.Line,
						Opcode:    inst.Opcode,
						Violation: ErrInvalidTMEMColumns,
						Actual:    cols,
						Message:   fmt.Sprintf("tcgen05.alloc column count %d must be in [32, 512] and a multiple of 32", cols),
					})
				}
				t.tmemAllocatedCols += int(cols)
				if t.tmemAllocatedCols > BlackwellTMEMColumns {
					violations = append(violations, ValidationError{
						Line:      inst.Line,
						Opcode:    inst.Opcode,
						Violation: ErrTMEMCapacityExceeded,
						Expected:  int64(BlackwellTMEMColumns),
						Actual:    int64(t.tmemAllocatedCols),
						Message:   fmt.Sprintf("total allocated TMEM columns %d exceeds SM limit of %d", t.tmemAllocatedCols, BlackwellTMEMColumns),
					})
				}
			}
		}
		return violations
	}

	// tcgen05.dealloc
	if strings.HasPrefix(inst.Opcode, "tcgen05.dealloc") {
		if len(operands) >= 2 {
			if cols, ok := t.resolveInt(operands[1]); ok {
				t.tmemAllocatedCols -= int(cols)
				if t.tmemAllocatedCols < 0 {
					t.tmemAllocatedCols = 0
				}
			}
		}
		return nil
	}

	// 1. mbarrier.init
	if strings.HasPrefix(inst.Opcode, "mbarrier.init") {
		if len(operands) < 1 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[0]))
		count := int64(1)
		if len(operands) >= 2 {
			if c, ok := t.resolveInt(operands[1]); ok && c > 0 {
				count = c
			}
		}
		if defaultCount, ok := t.barrierArrivals[addr]; ok && len(operands) < 2 {
			count = int64(defaultCount)
		}

		b, exists := t.barriers[addr]
		if exists && b.Initialized && !b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrDuplicateInit,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q already initialized at line %d", addr, b.InitLine),
			})
		}

		b = t.getOrCreateBarrier(addr)
		b.Initialized = true
		b.Invalidated = false
		b.InitLine = inst.Line
		b.ExpectedArrivals = int(count)
		b.PendingArrivals = int(count)
		b.Phase = 0
		b.ExpectedTxBytes = 0
		b.CompletedTxBytes = 0
		b.HasExpectedTx = false
		b.Arrived = false
		b.CopyCount = 0
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventInit,
			Phase:  0,
			Count:  int(count),
		})
		return violations
	}

	// 2. mbarrier.inval
	if strings.HasPrefix(inst.Opcode, "mbarrier.inval") {
		if len(operands) < 1 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[0]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized || b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q invalidated or uninitialized before mbarrier.inval", addr),
			})
			return violations
		}
		b.Invalidated = true
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventInval,
			Phase:  b.Phase,
		})
		return violations
	}

	// 3. mbarrier.arrive.expect_tx
	if strings.HasPrefix(inst.Opcode, "mbarrier.arrive.expect_tx") {
		if len(operands) < 3 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[1]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		txBytes, _ := t.resolveInt(operands[2])
		b.ExpectedTxBytes += txBytes
		b.HasExpectedTx = true
		b.Arrived = true
		if b.PendingArrivals > 0 {
			b.PendingArrivals--
		}
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventExpectTx,
			Phase:  b.Phase,
			Bytes:  txBytes,
			Count:  1,
		})
		return violations
	}

	// 4. mbarrier.expect_tx (standalone)
	if strings.HasPrefix(inst.Opcode, "mbarrier.expect_tx") {
		if len(operands) < 2 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[0]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		txBytes, _ := t.resolveInt(operands[1])
		b.ExpectedTxBytes += txBytes
		b.HasExpectedTx = true
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventExpectTx,
			Phase:  b.Phase,
			Bytes:  txBytes,
		})
		return violations
	}

	// 5. mbarrier.arrive (without expect_tx)
	if strings.HasPrefix(inst.Opcode, "mbarrier.arrive") && !strings.Contains(inst.Opcode, "expect_tx") {
		if len(operands) < 2 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[1]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		count := int64(1)
		if len(operands) >= 3 {
			if c, ok := t.resolveInt(operands[2]); ok && c > 0 {
				count = c
			}
		}
		b.Arrived = true
		b.PendingArrivals -= int(count)
		if b.PendingArrivals < 0 {
			b.PendingArrivals = 0
		}
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventArrive,
			Phase:  b.Phase,
			Count:  int(count),
		})
		return violations
	}

	// 6. mbarrier.complete_tx
	if strings.HasPrefix(inst.Opcode, "mbarrier.complete_tx") {
		if len(operands) < 2 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[0]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		txBytes, _ := t.resolveInt(operands[1])
		b.CompletedTxBytes += txBytes
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventCopy,
			Phase:  b.Phase,
			Bytes:  txBytes,
		})
		return violations
	}

	// 7. cp.async.bulk.tensor
	if strings.HasPrefix(inst.Opcode, "cp.async.bulk.tensor") {
		if !strings.Contains(inst.Opcode, "mbarrier::complete_tx::bytes") || len(operands) < 3 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[2]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		if !b.HasExpectedTx && b.ExpectedTxBytes == 0 {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUnexpectedTx,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("asynchronous tensor copy signaled barrier %q with 0 expected transaction bytes", addr),
			})
		}

		tmap := extractTensorMap(operands[1])
		size, hasSize := t.getTensorSize(tmap)
		if hasSize {
			b.CompletedTxBytes += size
		}
		b.CopyCount++
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventCopy,
			Phase:  b.Phase,
			Bytes:  size,
		})
		return violations
	}

	// 8. cp.async.bulk (non-tensor bulk)
	if strings.HasPrefix(inst.Opcode, "cp.async.bulk.") && !strings.HasPrefix(inst.Opcode, "cp.async.bulk.tensor") {
		if !strings.Contains(inst.Opcode, "mbarrier::complete_tx::bytes") || len(operands) < 4 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[3]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		if !b.HasExpectedTx && b.ExpectedTxBytes == 0 {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUnexpectedTx,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("asynchronous bulk copy signaled barrier %q with 0 expected transaction bytes", addr),
			})
		}

		size, _ := t.resolveInt(operands[2])
		b.CompletedTxBytes += size
		b.CopyCount++
		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventCopy,
			Phase:  b.Phase,
			Bytes:  size,
		})
		return violations
	}

	// 9. mbarrier.try_wait.parity and mbarrier.test_wait.parity
	isParityWait := strings.HasPrefix(inst.Opcode, "mbarrier.try_wait.parity") || strings.HasPrefix(inst.Opcode, "mbarrier.test_wait.parity")
	isGenericWait := (strings.HasPrefix(inst.Opcode, "mbarrier.try_wait") || strings.HasPrefix(inst.Opcode, "mbarrier.test_wait")) && !isParityWait

	if isParityWait {
		if len(operands) < 3 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[1]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		parity, ok := t.resolveInt(operands[2])
		if ok {
			if parity != 0 && parity != 1 {
				violations = append(violations, ValidationError{
					Line:      inst.Line,
					Opcode:    inst.Opcode,
					Violation: ErrInvalidParity,
					Barrier:   addr,
					Phase:     b.Phase,
					Actual:    parity,
					Message:   fmt.Sprintf("invalid phase parity value %d (must be 0 or 1)", parity),
				})
			} else if parity != int64(b.Parity()) {
				violations = append(violations, ValidationError{
					Line:      inst.Line,
					Opcode:    inst.Opcode,
					Violation: ErrPhaseParity,
					Barrier:   addr,
					Phase:     b.Phase,
					Expected:  int64(b.Parity()),
					Actual:    parity,
					Message:   fmt.Sprintf("phase parity mismatch on %q: expected parity %d for phase %d, got %d", addr, b.Parity(), b.Phase, parity),
				})
			}
		}

		if !b.Arrived || b.PendingArrivals > 0 {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrMissingArrive,
				Barrier:   addr,
				Phase:     b.Phase,
				Expected:  int64(b.ExpectedArrivals),
				Actual:    int64(b.ExpectedArrivals - b.PendingArrivals),
				Message:   fmt.Sprintf("barrier %q waited on without prior arrival for phase %d", addr, b.Phase),
			})
		}

		if b.HasExpectedTx {
			if b.CopyCount == 0 && b.ExpectedTxBytes > 0 {
				violations = append(violations, ValidationError{
					Line:      inst.Line,
					Opcode:    inst.Opcode,
					Violation: ErrMismatchedTxBytes,
					Barrier:   addr,
					Phase:     b.Phase,
					Expected:  b.ExpectedTxBytes,
					Actual:    0,
					Message:   fmt.Sprintf("transaction byte count mismatch on %q: expected %d bytes, 0 copy operations issued", addr, b.ExpectedTxBytes),
				})
			} else if b.CompletedTxBytes > 0 && b.ExpectedTxBytes != b.CompletedTxBytes {
				violations = append(violations, ValidationError{
					Line:      inst.Line,
					Opcode:    inst.Opcode,
					Violation: ErrMismatchedTxBytes,
					Barrier:   addr,
					Phase:     b.Phase,
					Expected:  b.ExpectedTxBytes,
					Actual:    b.CompletedTxBytes,
					Message:   fmt.Sprintf("transaction byte count mismatch on %q: expected %d bytes, completed %d bytes", addr, b.ExpectedTxBytes, b.CompletedTxBytes),
				})
			}
		}

		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventWait,
			Phase:  b.Phase,
			Parity: int(parity),
		})
		b.Phase++
		b.PendingArrivals = b.ExpectedArrivals
		b.ExpectedTxBytes = 0
		b.CompletedTxBytes = 0
		b.HasExpectedTx = false
		b.Arrived = false
		b.CopyCount = 0
		return violations
	}

	// 10. mbarrier.try_wait and mbarrier.test_wait (without parity)
	if isGenericWait {
		if len(operands) < 3 {
			return nil
		}
		addr := t.resolveSymbol(cleanAddress(operands[1]))
		b := t.Barrier(addr)
		if b == nil || !b.Initialized {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrUninitializedBarrier,
				Barrier:   addr,
				Message:   fmt.Sprintf("barrier %q used before mbarrier.init", addr),
			})
			return violations
		}
		if b.Invalidated {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrInvalidatedBarrier,
				Barrier:   addr,
				Phase:     b.Phase,
				Message:   fmt.Sprintf("barrier %q was invalidated by mbarrier.inval", addr),
			})
			return violations
		}

		if !b.Arrived || b.PendingArrivals > 0 {
			violations = append(violations, ValidationError{
				Line:      inst.Line,
				Opcode:    inst.Opcode,
				Violation: ErrMissingArrive,
				Barrier:   addr,
				Phase:     b.Phase,
				Expected:  int64(b.ExpectedArrivals),
				Actual:    int64(b.ExpectedArrivals - b.PendingArrivals),
				Message:   fmt.Sprintf("barrier %q waited on without prior arrival for phase %d", addr, b.Phase),
			})
		}

		if b.HasExpectedTx {
			if b.CopyCount == 0 && b.ExpectedTxBytes > 0 {
				violations = append(violations, ValidationError{
					Line:      inst.Line,
					Opcode:    inst.Opcode,
					Violation: ErrMismatchedTxBytes,
					Barrier:   addr,
					Phase:     b.Phase,
					Expected:  b.ExpectedTxBytes,
					Actual:    0,
					Message:   fmt.Sprintf("transaction byte count mismatch on %q: expected %d bytes, 0 copy operations issued", addr, b.ExpectedTxBytes),
				})
			} else if b.CompletedTxBytes > 0 && b.ExpectedTxBytes != b.CompletedTxBytes {
				violations = append(violations, ValidationError{
					Line:      inst.Line,
					Opcode:    inst.Opcode,
					Violation: ErrMismatchedTxBytes,
					Barrier:   addr,
					Phase:     b.Phase,
					Expected:  b.ExpectedTxBytes,
					Actual:    b.CompletedTxBytes,
					Message:   fmt.Sprintf("transaction byte count mismatch on %q: expected %d bytes, completed %d bytes", addr, b.ExpectedTxBytes, b.CompletedTxBytes),
				})
			}
		}

		b.Events = append(b.Events, BarrierEvent{
			Line:   inst.Line,
			Opcode: inst.Opcode,
			Kind:   EventWait,
			Phase:  b.Phase,
		})
		b.Phase++
		b.PendingArrivals = b.ExpectedArrivals
		b.ExpectedTxBytes = 0
		b.CompletedTxBytes = 0
		b.HasExpectedTx = false
		b.Arrived = false
		b.CopyCount = 0
		return violations
	}

	return nil
}

func (t *BarrierTracker) resolveInt(valStr string) (int64, bool) {
	valStr = strings.TrimSpace(valStr)
	if val, err := strconv.ParseInt(valStr, 0, 64); err == nil {
		return val, true
	}
	if v, ok := t.regs[valStr]; ok {
		return v, true
	}
	return 0, false
}

func cleanAddress(op string) string {
	op = strings.TrimSpace(op)
	if strings.HasPrefix(op, "[") && strings.HasSuffix(op, "]") {
		op = strings.TrimSpace(op[1 : len(op)-1])
	}
	return strings.Join(strings.Fields(op), "")
}

func extractTensorMap(op string) string {
	op = strings.TrimSpace(op)
	if strings.HasPrefix(op, "[") && strings.HasSuffix(op, "]") {
		op = strings.TrimSpace(op[1 : len(op)-1])
	}
	if i := strings.Index(op, ","); i >= 0 {
		op = op[:i]
	}
	return strings.TrimSpace(op)
}
