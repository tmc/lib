package tileir

import (
	"fmt"
	"strings"
)

// HazardKind describes the category of a static synchronization hazard.
type HazardKind int

const (
	HazardUnconsumedToken HazardKind = iota
	HazardUnsynchronizedRead
	HazardUnsynchronizedWrite
	HazardDanglingAsyncTransfer
)

// String returns the string representation of the hazard kind.
func (k HazardKind) String() string {
	switch k {
	case HazardUnconsumedToken:
		return "unconsumed_token"
	case HazardUnsynchronizedRead:
		return "unsynchronized_read"
	case HazardUnsynchronizedWrite:
		return "unsynchronized_write"
	case HazardDanglingAsyncTransfer:
		return "dangling_async_transfer"
	default:
		return fmt.Sprintf("hazard(%d)", int(k))
	}
}

// TokenHazard represents a static synchronization hazard in Tile IR.
type TokenHazard struct {
	Op      *Operation
	Kind    HazardKind
	Message string
}

func (h TokenHazard) String() string {
	loc := ""
	if h.Op != nil && h.Op.Line > 0 {
		loc = fmt.Sprintf("line %d: ", h.Op.Line)
	}
	return fmt.Sprintf("tileir: %s%s: %s", loc, h.Kind, h.Message)
}

// ShapeError represents a tile dimension or rank mismatch.
type ShapeError struct {
	Op       *Operation
	Expected string
	Actual   string
	Reason   string
}

func (e ShapeError) String() string {
	loc := ""
	if e.Op != nil && e.Op.Line > 0 {
		loc = fmt.Sprintf("line %d: ", e.Op.Line)
	}
	return fmt.Sprintf("tileir: %sshape error: %s (expected %s, got %s)", loc, e.Reason, e.Expected, e.Actual)
}

// VerifyTokens verifies token lifetimes, DAG dependencies, and memory coherence.
func (f *Function) VerifyTokens() []TokenHazard {
	if f == nil || len(f.Operations) == 0 {
		return nil
	}

	var hazards []TokenHazard

	// 1. Identify all tokens produced and where they are consumed.
	tokenProducers := make(map[string]*Operation)
	tokenConsumers := make(map[string][]*Operation)
	tokenDeps := make(map[string]map[string]bool) // tok -> set of predecessor tokens

	// Track async transfers and their output tokens
	type asyncTransfer struct {
		op       *Operation
		outToken string
	}
	var asyncTransfers []asyncTransfer

	// Track memory writes and reads by view/pointer
	type memAccess struct {
		op      *Operation
		view    string
		token   string
		isWrite bool
	}
	var memAccesses []memAccess

	isTokenType := func(t Type) bool {
		if t == nil {
			return false
		}
		return t.Kind() == KindToken
	}

	for _, op := range f.Operations {
		opCode := strings.ToLower(op.OpCode)
		cleanOp := strings.TrimPrefix(opCode, "cuda_tile.")

		// Determine tokens produced by this operation
		var producedTokens []string
		if cleanOp == "make_token" {
			for _, r := range op.Results {
				tokenProducers[r] = op
				producedTokens = append(producedTokens, r)
				tokenDeps[r] = make(map[string]bool)
			}
		} else if cleanOp == "join_tokens" {
			for _, r := range op.Results {
				tokenProducers[r] = op
				producedTokens = append(producedTokens, r)
				deps := make(map[string]bool)
				for _, inTok := range op.Operands {
					if strings.HasPrefix(inTok, "%") {
						deps[inTok] = true
						for pred := range tokenDeps[inTok] {
							deps[pred] = true
						}
					}
				}
				tokenDeps[r] = deps
			}
		} else {
			// Check if any results are tokens (e.g. from load_view_tko, store_view_tko)
			if fnType, ok := op.TypeVal.(*FunctionType); ok {
				for i, resType := range fnType.Results {
					if isTokenType(resType) && i < len(op.Results) {
						r := op.Results[i]
						tokenProducers[r] = op
						producedTokens = append(producedTokens, r)
						tokenDeps[r] = make(map[string]bool)
					}
				}
			} else if op.TypeVal != nil && isTokenType(op.TypeVal) {
				for _, r := range op.Results {
					tokenProducers[r] = op
					producedTokens = append(producedTokens, r)
					tokenDeps[r] = make(map[string]bool)
				}
			} else {
				// Fallback: result names starting with %tok
				for _, r := range op.Results {
					if strings.HasPrefix(r, "%tok") {
						tokenProducers[r] = op
						producedTokens = append(producedTokens, r)
						tokenDeps[r] = make(map[string]bool)
					}
				}
			}
		}

		// Propagate predecessor dependencies to produced tokens
		for _, outTok := range producedTokens {
			if deps, ok := tokenDeps[outTok]; ok {
				for _, opnd := range op.Operands {
					if _, isToken := tokenProducers[opnd]; isToken {
						deps[opnd] = true
						for pred := range tokenDeps[opnd] {
							deps[pred] = true
						}
					}
				}
			}
		}

		// Record consumed tokens
		for _, opnd := range op.Operands {
			if strings.HasPrefix(opnd, "%") {
				if _, isTok := tokenProducers[opnd]; isTok || strings.HasPrefix(opnd, "%tok") {
					tokenConsumers[opnd] = append(tokenConsumers[opnd], op)
				}
			}
		}

		// Track async transfers (TMA loads / stores)
		if op.IsTMA() || cleanOp == "load_view_tko" || cleanOp == "store_view_tko" || cleanOp == "load_ptr_tko" {
			outTok := ""
			if len(producedTokens) > 0 {
				outTok = producedTokens[0]
			}
			asyncTransfers = append(asyncTransfers, asyncTransfer{op: op, outToken: outTok})

			// Record memory access
			viewName := ""
			tokenName := ""
			isWrite := strings.Contains(cleanOp, "store")

			for _, opnd := range op.Operands {
				if strings.HasPrefix(opnd, "%tok") || (tokenProducers[opnd] != nil) {
					tokenName = opnd
				} else if viewName == "" && !strings.HasPrefix(opnd, "{") && !strings.HasPrefix(opnd, "<") {
					// In store_view_tko %val, %view, %tok, second non-token or named view is view
					if isWrite && len(op.Operands) >= 2 && opnd == op.Operands[0] && !strings.Contains(opnd, "view") {
						continue
					}
					viewName = opnd
				}
			}

			if viewName != "" {
				memAccesses = append(memAccesses, memAccess{
					op:      op,
					view:    viewName,
					token:   tokenName,
					isWrite: isWrite,
				})
			}
		}
	}

	// Rule 1: Token Linearity - every token created must be consumed or joined
	for tok, prodOp := range tokenProducers {
		consumers := tokenConsumers[tok]
		if len(consumers) == 0 {
			hazards = append(hazards, TokenHazard{
				Op:      prodOp,
				Kind:    HazardUnconsumedToken,
				Message: fmt.Sprintf("token %q is created but never consumed or joined", tok),
			})
		}
	}

	// Rule 2: RAW / WAR / WAW Hazards
	for i := 0; i < len(memAccesses); i++ {
		first := memAccesses[i]
		for j := i + 1; j < len(memAccesses); j++ {
			second := memAccesses[j]
			if first.view != second.view {
				continue
			}

			// If both are reads, no hazard
			if !first.isWrite && !second.isWrite {
				continue
			}

			// Check if second access synchronizes on first access
			// Second access's token must transitively depend on first access's completion
			syncs := false
			if second.token != "" {
				if first.op != nil && len(first.op.Results) > 0 {
					for _, res := range first.op.Results {
						if deps, ok := tokenDeps[second.token]; ok && (deps[res] || second.token == res) {
							syncs = true
							break
						}
					}
				}
			}

			if !syncs {
				if first.isWrite && !second.isWrite {
					// Read-After-Write (RAW) hazard
					hazards = append(hazards, TokenHazard{
						Op:      second.op,
						Kind:    HazardUnsynchronizedRead,
						Message: fmt.Sprintf("read of view %q without synchronizing on prior write at line %d", second.view, first.op.Line),
					})
				} else {
					// Write-After-Write or Write-After-Read
					hazards = append(hazards, TokenHazard{
						Op:      second.op,
						Kind:    HazardUnsynchronizedWrite,
						Message: fmt.Sprintf("write to view %q without synchronizing on prior access at line %d", second.view, first.op.Line),
					})
				}
			}
		}
	}

	// Rule 3: Dangling Async Transfers - async operations pending at return
	for _, transfer := range asyncTransfers {
		if transfer.outToken != "" {
			consumers := tokenConsumers[transfer.outToken]
			if len(consumers) == 0 {
				hazards = append(hazards, TokenHazard{
					Op:      transfer.op,
					Kind:    HazardDanglingAsyncTransfer,
					Message: fmt.Sprintf("asynchronous transfer %q at line %d has pending token %q that is not joined before return", transfer.op.OpCode, transfer.op.Line, transfer.outToken),
				})
			}
		}
	}

	return hazards
}

// VerifyShapes validates that tile shapes and dimensions are mathematically compatible.
func (f *Function) VerifyShapes() []ShapeError {
	if f == nil || len(f.Operations) == 0 {
		return nil
	}

	var errors []ShapeError

	// Build SSA value to Type map
	ssaTypes := make(map[string]Type)
	for _, p := range f.Params {
		if p.TypeVal != nil {
			ssaTypes[p.Name] = p.TypeVal
		}
	}

	for _, op := range f.Operations {
		cleanOp := strings.TrimPrefix(strings.ToLower(op.OpCode), "cuda_tile.")

		// Record types of results for downstream lookups
		if len(op.Results) > 0 && op.TypeVal != nil {
			if fnType, ok := op.TypeVal.(*FunctionType); ok {
				if len(fnType.Results) == len(op.Results) {
					for i, r := range op.Results {
						ssaTypes[r] = fnType.Results[i]
					}
				} else if len(fnType.Results) == 1 {
					ssaTypes[op.Results[0]] = fnType.Results[0]
				}
			} else {
				ssaTypes[op.Results[0]] = op.TypeVal
			}
		}

		// Verify Tensor Core MMA operations
		if op.IsMMA() || cleanOp == "mma_f" || cleanOp == "mma_i" || cleanOp == "mmaf_scaled" {
			var typeA, typeB, typeC Type

			if fnType, ok := op.TypeVal.(*FunctionType); ok && len(fnType.Inputs) >= 2 {
				typeA = fnType.Inputs[0]
				typeB = fnType.Inputs[1]
				if len(fnType.Inputs) >= 3 {
					typeC = fnType.Inputs[2]
				}
			} else {
				// Lookup via operands
				if len(op.Operands) >= 2 {
					typeA = ssaTypes[op.Operands[0]]
					typeB = ssaTypes[op.Operands[1]]
					if len(op.Operands) >= 3 {
						typeC = ssaTypes[op.Operands[2]]
					}
				}
			}

			tileA, okA := typeA.(*TileType)
			tileB, okB := typeB.(*TileType)

			if okA && okB {
				if tileA.Rank() < 2 || tileB.Rank() < 2 {
					errors = append(errors, ShapeError{
						Op:       op,
						Expected: "rank-2 tile operands",
						Actual:   fmt.Sprintf("A rank %d, B rank %d", tileA.Rank(), tileB.Rank()),
						Reason:   "MMA operands must be 2D tiles",
					})
				} else {
					kA := tileA.Shape[1]
					kB := tileB.Shape[0]
					mA := tileA.Shape[0]
					nB := tileB.Shape[1]

					if kA != kB {
						errors = append(errors, ShapeError{
							Op:       op,
							Expected: fmt.Sprintf("inner dimension K to match (%d)", kA),
							Actual:   fmt.Sprintf("A inner K=%d, B inner K=%d (A: %v, B: %v)", kA, kB, tileA.Shape, tileB.Shape),
							Reason:   fmt.Sprintf("MMA inner dimension mismatch: %d != %d", kA, kB),
						})
					}

					// Verify accumulator shape if available
					if tileC, okC := typeC.(*TileType); okC && tileC.Rank() == 2 {
						mC := tileC.Shape[0]
						nC := tileC.Shape[1]
						if mC != mA || nC != nB {
							errors = append(errors, ShapeError{
								Op:       op,
								Expected: fmt.Sprintf("accumulator shape [%d, %d]", mA, nB),
								Actual:   fmt.Sprintf("[%d, %d]", mC, nC),
								Reason:   fmt.Sprintf("MMA accumulator shape [%d, %d] does not match result dimensions [%d, %d]", mC, nC, mA, nB),
							})
						}
					}
				}
			}
		}

		// Verify reshape operations
		if cleanOp == "reshape" {
			if len(op.Operands) > 0 {
				srcType := ssaTypes[op.Operands[0]]
				dstType := op.TypeVal
				if fnType, ok := op.TypeVal.(*FunctionType); ok {
					if len(fnType.Inputs) > 0 {
						srcType = fnType.Inputs[0]
					}
					if len(fnType.Results) > 0 {
						dstType = fnType.Results[0]
					}
				}
				srcTile, okSrc := srcType.(*TileType)
				dstTile, okDst := dstType.(*TileType)
				if okSrc && okDst {
					srcNum := srcTile.NumElements()
					dstNum := dstTile.NumElements()
					if srcNum != dstNum {
						errors = append(errors, ShapeError{
							Op:       op,
							Expected: fmt.Sprintf("%d elements", srcNum),
							Actual:   fmt.Sprintf("%d elements", dstNum),
							Reason:   fmt.Sprintf("reshape total elements mismatch: %d != %d", srcNum, dstNum),
						})
					}
				}
			}
		}
	}

	return errors
}

// Verify runs all static verifications (token hazards and shape errors) on the function.
func (f *Function) Verify() ([]TokenHazard, []ShapeError) {
	if f == nil {
		return nil, nil
	}
	return f.VerifyTokens(), f.VerifyShapes()
}

// VerifyTokens runs token hazard verification across all functions in the module.
func (m *Module) VerifyTokens() []TokenHazard {
	if m == nil {
		return nil
	}
	var hazards []TokenHazard
	for _, fn := range m.Functions {
		hazards = append(hazards, fn.VerifyTokens()...)
	}
	return hazards
}

// VerifyShapes runs shape verification across all functions in the module.
func (m *Module) VerifyShapes() []ShapeError {
	if m == nil {
		return nil
	}
	var errors []ShapeError
	for _, fn := range m.Functions {
		errors = append(errors, fn.VerifyShapes()...)
	}
	return errors
}

// Verify checks all functions in the module for token hazards and shape errors.
func (m *Module) Verify() ([]TokenHazard, []ShapeError) {
	if m == nil {
		return nil, nil
	}
	return m.VerifyTokens(), m.VerifyShapes()
}
