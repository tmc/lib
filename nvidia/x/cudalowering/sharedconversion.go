package cudalowering

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// This recognizes one observed SM121 textual mechanism, not arbitrary SASS
// semantics or a binary encoding. A conversion requires an exact reaching
// value match between a shared address and the low half of a generic pair.
// Neither an S2R query nor a generic memory opcode suffices on its own.
type sharedConversionChain struct {
	window, lowCopy, highCopy, addressDefinition, sharedAccess, genericAccess uint64
}

func (c sharedConversionChain) String() string {
	return fmt.Sprintf("window=0x%x low-copy=0x%x high-copy=0x%x address-def=0x%x shared-access=0x%x generic-access=0x%x", c.window, c.lowCopy, c.highCopy, c.addressDefinition, c.sharedAccess, c.genericAccess)
}

type sharedConversionScan struct {
	chains         []sharedConversionChain
	sharedAccesses []uint64
	refusals       []string
}
type sharedRegisterValue struct {
	identity                 int
	definition, window, copy uint64
	isWindow, hasCopy        bool
}

func scanSharedConversions(fn *cudasass.Function) sharedConversionScan {
	var result sharedConversionScan
	if fn == nil || len(fn.Instructions) == 0 {
		result.refusals = append(result.refusals, "empty instruction sequence")
		return result
	}
	if len(fn.Instructions) > 65536 || len(fn.Labels) > 4096 || len(fn.Symbols) > 4096 {
		result.refusals = append(result.refusals, "instruction or label budget exceeded")
		return result
	}
	labels := make(map[uint64]bool, len(fn.Labels))
	for _, pc := range fn.Labels {
		labels[pc] = true
	}
	for _, symbol := range fn.Symbols {
		labels[symbol.Address] = true
	}
	values := make(map[int]sharedRegisterValue)
	shared := make(map[int][]uint64)
	identity := 0
	blocked := false
	terminated := false
	refuse := func(pc uint64, reason string) {
		result.refusals = append(result.refusals, fmt.Sprintf("0x%x: %s", pc, reason))
		blocked = true
		clear(values)
		clear(shared)
	}
	fresh := func(pc uint64) sharedRegisterValue {
		identity++
		return sharedRegisterValue{identity: identity, definition: pc}
	}
	var previous uint64
	for index, inst := range fn.Instructions {
		if index == 0 && inst.Address != 0 {
			refuse(inst.Address, "entry instruction does not start at PC zero")
			break
		}
		// The observed SM121 entry block uses 16-byte instructions. A gap
		// may hide an instruction that the text parser did not retain.
		if index > 0 && (previous > ^uint64(0)-16 || inst.Address != previous+16) {
			refuse(inst.Address, "incomplete entry instruction PCs (requires 16-byte steps)")
			break
		}
		previous = inst.Address
		if len(inst.Operands) > 4096 || len(inst.Opcode) > 128 || len(inst.Predicate) > 128 {
			refuse(inst.Address, "instruction text budget exceeded")
			break
		}
		if inst.Predicate != "" {
			refuse(inst.Address, "predicated instruction")
			continue
		}
		op := strings.ToUpper(inst.Opcode)
		if op == "EXIT" || op == "RET" {
			terminated = true
			break
		} // bounded entry block; subsequent padding is unreachable
		if op == "BRA" || strings.HasPrefix(op, "BRA.") || op == "JMP" || op == "CALL" {
			refuse(inst.Address, "control-flow boundary")
			break
		}
		if index > 0 && labels[inst.Address] {
			refuse(inst.Address, "label boundary")
		}
		if blocked {
			continue
		}
		operands, ok := conversionOperands(inst.Operands)
		if !ok {
			refuse(inst.Address, "unsupported operand structure")
			continue
		}
		switch op {
		case "NOP":
			if len(operands) != 0 {
				refuse(inst.Address, "unexpected NOP operands")
			}
		case "S2R":
			if len(operands) != 2 {
				refuse(inst.Address, "unsupported S2R form")
				continue
			}
			dest, ok := conversionRegister(operands[0])
			if !ok {
				refuse(inst.Address, "unsupported S2R destination")
				continue
			}
			v := fresh(inst.Address)
			if operands[1] == "SR_SWINHI" {
				v.window = inst.Address
				v.isWindow = true
			}
			values[dest] = v
		case "MOV":
			if len(operands) != 2 {
				refuse(inst.Address, "unsupported MOV form")
				continue
			}
			dest, ok := conversionRegister(operands[0])
			if !ok {
				refuse(inst.Address, "unsupported MOV destination")
				continue
			}
			if source, ok := conversionRegister(operands[1]); ok {
				v, known := values[source]
				if !known {
					delete(values, dest)
					continue
				}
				v.copy = inst.Address
				v.hasCopy = true
				values[dest] = v
			} else if conversionImmediate(operands[1]) {
				values[dest] = fresh(inst.Address)
			} else {
				refuse(inst.Address, "unsupported MOV source")
			}
		case "STS", "LDS":
			if len(operands) != 2 {
				refuse(inst.Address, "unsupported shared access")
				continue
			}
			if op == "STS" {
				if _, ok := conversionRegister(operands[1]); !ok {
					refuse(inst.Address, "unsupported STS value")
					continue
				}
			}
			address := operands[0]
			if op == "LDS" {
				address = operands[1]
			}
			reg, ok := conversionSharedAddress(address)
			uniform := len(address) > 2 && address[0] == '[' && address[len(address)-1] == ']' && conversionUniformRegister(address[1:len(address)-1])
			if !ok && !uniform {
				refuse(inst.Address, "unsupported shared address")
				continue
			}
			result.sharedAccesses = append(result.sharedAccesses, inst.Address)
			v, known := values[reg]
			if ok && known && !v.isWindow {
				shared[v.identity] = append(shared[v.identity], inst.Address)
			}
			if op == "LDS" {
				dest, ok := conversionRegister(operands[0])
				if !ok {
					refuse(inst.Address, "unsupported LDS destination")
					continue
				}
				values[dest] = fresh(inst.Address)
			}
		case "LD.E", "ST.E":
			if len(operands) != 2 {
				refuse(inst.Address, "unsupported generic access")
				continue
			}
			address := operands[0]
			if op == "LD.E" {
				address = operands[1]
			}
			reg, ok := conversionGenericAddress(address)
			if !ok {
				refuse(inst.Address, "unsupported generic address pair")
				continue
			}
			low, lok := values[reg]
			high, hok := values[reg+1]
			witness := shared[low.identity]
			if lok && hok && !low.isWindow && high.isWindow && low.hasCopy && high.hasCopy && len(witness) == 1 {
				result.chains = append(result.chains, sharedConversionChain{window: high.window, lowCopy: low.copy, highCopy: high.copy, addressDefinition: low.definition, sharedAccess: witness[0], genericAccess: inst.Address})
			} else {
				result.refusals = append(result.refusals, fmt.Sprintf("0x%x: generic pair lacks unique supported shared-conversion provenance", inst.Address))
			}
			if op == "LD.E" {
				dest, ok := conversionRegister(operands[0])
				if !ok {
					refuse(inst.Address, "unsupported LD destination")
					continue
				}
				values[dest] = fresh(inst.Address)
			}
		// These observed writes establish a new opaque reaching identity. No
		// address arithmetic or constant-bank semantics are inferred from them.
		case "LEA", "LDC", "IADD", "LDG.E":
			arity := map[string]int{"LEA": 4, "LDC": 2, "IADD": 3, "LDG.E": 2}[op]
			if len(operands) != arity {
				refuse(inst.Address, "unsupported known write operands")
				continue
			}
			dest, ok := conversionRegister(operands[0])
			if !ok {
				refuse(inst.Address, "unsupported known write destination")
				continue
			}
			values[dest] = fresh(inst.Address)
		case "IADD3":
			// Only the observed scalar form with discarded predicate outputs
			// is supported. Its result has no inferred address provenance.
			if len(operands) != 6 || operands[1] != "PT" || operands[2] != "PT" || operands[5] != "RZ" || !conversionImmediate(operands[4]) {
				refuse(inst.Address, "unsupported IADD3 operands")
				continue
			}
			dest, destOK := conversionRegister(operands[0])
			source, sourceOK := conversionRegister(operands[3])
			if !destOK || !sourceOK || dest == 255 || source == 255 {
				refuse(inst.Address, "unsupported IADD3 register")
				continue
			}
			values[dest] = fresh(inst.Address)
		case "LDC.64":
			if len(operands) != 2 {
				refuse(inst.Address, "unsupported pair write")
				continue
			}
			dest, ok := conversionRegister(operands[0])
			if !ok || dest == 255 {
				refuse(inst.Address, "unsupported pair destination")
				continue
			}
			values[dest] = fresh(inst.Address)
			values[dest+1] = fresh(inst.Address)
		case "LDCU.64", "UMOV", "ULEA", "S2UR":
			// Uniform-register operations cannot alias the tracked R register
			// file. Only explicit observed forms are accepted here.
			arity := 2
			if op == "ULEA" {
				arity = 4
			}
			if len(operands) != arity || !conversionUniformRegister(operands[0]) {
				refuse(inst.Address, "unsupported uniform write")
			}
		case "STG.E":
			if len(operands) != 2 {
				refuse(inst.Address, "unsupported global store")
			}
		default:
			refuse(inst.Address, "unknown instruction "+op)
		}
	}
	if !terminated && len(result.refusals) == 0 {
		result.refusals = append(result.refusals, "entry block has no supported terminator")
	}
	return result
}

func conversionRegister(s string) (int, bool) {
	if len(s) < 2 || s[0] != 'R' {
		return 0, false
	}
	n, e := strconv.Atoi(s[1:])
	return n, e == nil && n >= 0 && n < 256 && s == fmt.Sprintf("R%d", n)
}
func conversionUniformRegister(s string) bool {
	if !strings.HasPrefix(s, "UR") {
		return false
	}
	_, ok := conversionRegister(s[1:])
	return ok
}
func conversionImmediate(s string) bool { _, e := strconv.ParseUint(s, 0, 32); return e == nil }
func conversionSharedAddress(s string) (int, bool) {
	if len(s) < 3 || s[0] != '[' || s[len(s)-1] != ']' {
		return 0, false
	}
	return conversionRegister(strings.TrimSpace(s[1 : len(s)-1]))
}
func conversionGenericAddress(s string) (int, bool) {
	// The calibrated textual mechanism uses a uniform descriptor and an
	// exact consecutive .64 general-register address pair, with no offsets.
	if !strings.HasPrefix(s, "desc[") {
		return 0, false
	}
	end := strings.IndexByte(s[5:], ']')
	if end < 0 {
		return 0, false
	}
	end += 5
	if !conversionUniformRegister(s[5:end]) {
		return 0, false
	}
	address := s[end+1:]
	if !strings.HasPrefix(address, "[R") || !strings.HasSuffix(address, ".64]") {
		return 0, false
	}
	r, ok := conversionRegister(address[1 : len(address)-4])
	return r, ok && r < 255
}
func conversionOperands(s string) ([]string, bool) {
	if strings.TrimSpace(s) == "" {
		return nil, true
	}
	var result []string
	start, depth := 0, 0
	for i, c := range s {
		switch c {
		case '[':
			depth++
		case ']':
			depth--
			if depth < 0 {
				return nil, false
			}
		case ',':
			if depth == 0 {
				result = append(result, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	if depth != 0 {
		return nil, false
	}
	result = append(result, strings.TrimSpace(s[start:]))
	for _, s := range result {
		if s == "" {
			return nil, false
		}
	}
	return result, true
}
