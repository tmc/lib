package xidcatalog

import (
	"errors"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// MaxRecoveryMessageBytes bounds a single XID154 message, including its prefix.
const MaxRecoveryMessageBytes = 4096

// RecoveryText identifies documented text, independently of the numeric token.
// These values are not NVML enums and do not authorize recovery operations.
type RecoveryText uint8

const (
	UnknownRecoveryText RecoveryText = iota
	RecoveryNone
	RecoveryDrainP2P
	RecoveryDrainAndReset
	RecoveryGPUReset
	RecoveryNodeReboot
)

// RecognizeRecoveryText recognizes only the five pinned workbook spellings.
// Unknown future text is not interpreted as None or an instruction to reset.
func RecognizeRecoveryText(text string) (RecoveryText, bool) {
	switch text {
	case "None":
		return RecoveryNone, true
	case "Drain P2P":
		return RecoveryDrainP2P, true
	case "Drain and Reset":
		return RecoveryDrainAndReset, true
	case "GPU Reset Required":
		return RecoveryGPUReset, true
	case "Node Reboot Required":
		return RecoveryNodeReboot, true
	default:
		return UnknownRecoveryText, false
	}
}

// RecoveryToken retains the original hexadecimal and text tokens and the
// parsed uint32 value. Recognized refers to Text only: numeric/text equivalence
// is not inferred from an NVML enum or generalized from a workbook example.
type RecoveryToken struct {
	Hex        string
	Value      uint32
	Text       string
	State      RecoveryText
	Recognized bool
}

// RecoveryTransition preserves one complete XID154 transition. PCI is empty
// for the workbook's bare message form; otherwise its original token is kept.
// Message has no inferred driver/product identity or causal preceding event.
type RecoveryTransition struct {
	XID     uint64
	PCI     string
	Message string
	From    RecoveryToken
	To      RecoveryToken
}

var recoveryPCI = regexp.MustCompile(`^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}$`)
var recoveryMessage = regexp.MustCompile(`^GPU recovery action changed from (0x[0-9a-fA-F]{1,8}) \(([^()\r\n]{1,128})\) to (0x[0-9a-fA-F]{1,8}) \(([^()\r\n]{1,128})\)$`)

// ParseRecoveryTransition parses the documented bare "Xid 154 ..." or NVRM
// "Xid (PCI:...): 154, ..." message. Transport timestamps and unrelated log
// prefixes are not guessed. Missing, truncated, multiline or multiple events
// are refused. Numeric and text tokens remain separate observations; parsing
// is not a health classification, causal inference or remediation policy.
func ParseRecoveryTransition(message string) (RecoveryTransition, error) {
	var result RecoveryTransition
	if len(message) == 0 || len(message) > MaxRecoveryMessageBytes || !utf8.ValidString(message) {
		return result, errors.New("xidcatalog: recovery message extent or encoding refused")
	}
	for _, c := range message {
		if unicode.IsControl(c) {
			return result, errors.New("xidcatalog: recovery message control character")
		}
	}
	rest := message
	if strings.HasPrefix(rest, "Xid 154 ") {
		rest = strings.TrimPrefix(rest, "Xid 154 ")
	} else if strings.HasPrefix(rest, "NVRM: Xid (PCI:") {
		rest = strings.TrimPrefix(rest, "NVRM: Xid (PCI:")
		end := strings.Index(rest, "): 154, ")
		if end < 0 || !recoveryPCI.MatchString(rest[:end]) {
			return result, errors.New("xidcatalog: recovery PCI prefix refused")
		}
		device, err := strconv.ParseUint(rest[end-2:end], 16, 8)
		if err != nil || device > 31 {
			return result, errors.New("xidcatalog: recovery PCI device refused")
		}
		result.PCI = rest[:end]
		rest = rest[end+len("): 154, "):]
	} else {
		return result, errors.New("xidcatalog: XID154 transition prefix required")
	}
	fields := recoveryMessage.FindStringSubmatch(rest)
	if fields == nil {
		return RecoveryTransition{}, errors.New("xidcatalog: complete recovery transition required")
	}
	token := func(hex, text string) (RecoveryToken, error) {
		if strings.TrimSpace(text) != text || text == "" {
			return RecoveryToken{}, errors.New("xidcatalog: recovery text refused")
		}
		n, err := strconv.ParseUint(hex[2:], 16, 32)
		if err != nil {
			return RecoveryToken{}, err
		}
		state, recognized := RecognizeRecoveryText(text)
		return RecoveryToken{hex, uint32(n), text, state, recognized}, nil
	}
	var err error
	result.From, err = token(fields[1], fields[2])
	if err != nil {
		return RecoveryTransition{}, err
	}
	result.To, err = token(fields[3], fields[4])
	if err != nil {
		return RecoveryTransition{}, err
	}
	result.XID, result.Message = 154, message
	return result, nil
}
