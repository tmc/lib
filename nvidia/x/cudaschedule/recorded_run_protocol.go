package cudaschedule

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const recordedProtocolVersion = 1
const recordedMessageLimit = 64 << 10

// RecordedProcess identifies a retained Linux process and its executable.
// A numeric PID alone is not an ownership or signal capability.
type RecordedProcess struct {
	PID        int    `json:"pid"`
	Parent     int    `json:"parent"`
	Group      int    `json:"group"`
	Start      uint64 `json:"start"`
	Executable string `json:"executable_sha256"`
}

type recordedProcess = RecordedProcess

func recordedSignalTarget(proc recordedProcess) (int, error) {
	if proc.PID <= 1 || proc.Group <= 1 || proc.Start == 0 {
		return 0, fmt.Errorf("recorded helper: refuse invalid retained process identity")
	}
	return proc.PID, nil
}

// recordedMessage is an individual packet, not a stream of JSON documents.
// Process identity comes from the authenticated peer, not these fields.
type recordedMessage struct {
	Version int    `json:"version"`
	Nonce   string `json:"nonce"`
	Kind    string `json:"kind"`
	Digest  string `json:"digest,omitempty"`
}

// readRecordedMessage checks the spelling and multiplicity of keys before
// decoding. encoding/json alone accepts duplicate and case-folded keys.
func readRecordedMessage(packet []byte, nonce string) (recordedMessage, error) {
	var msg recordedMessage
	if len(packet) == 0 || len(packet) > recordedMessageLimit {
		return msg, fmt.Errorf("recorded helper: packet size out of range")
	}
	d := json.NewDecoder(bytes.NewReader(packet))
	token, err := d.Token()
	if err != nil || token != json.Delim('{') {
		return msg, fmt.Errorf("recorded helper: expected object")
	}
	seen := make(map[string]bool)
	for d.More() {
		keyToken, err := d.Token()
		if err != nil {
			return msg, fmt.Errorf("recorded helper: key: %w", err)
		}
		key, ok := keyToken.(string)
		if !ok || seen[key] {
			return msg, fmt.Errorf("recorded helper: duplicate or invalid key")
		}
		seen[key] = true
		switch key {
		case "version":
			err = d.Decode(&msg.Version)
		case "nonce":
			err = d.Decode(&msg.Nonce)
		case "kind":
			err = d.Decode(&msg.Kind)
		case "digest":
			err = d.Decode(&msg.Digest)
		default:
			return msg, fmt.Errorf("recorded helper: unknown key %q", key)
		}
		if err != nil {
			return msg, fmt.Errorf("recorded helper: %s: %w", key, err)
		}
	}
	if token, err = d.Token(); err != nil || token != json.Delim('}') {
		return msg, fmt.Errorf("recorded helper: unterminated object")
	}
	if _, err = d.Token(); err != io.EOF {
		return msg, fmt.Errorf("recorded helper: trailing data")
	}
	if !seen["version"] || msg.Version != recordedProtocolVersion ||
		!seen["nonce"] || !validRecordedNonce(nonce) || msg.Nonce != nonce ||
		!seen["kind"] || !validRecordedKind(msg.Kind) {
		return msg, fmt.Errorf("recorded helper: invalid protocol identity")
	}
	if seen["digest"] && !validPTXDigest(msg.Digest) {
		return msg, fmt.Errorf("recorded helper: invalid digest")
	}
	return msg, nil
}

func validRecordedNonce(nonce string) bool {
	return len(nonce) == 64 && validPTXDigest(nonce)
}

func validRecordedKind(kind string) bool {
	switch kind {
	case "holder", "held", "ready", "start", "admit", "retained", "exec", "worker-ready", "verified", "permit", "done":
		return true
	}
	return false
}

func writeRecordedMessage(msg recordedMessage) ([]byte, error) {
	data, err := json.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("recorded helper: encode: %w", err)
	}
	if _, err := readRecordedMessage(data, msg.Nonce); err != nil {
		return nil, err
	}
	return data, nil
}

// recordedHandshake describes one attempt. Retention and image verification
// must finish before the worker can receive permission to initialize CUDA.
// The caller advances this only after verifying the packet's peer identity.
type recordedHandshake struct {
	next int
}

func (h *recordedHandshake) accept(kind string) error {
	order := [...]string{"ready", "start", "admit", "retained", "exec", "worker-ready", "verified", "permit", "done"}
	if h.next >= len(order) || kind != order[h.next] {
		return fmt.Errorf("recorded helper: unexpected %q at step %d", kind, h.next)
	}
	h.next++
	return nil
}

func (h recordedHandshake) permitted() bool {
	return h.next >= 8
}

func recordedPacketError(err error) string {
	// Diagnostic text has its own bound even when an underlying OS error embeds
	// caller input. It never serves as a process identity or a verdict.
	text := strings.TrimSpace(err.Error())
	if len(text) > 4096 {
		text = text[:4096]
	}
	return text
}
