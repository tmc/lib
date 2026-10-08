package imex

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
)

// Profile identifies an explicitly supported input format. It is selected by
// the caller, not inferred from daemon versions inside a snapshot.
type Profile string

// DCGMSpec58030 is the specification fixture retained by NVIDIA/DCGM revision
// 64df9f894541e426e416131a9820cae97aa4dd81. Its nodes report version 580.30.
// This does not declare all output from an IMEX release compatible.
const DCGMSpec58030 Profile = "dcgm-64df9f89-spec-580.30"

// UnsupportedProfileError reports a profile the parser does not implement.
// Profile retains exactly the caller selection; it is not a daemon version.
type UnsupportedProfileError struct{ Profile Profile }

func (e *UnsupportedProfileError) Error() string {
	return fmt.Sprintf("imex: unsupported profile %q", e.Profile)
}

// Limits bounds one finite snapshot. Zero fields use 4 MiB, 1024 nodes,
// 65536 connections and JSON depth 32. Maxima are 16 MiB, 4096 nodes,
// 262144 connections and depth 64. Connections is the total across all nodes.
type Limits struct{ Bytes, Nodes, Connections, Depth int }

// Snapshot retains producer-reported domain state and timestamp. Timestamp is
// unparsed: the format does not supply a timezone. A status is not a verdict
// made by this package. Nodes is the supplied map, not verified membership.
// Raw includes every input byte; Unknown retains unrecognized top-level fields.
type Snapshot struct {
	Profile   Profile
	Status    string
	Timestamp string
	Nodes     map[string]Node
	Unknown   map[string]json.RawMessage
	Raw       []byte
}

// Node retains one reported daemon. Version and Hostname are nil if absent;
// a nonnil pointer preserves the reported string, including an empty string.
// An empty string supplies no product version or hostname identity. Changed
// is a producer flag. Missing targets are not manufactured as nodes.
type Node struct {
	Status      string
	Host        string
	Changed     bool
	Version     *string
	Hostname    *string
	Connections map[string]Connection
	Unknown     map[string]json.RawMessage
}

// Connection is directional, from the containing node to its map key. Status
// and Changed are producer observations, not a symmetric or healthy-link claim.
type Connection struct {
	Host    string
	Status  string
	Changed bool
	Unknown map[string]json.RawMessage
}

// Read parses a finite nvidia-imex-ctl -N -j snapshot for the caller-selected
// profile. It invokes no command or service. Empty, malformed, duplicate-key,
// trailing and over-limit input is refused. Unknown fields are retained, not
// interpreted. Read blocks according to r; the caller owns cancellation.
func Read(r io.Reader, profile Profile, limits Limits) (Snapshot, error) {
	var snapshot Snapshot
	if profile != DCGMSpec58030 {
		return snapshot, &UnsupportedProfileError{Profile: profile}
	}
	l, err := limits.checked()
	if err != nil {
		return snapshot, err
	}
	if r == nil {
		return snapshot, errors.New("imex: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(l.Bytes)+1))
	if err != nil {
		return snapshot, fmt.Errorf("imex read: %w", err)
	}
	if len(b) > l.Bytes {
		return snapshot, errors.New("imex: byte limit exceeded")
	}
	if err := unique(b, l); err != nil {
		return snapshot, err
	}
	obj, err := object(b, "nodes", "timestamp", "status")
	if err != nil {
		return snapshot, err
	}
	if err := text(obj["status"], &snapshot.Status, 256, false); err != nil {
		return Snapshot{}, err
	}
	if err := text(obj["timestamp"], &snapshot.Timestamp, 128, false); err != nil {
		return Snapshot{}, err
	}
	nodes, err := object(obj["nodes"])
	if err != nil {
		return Snapshot{}, err
	}
	if len(nodes) > l.Nodes {
		return Snapshot{}, errors.New("imex: node limit exceeded")
	}
	snapshot.Nodes = make(map[string]Node, len(nodes))
	total := 0
	for id, raw := range nodes {
		if err := nodeID(id); err != nil {
			return Snapshot{}, err
		}
		fields, err := object(raw, "status", "host", "connections", "changed")
		if err != nil {
			return Snapshot{}, fmt.Errorf("imex node %s: %w", id, err)
		}
		var node Node
		if err = text(fields["status"], &node.Status, 256, false); err != nil {
			return Snapshot{}, err
		}
		if err = text(fields["host"], &node.Host, 1024, false); err != nil {
			return Snapshot{}, err
		}
		if err = boolean(fields["changed"], &node.Changed); err != nil {
			return Snapshot{}, err
		}
		for key, target := range map[string]**string{"version": &node.Version, "hostName": &node.Hostname} {
			if value, ok := fields[key]; ok {
				var v string
				if err = text(value, &v, 1024, true); err != nil {
					return Snapshot{}, err
				}
				*target = &v
			}
		}
		connections, err := object(fields["connections"])
		if err != nil {
			return Snapshot{}, err
		}
		if len(connections) > l.Connections-total {
			return Snapshot{}, errors.New("imex: connection limit exceeded")
		}
		total += len(connections)
		node.Connections = make(map[string]Connection, len(connections))
		for to, raw := range connections {
			if err = nodeID(to); err != nil {
				return Snapshot{}, err
			}
			f, err := object(raw, "host", "status", "changed")
			if err != nil {
				return Snapshot{}, err
			}
			var c Connection
			if err = text(f["host"], &c.Host, 1024, false); err != nil {
				return Snapshot{}, err
			}
			if err = text(f["status"], &c.Status, 256, false); err != nil {
				return Snapshot{}, err
			}
			if err = boolean(f["changed"], &c.Changed); err != nil {
				return Snapshot{}, err
			}
			c.Unknown = unknown(f, "host", "status", "changed")
			node.Connections[to] = c
		}
		node.Unknown = unknown(fields, "status", "host", "changed", "version", "hostName", "connections")
		snapshot.Nodes[id] = node
	}
	snapshot.Unknown = unknown(obj, "nodes", "timestamp", "status")
	snapshot.Profile = profile
	snapshot.Raw = append([]byte(nil), b...)
	return snapshot, nil
}

func (l Limits) checked() (Limits, error) {
	if l.Bytes == 0 {
		l.Bytes = 4 << 20
	}
	if l.Nodes == 0 {
		l.Nodes = 1024
	}
	if l.Connections == 0 {
		l.Connections = 65536
	}
	if l.Depth == 0 {
		l.Depth = 32
	}
	if l.Bytes < 1 || l.Bytes > 16<<20 || l.Nodes < 1 || l.Nodes > 4096 || l.Connections < 1 || l.Connections > 262144 || l.Depth < 1 || l.Depth > 64 {
		return l, errors.New("imex: invalid limits")
	}
	return l, nil
}

func object(b []byte, required ...string) (map[string]json.RawMessage, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err != nil || obj == nil {
		return nil, errors.New("imex: expected object")
	}
	for _, key := range required {
		if v, ok := obj[key]; !ok || bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			return nil, fmt.Errorf("imex: missing %s", key)
		}
	}
	return obj, nil
}

func text(b []byte, target *string, limit int, empty bool) error {
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		return errors.New("imex: null string")
	}
	if err := json.Unmarshal(b, target); err != nil {
		return fmt.Errorf("imex string: %w", err)
	}
	if len(*target) > limit || (!empty && *target == "") {
		return errors.New("imex: invalid string extent")
	}
	return nil
}
func boolean(b []byte, target *bool) error {
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		return errors.New("imex: null flag")
	}
	if err := json.Unmarshal(b, target); err != nil {
		return fmt.Errorf("imex flag: %w", err)
	}
	return nil
}
func nodeID(s string) error {
	n, err := strconv.ParseUint(s, 10, 64)
	if err != nil || strconv.FormatUint(n, 10) != s {
		return fmt.Errorf("imex: invalid node id %q", s)
	}
	return nil
}
func unknown(obj map[string]json.RawMessage, known ...string) map[string]json.RawMessage {
	for _, key := range known {
		delete(obj, key)
	}
	if len(obj) == 0 {
		return nil
	}
	return obj
}
