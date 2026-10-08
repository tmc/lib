package imex

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"
)

func unique(b []byte, limits Limits) error {
	if !utf8.Valid(b) {
		return errors.New("imex: invalid UTF-8")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	counts := scanCounts{limits: limits}
	if err := value(d, 0, limits.Depth, scanRoot, &counts); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("imex: trailing JSON data")
	}
	return nil
}

type scanContext uint8

const (
	scanGeneric scanContext = iota
	scanRoot
	scanNodes
	scanNode
	scanConnections
)

type scanCounts struct {
	limits      Limits
	nodes       int
	connections int
}

func value(d *json.Decoder, depth, max int, context scanContext, counts *scanCounts) error {
	if depth > max {
		return errors.New("imex: JSON depth exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return fmt.Errorf("imex JSON: %w", err)
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		keys := make(map[string]bool)
		for d.More() {
			t, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := t.(string)
			if !ok {
				return errors.New("imex: nonstring key")
			}
			// Fold aliases too: consumers must not disagree about known-field spelling.
			folded := foldKey(key)
			if keys[folded] {
				return fmt.Errorf("imex: duplicate field %q", key)
			}
			keys[folded] = true
			next := scanGeneric
			switch context {
			case scanRoot:
				if key == "nodes" {
					next = scanNodes
				}
			case scanNodes:
				counts.nodes++
				if counts.nodes > counts.limits.Nodes {
					return errors.New("imex: node limit exceeded")
				}
				next = scanNode
			case scanNode:
				if key == "connections" {
					next = scanConnections
				}
			case scanConnections:
				counts.connections++
				if counts.connections > counts.limits.Connections {
					return errors.New("imex: connection limit exceeded")
				}
			}
			if err := value(d, depth+1, max, next, counts); err != nil {
				return err
			}
		}
		t, err = d.Token()
		if err != nil || t != json.Delim('}') {
			return errors.New("imex: unclosed object")
		}
	case '[':
		for d.More() {
			if err := value(d, depth+1, max, scanGeneric, counts); err != nil {
				return err
			}
		}
		t, err = d.Token()
		if err != nil || t != json.Delim(']') {
			return errors.New("imex: unclosed array")
		}
	default:
		return errors.New("imex: unexpected delimiter")
	}
	return nil
}
func foldKey(s string) string {
	var b strings.Builder
	for _, r := range s {
		min := r
		for next := unicode.SimpleFold(r); next != r; next = unicode.SimpleFold(next) {
			if next < min {
				min = next
			}
		}
		b.WriteRune(min)
	}
	return b.String()
}
