package cudacache

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// memoryJSON checks duplicate keys and nesting before typed decoding. Inputs
// have already been byte-bounded; token count additionally bounds wide arrays.
func memoryJSON(data []byte) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	tokens := 0
	var value func(int) error
	value = func(depth int) error {
		if depth > 16 || tokens > 200000 {
			return fmt.Errorf("cudacache: search memory JSON complexity limit")
		}
		t, err := d.Token()
		if err != nil {
			return err
		}
		tokens++
		delim, ok := t.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			seen := make(map[string]bool)
			for d.More() {
				t, err := d.Token()
				if err != nil {
					return err
				}
				tokens++
				key, ok := t.(string)
				if !ok || seen[key] {
					return fmt.Errorf("cudacache: duplicate JSON field")
				}
				for _, c := range key {
					if c < 'a' || c > 'z' {
						if c != '_' && (c < '0' || c > '9') {
							return fmt.Errorf("cudacache: noncanonical JSON field")
						}
					}
				}
				seen[key] = true
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		case '[':
			for d.More() {
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("cudacache: invalid JSON delimiter")
		}
		_, err = d.Token()
		tokens++
		return err
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("cudacache: trailing JSON")
	}
	return nil
}
