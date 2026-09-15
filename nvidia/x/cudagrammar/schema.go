// SPDX-License-Identifier: Apache-2.0

package cudagrammar

import (
	"encoding/json"
	"fmt"
	"strings"
)

// kind is the shape of value a schema node describes.
type kind int

const (
	kObject kind = iota
	kArray
	kString
	kNumber
	kInteger
	kBool
	kNull
	kEnum
)

func (k kind) String() string {
	switch k {
	case kObject:
		return "object"
	case kArray:
		return "array"
	case kString:
		return "string"
	case kNumber:
		return "number"
	case kInteger:
		return "integer"
	case kBool:
		return "boolean"
	case kNull:
		return "null"
	case kEnum:
		return "enum"
	}
	return "unknown"
}

// node is a compiled schema node.
type node struct {
	kind  kind
	props []prop   // kObject, in declaration order
	items *node    // kArray
	enum  []string // kEnum
}

type prop struct {
	name string
	node *node
}

// Grammar is a compiled JSON schema. It is immutable once compiled and can be
// used to start any number of independent [Matcher] values.
type Grammar struct {
	root *node
}

// unsupportedError reports a schema this package cannot compile.
func unsupportedError(feature string) error {
	return fmt.Errorf("cudagrammar: unsupported schema feature: %s", feature)
}

// knownKeywords are the schema keywords Compile understands. The annotation
// keywords are read and ignored; the rest drive the automaton.
var knownKeywords = map[string]bool{
	"type": true, "properties": true, "required": true, "enum": true,
	"items": true, "additionalProperties": true,
	"$schema": true, "title": true, "description": true,
}

// Compile builds a grammar from a JSON schema.
//
// The supported subset is: objects with a fixed set of required properties,
// arrays, strings, numbers, integers, booleans, nulls, and enumerations of
// strings. Nesting is unrestricted. Every other schema keyword is rejected
// rather than ignored, so a schema that would not in fact be enforced never
// compiles.
//
// Objects are matched with their properties in the order the schema declares
// them. JSON itself does not order object members, but a decoder that is being
// constrained emits them one at a time, and fixing the order is what lets the
// automaton name the next key exactly.
func Compile(schema []byte) (*Grammar, error) {
	n, err := compileNode(schema, "")
	if err != nil {
		return nil, err
	}
	return &Grammar{root: n}, nil
}

// CompileString is Compile for a schema held in a string.
func CompileString(schema string) (*Grammar, error) {
	return Compile([]byte(schema))
}

func compileNode(raw []byte, path string) (*node, error) {
	where := func(s string) string {
		if path == "" {
			return s
		}
		return s + " at " + path
	}

	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, fmt.Errorf("cudagrammar: parsing schema%s: %w", pathSuffix(path), err)
	}
	for k := range fields {
		if !knownKeywords[k] {
			return nil, unsupportedError(where(k))
		}
	}

	var typeName string
	if rawType, ok := fields["type"]; ok {
		if err := json.Unmarshal(rawType, &typeName); err != nil {
			return nil, unsupportedError(where("type that is not a single type name"))
		}
	}

	if rawEnum, ok := fields["enum"]; ok {
		if typeName != "" && typeName != "string" {
			return nil, unsupportedError(where("enum of type " + typeName))
		}
		var values []string
		if err := json.Unmarshal(rawEnum, &values); err != nil {
			return nil, unsupportedError(where("enum whose values are not all strings"))
		}
		if len(values) == 0 {
			return nil, unsupportedError(where("empty enum"))
		}
		for _, v := range values {
			if !plainString(v) {
				return nil, unsupportedError(where("enum value needing escapes: " + v))
			}
		}
		return &node{kind: kEnum, enum: values}, nil
	}

	switch typeName {
	case "":
		return nil, unsupportedError(where("schema without a type"))
	case "string":
		return &node{kind: kString}, nil
	case "number":
		return &node{kind: kNumber}, nil
	case "integer":
		return &node{kind: kInteger}, nil
	case "boolean":
		return &node{kind: kBool}, nil
	case "null":
		return &node{kind: kNull}, nil
	case "array":
		rawItems, ok := fields["items"]
		if !ok {
			return nil, unsupportedError(where("array without items"))
		}
		items, err := compileNode(rawItems, join(path, "items"))
		if err != nil {
			return nil, err
		}
		return &node{kind: kArray, items: items}, nil
	case "object":
		return compileObject(fields, path, where)
	default:
		return nil, unsupportedError(where("type " + typeName))
	}
}

func compileObject(fields map[string]json.RawMessage, path string, where func(string) string) (*node, error) {
	if rawAdd, ok := fields["additionalProperties"]; ok {
		var allowed bool
		if err := json.Unmarshal(rawAdd, &allowed); err != nil || allowed {
			return nil, unsupportedError(where("additionalProperties other than false"))
		}
	}
	rawProps, ok := fields["properties"]
	if !ok {
		return nil, unsupportedError(where("object without properties"))
	}
	names, err := objectKeys(rawProps)
	if err != nil {
		return nil, fmt.Errorf("cudagrammar: parsing properties%s: %w", pathSuffix(path), err)
	}
	if len(names) == 0 {
		return nil, unsupportedError(where("object with no properties"))
	}
	var members map[string]json.RawMessage
	if err := json.Unmarshal(rawProps, &members); err != nil {
		return nil, fmt.Errorf("cudagrammar: parsing properties%s: %w", pathSuffix(path), err)
	}

	if rawReq, ok := fields["required"]; ok {
		var required []string
		if err := json.Unmarshal(rawReq, &required); err != nil {
			return nil, unsupportedError(where("required that is not a list of names"))
		}
		have := make(map[string]bool, len(required))
		for _, r := range required {
			have[r] = true
		}
		for _, n := range names {
			if !have[n] {
				return nil, unsupportedError(where("optional property " + n))
			}
		}
		for _, r := range required {
			if _, ok := members[r]; !ok {
				return nil, unsupportedError(where("required property " + r + " that is not declared"))
			}
		}
	}

	n := &node{kind: kObject, props: make([]prop, 0, len(names))}
	for _, name := range names {
		if !plainString(name) {
			return nil, unsupportedError(where("property name needing escapes: " + name))
		}
		child, err := compileNode(members[name], join(path, name))
		if err != nil {
			return nil, err
		}
		n.props = append(n.props, prop{name: name, node: child})
	}
	return n, nil
}

// objectKeys returns the member names of a JSON object in document order,
// which json.Unmarshal into a map does not preserve.
func objectKeys(raw []byte) ([]string, error) {
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("expected an object, got %v", tok)
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		name, ok := tok.(string)
		if !ok {
			return nil, fmt.Errorf("expected a member name, got %v", tok)
		}
		keys = append(keys, name)
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return nil, err
		}
	}
	return keys, nil
}

// plainString reports whether s can appear inside a JSON string literal
// without escapes, which is what lets the automaton match it byte for byte.
func plainString(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < 0x20 || c == '"' || c == '\\' {
			return false
		}
	}
	return true
}

func join(path, elem string) string {
	if path == "" {
		return elem
	}
	return path + "." + elem
}

func pathSuffix(path string) string {
	if path == "" {
		return ""
	}
	return " at " + path
}
