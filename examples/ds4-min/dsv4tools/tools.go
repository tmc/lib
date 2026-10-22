// Package dsv4tools parses the bounded DSML tool representation used by the
// DeepSeek server. It does not execute tools or interpret their schemas.
package dsv4tools

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

const (
	maxTools      = 64
	maxSchemaSize = 64 << 10
	maxCalls      = 64
	maxName       = 256
	maxParameters = 256
	maxDSMLSize   = 1 << 20
	maxNesting    = 8
)

// Schema is one validated tool schema. Raw contains the normalized function
// object, not the OpenAI type:function wrapper.
type Schema struct {
	Name string
	Raw  json.RawMessage
}

// Call is a generated tool call. Arguments is a compact JSON object.
type Call struct {
	Name      string
	Arguments json.RawMessage
}

// Validate checks and normalizes tool schemas. OpenAI's
// {type:"function",function:{...}} form is unwrapped.
func Validate(raw []json.RawMessage) ([]Schema, error) {
	if len(raw) > maxTools {
		return nil, fmt.Errorf("too many tools")
	}
	out := make([]Schema, 0, len(raw))
	for i, item := range raw {
		if len(item) == 0 || len(item) > maxSchemaSize {
			return nil, fmt.Errorf("tool %d schema is too large", i)
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(item, &obj); err != nil {
			return nil, fmt.Errorf("tool %d schema is not an object: %w", i, err)
		}
		if wrapped, ok := obj["function"]; ok {
			var typ string
			if err := json.Unmarshal(obj["type"], &typ); err != nil || typ != "function" {
				return nil, fmt.Errorf("tool %d has invalid function wrapper", i)
			}
			var function map[string]json.RawMessage
			if err := json.Unmarshal(wrapped, &function); err != nil {
				return nil, fmt.Errorf("tool %d function is not an object", i)
			}
			obj = function
		}
		name, ok := stringField(obj, "name")
		if !ok || name == "" || len(name) > maxName {
			return nil, fmt.Errorf("tool %d has invalid name", i)
		}
		if p, ok := obj["parameters"]; ok && !isObject(p) {
			return nil, fmt.Errorf("tool %d parameters is not an object", i)
		}
		data, err := json.Marshal(obj)
		if err != nil {
			return nil, fmt.Errorf("tool %d schema: %w", i, err)
		}
		out = append(out, Schema{Name: name, Raw: data})
	}
	return out, nil
}

// Parse extracts canonical tool calls from generated DSML. Text outside the
// tool-call wrapper is returned as content. Unsupported or malformed DSML is
// an error; callers must not silently treat a partial call as normal text.
func Parse(text string) (content string, calls []Call, err error) {
	if len(text) > maxDSMLSize {
		return "", nil, errors.New("tool markup is too large")
	}
	markers := []struct{ open, close, invokeOpen, invokeClose, parameterOpen, parameterClose string }{
		{"<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>", "<｜DSML｜invoke", "</｜DSML｜invoke>", "<｜DSML｜parameter", "</｜DSML｜parameter>"},
		{"<DSML｜tool_calls>", "</DSML｜tool_calls>", "<DSML｜invoke", "</DSML｜invoke>", "<DSML｜parameter", "</DSML｜parameter>"},
		{"<tool_calls>", "</tool_calls>", "<invoke", "</invoke>", "<parameter", "</parameter>"},
	}
	var marker struct{ open, close, invokeOpen, invokeClose, parameterOpen, parameterClose string }
	start := -1
	for _, candidate := range markers {
		if i := strings.Index(text, candidate.open); i >= 0 && (start < 0 || i < start) {
			start, marker = i, candidate
		}
	}
	if start < 0 {
		for _, prefix := range []string{"<｜DSML｜invoke", "<｜DSML｜parameter", "<DSML｜invoke", "<DSML｜parameter", "<invoke", "<parameter"} {
			if strings.Contains(text, prefix) {
				return "", nil, errors.New("tool markup is missing tool_calls wrapper")
			}
		}
		return text, nil, nil
	}
	end := strings.Index(text[start+len(marker.open):], marker.close)
	if end < 0 {
		return "", nil, errors.New("unterminated tool_calls")
	}
	end += start + len(marker.open)
	content = strings.TrimSpace(text[:start] + text[end+len(marker.close):])
	body := text[start+len(marker.open) : end]
	for {
		i := strings.Index(body, marker.invokeOpen)
		if i < 0 {
			if strings.TrimSpace(body) != "" {
				return "", nil, errors.New("unexpected content in tool_calls")
			}
			break
		}
		j := strings.Index(body[i:], ">")
		if j < 0 {
			return "", nil, errors.New("malformed invoke")
		}
		j += i
		attrs := body[i+len(marker.invokeOpen) : j]
		name, ok := attr(attrs, "name")
		if !ok || name == "" {
			return "", nil, errors.New("invoke has no name")
		}
		closeInvoke := strings.Index(body[j+1:], marker.invokeClose)
		if closeInvoke < 0 {
			return "", nil, errors.New("unterminated invoke")
		}
		closeInvoke += j + 1
		params, e := parseParameters(body[j+1:closeInvoke], marker.parameterOpen, marker.parameterClose, 0)
		if e != nil {
			return "", nil, e
		}
		args, e := json.Marshal(params)
		if e != nil {
			return "", nil, e
		}
		calls = append(calls, Call{Name: name, Arguments: args})
		if len(calls) > maxCalls {
			return "", nil, errors.New("too many tool calls")
		}
		body = body[closeInvoke+len(marker.invokeClose):]
	}
	if len(calls) == 0 {
		return "", nil, errors.New("tool_calls has no invoke")
	}
	return content, calls, nil
}

func parseParameters(body, open, close string, depth int) (map[string]json.RawMessage, error) {
	if depth >= maxNesting {
		return nil, errors.New("parameter nesting is too deep")
	}
	out := make(map[string]json.RawMessage)
	count := 0
	for {
		body = strings.TrimSpace(body)
		if body == "" {
			return out, nil
		}
		if !strings.HasPrefix(body, open) {
			return nil, errors.New("unexpected content in invoke")
		}
		j := strings.IndexByte(body, '>')
		if j < 0 {
			return nil, errors.New("malformed parameter")
		}
		attrs := body[len(open):j]
		name, ok := attr(attrs, "name")
		if !ok || name == "" {
			return nil, errors.New("parameter has no name")
		}
		stringValue, _ := attr(attrs, "string")
		k := matchingParameterClose(body[j+1:], open, close)
		if k < 0 {
			return nil, errors.New("unterminated parameter")
		}
		k += j + 1
		value := strings.TrimSpace(body[j+1 : k])
		var raw json.RawMessage
		if strings.HasPrefix(value, open) {
			nested, err := parseParameters(value, open, close, depth+1)
			if err != nil {
				return nil, err
			}
			raw, err = json.Marshal(nested)
			if err != nil {
				return nil, err
			}
		} else if stringValue == "true" {
			raw = json.RawMessage(strconv.Quote(value))
		} else if !json.Valid([]byte(value)) {
			return nil, fmt.Errorf("parameter %q is not JSON", name)
		} else {
			raw = json.RawMessage(value)
		}
		count++
		if count > maxParameters {
			return nil, errors.New("too many parameters")
		}
		out[name] = raw
		body = body[k+len(close):]
	}
}

// matchingParameterClose returns the offset of the close tag that balances
// the parameter whose body starts at body. Parameter values may themselves be
// parameter objects, so the first close tag is not necessarily the right one.
func matchingParameterClose(body, open, close string) int {
	depth := 1
	for pos := 0; pos < len(body); {
		i := strings.Index(body[pos:], open)
		j := strings.Index(body[pos:], close)
		if j < 0 {
			return -1
		}
		j += pos
		if i >= 0 {
			i += pos
		}
		if i >= 0 && i < j {
			depth++
			pos = i + len(open)
			continue
		}
		depth--
		if depth == 0 {
			return j
		}
		pos = j + len(close)
	}
	return -1
}

func attr(s, name string) (string, bool) {
	needle := name + "=\""
	i := strings.Index(s, needle)
	if i < 0 {
		return "", false
	}
	s = s[i+len(needle):]
	j := strings.IndexByte(s, '"')
	if j < 0 {
		return "", false
	}
	v, err := strconv.Unquote("\"" + s[:j] + "\"")
	return v, err == nil
}

func stringField(obj map[string]json.RawMessage, name string) (string, bool) {
	v, ok := obj[name]
	if !ok {
		return "", false
	}
	var s string
	err := json.Unmarshal(v, &s)
	return s, err == nil
}

func isObject(v json.RawMessage) bool {
	v = bytes.TrimSpace(v)
	return len(v) >= 2 && v[0] == '{' && v[len(v)-1] == '}' && json.Valid(v)
}
