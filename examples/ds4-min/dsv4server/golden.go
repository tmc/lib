package dsv4server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// GoldenTranscript is the bounded, host-side representation of one server
// transcript. It is private to the server gate and is not an upstream format.
type GoldenTranscript struct {
	Request  []byte
	Response []byte
	Tokens   []int
}

const maxGoldenBytes = 8 << 20

// LoadGolden parses a txtar transcript containing request.json, response.json
// and tokens.txt. Missing, duplicate, unknown or malformed files are errors;
// a partial corpus must never be treated as a passing comparison.
func LoadGolden(data []byte) (GoldenTranscript, error) {
	if len(data) > maxGoldenBytes {
		return GoldenTranscript{}, fmt.Errorf("golden archive: exceeds %d-byte limit", maxGoldenBytes)
	}
	files, err := parseGoldenArchive(data)
	if err != nil {
		return GoldenTranscript{}, err
	}
	var out GoldenTranscript
	out.Request = append([]byte(nil), files["request.json"]...)
	out.Response = append([]byte(nil), files["response.json"]...)
	for _, line := range strings.Fields(string(files["tokens.txt"])) {
		token, err := strconv.Atoi(line)
		if err != nil {
			return GoldenTranscript{}, fmt.Errorf("tokens.txt: invalid token %q", line)
		}
		out.Tokens = append(out.Tokens, token)
	}
	if _, err := canonicalJSON(out.Request); err != nil {
		return GoldenTranscript{}, fmt.Errorf("request.json: %w", err)
	}
	if _, err := canonicalJSON(out.Response); err != nil {
		return GoldenTranscript{}, fmt.Errorf("response.json: %w", err)
	}
	return out, nil
}

// CompareGolden compares one observed transcript with a golden transcript.
// Response IDs, timestamps and event sequence numbers are masked, while
// request JSON and generated token IDs remain exact. The returned error is
// suitable for a fail-closed test gate.
func CompareGolden(golden GoldenTranscript, request, response []byte, tokens []int) error {
	wantRequest, err := canonicalJSON(golden.Request)
	if err != nil {
		return fmt.Errorf("golden request: %w", err)
	}
	gotRequest, err := canonicalJSON(request)
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	if !bytes.Equal(wantRequest, gotRequest) {
		return errors.New("request transcript differs")
	}
	wantResponse, err := canonicalMaskedJSON(golden.Response)
	if err != nil {
		return fmt.Errorf("golden response: %w", err)
	}
	gotResponse, err := canonicalMaskedJSON(response)
	if err != nil {
		return fmt.Errorf("response: %w", err)
	}
	if !bytes.Equal(wantResponse, gotResponse) {
		return errors.New("response transcript differs")
	}
	if len(golden.Tokens) != len(tokens) {
		return fmt.Errorf("token trace length %d, want %d", len(tokens), len(golden.Tokens))
	}
	for i := range tokens {
		if tokens[i] != golden.Tokens[i] {
			return fmt.Errorf("token trace differs at %d: got %d, want %d", i, tokens[i], golden.Tokens[i])
		}
	}
	return nil
}

func parseGoldenArchive(data []byte) (map[string][]byte, error) {
	const prefix = "-- "
	files := make(map[string][]byte, 3)
	for len(data) != 0 {
		if !bytes.HasPrefix(data, []byte(prefix)) {
			return nil, errors.New("golden archive: expected file header")
		}
		lineEnd := bytes.IndexByte(data, '\n')
		if lineEnd < 0 {
			return nil, errors.New("golden archive: truncated file header")
		}
		header := string(data[:lineEnd])
		if !strings.HasSuffix(header, " --") {
			return nil, fmt.Errorf("golden archive: malformed header %q", header)
		}
		name := strings.TrimSuffix(strings.TrimPrefix(header, prefix), " --")
		if name != "request.json" && name != "response.json" && name != "tokens.txt" {
			return nil, fmt.Errorf("golden archive: unexpected file %q", name)
		}
		if _, exists := files[name]; exists {
			return nil, fmt.Errorf("golden archive: duplicate file %q", name)
		}
		data = data[lineEnd+1:]
		next := bytes.Index(data, []byte("\n-- "))
		var body []byte
		if next < 0 {
			body = data
			data = nil
		} else {
			body = data[:next+1]
			data = data[next+1:]
		}
		files[name] = bytes.TrimSuffix(append([]byte(nil), body...), []byte("\n"))
	}
	for _, name := range []string{"request.json", "response.json", "tokens.txt"} {
		if _, ok := files[name]; !ok {
			return nil, fmt.Errorf("golden archive: missing file %q", name)
		}
	}
	return files, nil
}

func canonicalJSON(data []byte) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	value, err := decodeUniqueJSON(dec)
	if err != nil {
		return nil, err
	}
	var extra any
	if err := dec.Decode(&extra); err == nil {
		return nil, errors.New("multiple JSON values")
	} else if !errors.Is(err, io.EOF) {
		// Decoder errors after the first value are malformed trailing data.
		return nil, errors.New("invalid trailing JSON")
	}
	return json.Marshal(value)
}

// decodeUniqueJSON rejects duplicate object members before canonicalizing.
// encoding/json otherwise keeps the last member, which could hide a mutated
// transcript field from the comparison gate.
func decodeUniqueJSON(dec *json.Decoder) (any, error) {
	token, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch token {
	case json.Delim('{'):
		object := make(map[string]any)
		for dec.More() {
			keyToken, err := dec.Token()
			if err != nil {
				return nil, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, errors.New("object key is not a string")
			}
			if _, exists := object[key]; exists {
				return nil, fmt.Errorf("duplicate object key %q", key)
			}
			value, err := decodeUniqueJSON(dec)
			if err != nil {
				return nil, err
			}
			object[key] = value
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return object, nil
	case json.Delim('['):
		array := []any{}
		for dec.More() {
			value, err := decodeUniqueJSON(dec)
			if err != nil {
				return nil, err
			}
			array = append(array, value)
		}
		if _, err := dec.Token(); err != nil {
			return nil, err
		}
		return array, nil
	default:
		return token, nil
	}
}

func canonicalMaskedJSON(data []byte) ([]byte, error) {
	canonical, err := canonicalJSON(data)
	if err != nil {
		return nil, err
	}
	var value any
	if err := json.Unmarshal(canonical, &value); err != nil {
		return nil, err
	}
	return json.Marshal(maskVolatile(value))
}

func maskVolatile(value any) any {
	switch value := value.(type) {
	case []any:
		for i := range value {
			value[i] = maskVolatile(value[i])
		}
	case map[string]any:
		for key := range value {
			if volatileResponseField(key) {
				delete(value, key)
				continue
			}
			value[key] = maskVolatile(value[key])
		}
	}
	return value
}

// volatileResponseField identifies response metadata whose value is assigned
// by a particular server run. Generated nested identifiers use names such as
// call_id and output_item_id; masking the suffix keeps the transcript gate
// strict about content while avoiding a false mismatch on run identity.
func volatileResponseField(key string) bool {
	return key == "id" || key == "created" || key == "created_at" ||
		key == "sequence_number" || key == "signature" || strings.HasSuffix(key, "_id")
}
