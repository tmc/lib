package dsv4eval

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// TraceHeader contains the stable key-value metadata preceding cases.
type TraceHeader struct{ Fields map[string]string }

// TraceCase is one model-free trace result. Blocks are named prompt/output
// sections; their values contain the exact counted bytes. A final LF is optional.
type TraceCase struct {
	Index, Total int
	Source, ID   string
	Fields       map[string]string
	Blocks       map[string]string
	// Choices preserves upstream's contiguous A..J, single-line metadata.
	Choices []string
}

// Trace is a parsed trace document.
type Trace struct {
	Header TraceHeader
	Cases  []TraceCase
	// Summary contains the counters written after the terminating summary
	// marker. Keeping it separate prevents a summary from being mistaken for
	// header provenance.
	Summary map[string]string
	// SummaryPresent is true only when the terminating summary marker was
	// present.  A trace without it is incomplete, even if its last case was
	// otherwise parseable.
	SummaryPresent bool
}

// TraceWriter writes the strict key-value and counted-block trace format.
type TraceWriter struct {
	w   io.Writer
	err error
}

// NewTraceWriter writes the trace header and returns a writer for cases.
func NewTraceWriter(w io.Writer, h TraceHeader) *TraceWriter {
	t := &TraceWriter{w: w}
	if _, err := io.WriteString(w, "# ds4-eval trace\n"); err != nil {
		t.err = err
		return t
	}
	t.err = writeFields(w, h.Fields)
	return t
}

// Case appends one case, including all counted blocks.
func (t *TraceWriter) Case(c TraceCase) error {
	if t.err != nil {
		return t.err
	}
	if c.Index <= 0 || c.Total <= 0 || c.Index > c.Total || c.Source == "" || c.ID == "" {
		return fmt.Errorf("write trace case: invalid identity")
	}
	if strings.ContainsAny(c.Source+"/"+c.ID, "\r\n") {
		return fmt.Errorf("write trace case: invalid source or id")
	}
	fields := make(map[string]string, len(c.Fields)+2)
	for key, value := range c.Fields {
		if key == "source" && value != c.Source || key == "id" && value != c.ID {
			return fmt.Errorf("write trace case: %s field disagrees with header", key)
		}
		fields[key] = value
	}
	fields["source"], fields["id"] = c.Source, c.ID
	if _, err := fmt.Fprintf(t.w, "===== CASE %d/%d %s/%s =====\n", c.Index, c.Total, c.Source, c.ID); err != nil {
		return err
	}
	if err := writeFields(t.w, fields); err != nil {
		return err
	}
	for _, name := range []string{"SYSTEM_PROMPT", "QUESTION_PROMPT", "MODEL_OUTPUT"} {
		if body, ok := c.Blocks[name]; ok {
			if !strings.HasSuffix(body, "\n") {
				return fmt.Errorf("write trace case: %s must end with newline", name)
			}
			if _, err := fmt.Fprintf(t.w, "%s_BEGIN bytes=%d\n", name, len([]byte(body))); err != nil {
				return err
			}
			if _, err := io.WriteString(t.w, body); err != nil {
				return err
			}
			if _, err := fmt.Fprintf(t.w, "%s_END\n", name); err != nil {
				return err
			}
		}
	}
	return nil
}

// Summary terminates the trace with the optional summary key-value fields.
func (t *TraceWriter) Summary(fields map[string]string) error {
	if t.err != nil {
		return t.err
	}
	if _, err := io.WriteString(t.w, "===== SUMMARY =====\n"); err != nil {
		return err
	}
	return writeFields(t.w, fields)
}

func writeFields(w io.Writer, fields map[string]string) error {
	keys := make([]string, 0, len(fields))
	for k := range fields {
		if k == "" || strings.ContainsAny(k, "\r\n:") {
			return fmt.Errorf("write trace: invalid field key %q", k)
		}
		if strings.ContainsAny(fields[k], "\r\n") {
			return fmt.Errorf("write trace: field %q contains newline", k)
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if _, err := fmt.Fprintf(w, "%s: %s\n", k, fields[k]); err != nil {
			return err
		}
	}
	return nil
}

// ReadTrace parses a trace and rejects malformed key-value or counted-block
// records. It does not grade model output; use Regrade for that operation.
func ReadTrace(r io.Reader) (Trace, error) {
	br := bufio.NewReader(r)
	line, err := br.ReadString('\n')
	if err != nil || line != "# ds4-eval trace\n" {
		return Trace{}, fmt.Errorf("read trace: missing header")
	}
	trace := Trace{Header: TraceHeader{Fields: make(map[string]string)}, Summary: make(map[string]string)}
	var current *TraceCase
	inSummary := false
	finish := func() error {
		if current != nil {
			for key, want := range map[string]string{"source": current.Source, "id": current.ID} {
				if value, exists := current.Fields[key]; exists && value != want {
					return fmt.Errorf("read trace: %s field disagrees with case header", key)
				}
			}
			trace.Cases = append(trace.Cases, *current)
			current = nil
		}
		return nil
	}
	for {
		line, err = br.ReadString('\n')
		if err == io.EOF && line == "" {
			break
		}
		if err != nil && err != io.EOF {
			return Trace{}, fmt.Errorf("read trace: %w", err)
		}
		line = strings.TrimSuffix(line, "\n")
		if strings.HasSuffix(line, "\r") {
			return Trace{}, fmt.Errorf("read trace: CRLF is not supported")
		}
		if line == "===== SUMMARY =====" {
			if inSummary {
				return Trace{}, fmt.Errorf("read trace: duplicate summary")
			}
			if err := finish(); err != nil {
				return Trace{}, err
			}
			trace.SummaryPresent = true
			inSummary = true
			continue
		}
		if inSummary && strings.HasPrefix(line, "===== CASE ") {
			return Trace{}, fmt.Errorf("read trace: case after summary")
		}
		if strings.HasPrefix(line, "===== CASE ") {
			if err := finish(); err != nil {
				return Trace{}, err
			}
			c, ok := parseTraceCaseLine(line)
			if !ok {
				return Trace{}, fmt.Errorf("read trace: malformed case header")
			}
			current = &c
			continue
		}
		if line == "choices:" {
			if current == nil || current.Fields == nil || current.Choices != nil || len(current.Blocks) != 0 {
				return Trace{}, fmt.Errorf("read trace: misplaced or duplicate choices")
			}
			for {
				choice, err := br.ReadString('\n')
				if err != nil {
					return Trace{}, fmt.Errorf("read trace: truncated choices")
				}
				choice = strings.TrimSuffix(choice, "\n")
				if strings.HasPrefix(choice, "answer_kind: ") {
					if len(current.Choices) == 0 || choice != "answer_kind: choice" {
						return Trace{}, fmt.Errorf("read trace: invalid choices answer kind")
					}
					if _, exists := current.Fields["answer_kind"]; exists {
						return Trace{}, fmt.Errorf("read trace: duplicate answer_kind")
					}
					current.Fields["answer_kind"] = "choice"
					break
				}
				prefix := fmt.Sprintf("  %c. ", 'A'+len(current.Choices))
				if len(current.Choices) >= 10 || !strings.HasPrefix(choice, prefix) || strings.ContainsRune(choice, '\r') {
					return Trace{}, fmt.Errorf("read trace: malformed choice")
				}
				current.Choices = append(current.Choices, strings.TrimPrefix(choice, prefix))
			}
			continue
		}
		if strings.HasSuffix(line, "_BEGIN") || strings.Contains(line, "_BEGIN bytes=") {
			if current == nil {
				return Trace{}, fmt.Errorf("read trace: block outside case")
			}
			name, n, ok := parseBlockLine(line)
			if !ok {
				return Trace{}, fmt.Errorf("read trace: malformed block header")
			}
			body, err := io.ReadAll(io.LimitReader(br, int64(n)))
			if err != nil {
				return Trace{}, fmt.Errorf("read trace: block %s: %w", name, err)
			}
			if len(body) != n {
				return Trace{}, fmt.Errorf("read trace: block %s: %w", name, io.ErrUnexpectedEOF)
			}
			// Upstream counts only the exact body bytes, then writes an
			// uncounted LF separator when a nonempty body does not end in LF.
			// Empty bodies have no separator. Preserve the counted bytes.
			if len(body) != 0 && body[len(body)-1] != '\n' {
				separator, err := br.ReadByte()
				if err != nil || separator != '\n' {
					return Trace{}, fmt.Errorf("read trace: block %s lacks separator", name)
				}
			}
			end, err := br.ReadString('\n')
			if err != nil || strings.TrimSuffix(end, "\n") != name+"_END" {
				return Trace{}, fmt.Errorf("read trace: malformed block terminator %s", name)
			}
			if current.Blocks == nil {
				current.Blocks = make(map[string]string)
			}
			if _, exists := current.Blocks[name]; exists {
				return Trace{}, fmt.Errorf("read trace: duplicate block %s", name)
			}
			current.Blocks[name] = string(body)
			continue
		}
		if strings.Contains(line, ": ") {
			key, value, ok := strings.Cut(line, ": ")
			if !ok || key == "" || strings.ContainsAny(key, "\r\n:") {
				return Trace{}, fmt.Errorf("read trace: malformed field")
			}
			fields := trace.Header.Fields
			if current != nil {
				if current.Fields == nil {
					current.Fields = make(map[string]string)
				}
				fields = current.Fields
			} else if inSummary {
				fields = trace.Summary
			}
			if _, exists := fields[key]; exists {
				return Trace{}, fmt.Errorf("read trace: duplicate field %s", key)
			}
			fields[key] = value
			continue
		}
		if line != "" {
			return Trace{}, fmt.Errorf("read trace: unexpected line %q", line)
		}
		if err == io.EOF {
			break
		}
	}
	if current != nil {
		return Trace{}, fmt.Errorf("read trace: missing summary")
	}
	if !trace.SummaryPresent {
		return Trace{}, fmt.Errorf("read trace: missing summary")
	}
	return trace, nil
}

func parseTraceCaseLine(line string) (TraceCase, bool) {
	const prefix = "===== CASE "
	if !strings.HasPrefix(line, prefix) || !strings.HasSuffix(line, " =====") {
		return TraceCase{}, false
	}
	body := strings.TrimSuffix(strings.TrimPrefix(line, prefix), " =====")
	parts := strings.SplitN(body, " ", 2)
	if len(parts) != 2 {
		return TraceCase{}, false
	}
	nums, name, ok := strings.Cut(parts[0], "/")
	if !ok {
		return TraceCase{}, false
	}
	index, err1 := strconv.Atoi(nums)
	parts2 := strings.SplitN(parts[1], "/", 2)
	if err1 != nil || len(parts2) != 2 {
		return TraceCase{}, false
	}
	total, err2 := strconv.Atoi(name)
	if err2 != nil || index <= 0 || total < index || parts2[0] == "" || parts2[1] == "" {
		return TraceCase{}, false
	}
	return TraceCase{Index: index, Total: total, Source: parts2[0], ID: parts2[1]}, true
}

func parseBlockLine(line string) (string, int, bool) {
	name, rest, ok := strings.Cut(line, "_BEGIN bytes=")
	if !ok || name == "" {
		return "", 0, false
	}
	n, err := strconv.Atoi(rest)
	return name, n, err == nil && n >= 0
}
