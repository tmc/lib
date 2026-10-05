package httpapi

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
)

// EventStream owns a streaming HTTP body. Close disconnects and releases the
// server subscription. Read is serial; Close may run concurrently to stop it.
type EventStream struct {
	body    io.ReadCloser
	scanner *bufio.Scanner
}

func newEventStream(body io.ReadCloser) *EventStream {
	s := bufio.NewScanner(body)
	s.Buffer(make([]byte, 4096), 512<<10)
	return &EventStream{body: body, scanner: s}
}

// Read reads one bounded frame, including explicit retention gaps.
func (s *EventStream) Read() (Frame, error) {
	var f Frame
	if s == nil || s.scanner == nil {
		return f, io.EOF
	}
	if !s.scanner.Scan() {
		err := s.scanner.Err()
		if err == nil {
			err = io.EOF
		}
		return f, err
	}
	if err := json.Unmarshal(s.scanner.Bytes(), &f); err != nil {
		return f, err
	}
	if (f.Event == nil) == (f.Result == nil) {
		return f, errors.New("event frame must contain exactly one value")
	}
	return f, nil
}

// Close disconnects this stream.
func (s *EventStream) Close() error {
	if s == nil || s.body == nil {
		return nil
	}
	return s.body.Close()
}
