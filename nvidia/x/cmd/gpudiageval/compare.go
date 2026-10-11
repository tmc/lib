package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"syscall"

	"github.com/tmc/lib/nvidia/x/cudasotaeval"
)

func cmdCompare(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	protocol := fs.String("protocol", "", "Frozen comparison protocol JSON")
	candidate := fs.String("candidate", "", "Candidate raw receipt JSON")
	comparator := fs.String("comparator", "", "Comparator raw receipt JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *protocol == "" || *candidate == "" || *comparator == "" || fs.NArg() != 0 {
		return errors.New("compare requires -protocol, -candidate and -comparator")
	}
	pbytes, err := readComparisonFile(*protocol, 4<<20)
	if err != nil {
		return err
	}
	var p cudasotaeval.Protocol
	if err := decodeComparison(pbytes, &p); err != nil {
		return fmt.Errorf("decode protocol: %w", err)
	}
	cbytes, err := readComparisonFile(*candidate, 16<<20)
	if err != nil {
		return err
	}
	bbytes, err := readComparisonFile(*comparator, 16<<20)
	if err != nil {
		return err
	}
	var c, b cudasotaeval.ToolReceipt
	if err := decodeComparison(cbytes, &c); err != nil {
		return fmt.Errorf("decode candidate: %w", err)
	}
	if err := decodeComparison(bbytes, &b); err != nil {
		return fmt.Errorf("decode comparator: %w", err)
	}
	if c.RawDigest != "" || b.RawDigest != "" {
		return errors.New("raw receipts must omit their external digest")
	}
	c.RawDigest = fmt.Sprintf("%x", sha256.Sum256(cbytes))
	b.RawDigest = fmt.Sprintf("%x", sha256.Sum256(bbytes))
	results, err := cudasotaeval.CompareTools(p, c, b, bytes.NewReader(cbytes), bytes.NewReader(bbytes))
	if err != nil {
		return err
	}
	return writeComparisonJSON(stdout, results)
}

func readComparisonFile(path string, limit int64) ([]byte, error) {
	f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, fmt.Errorf("open comparison input: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return nil, errors.New("comparison input must be a bounded regular file")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, errors.New("comparison input exceeds size limit")
	}
	return b, nil
}

// Protocol fields use exact struct tags. Caller-defined property maps retain
// their names; Protocol.Validate decides whether those properties are admitted.
func decodeComparison(b []byte, out any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var walk func() error
	walk = func() error {
		tok, err := d.Token()
		if err != nil {
			return err
		}
		if delim, ok := tok.(json.Delim); ok {
			switch delim {
			case '{':
				seen := map[string]bool{}
				for d.More() {
					key, err := d.Token()
					if err != nil {
						return err
					}
					name, ok := key.(string)
					if !ok || seen[name] {
						return errors.New("duplicate comparison input field")
					}
					seen[name] = true
					if err := walk(); err != nil {
						return err
					}
				}
			case '[':
				for d.More() {
					if err := walk(); err != nil {
						return err
					}
				}
			default:
				return errors.New("invalid comparison input delimiter")
			}
			_, err = d.Token()
			return err
		}
		return nil
	}
	if err := walk(); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing comparison input")
	}
	if err := comparisonKeys(b, reflect.TypeOf(out).Elem()); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	return d.Decode(out)
}

func comparisonKeys(b json.RawMessage, t reflect.Type) error {
	if t.Kind() == reflect.Pointer {
		if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
			return nil
		}
		return comparisonKeys(b, t.Elem())
	}
	if reflect.PointerTo(t).Implements(reflect.TypeFor[json.Unmarshaler]()) {
		return nil
	}
	switch t.Kind() {
	case reflect.Struct:
		var values map[string]json.RawMessage
		if err := json.Unmarshal(b, &values); err != nil {
			return err
		}
		if values == nil {
			return errors.New("null comparison input object")
		}
		fields := map[string]reflect.Type{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			name := strings.Split(f.Tag.Get("json"), ",")[0]
			if name == "" {
				name = f.Name
			}
			fields[name] = f.Type
		}
		for name, v := range values {
			typ, ok := fields[name]
			if !ok {
				return fmt.Errorf("noncanonical comparison field %q", name)
			}
			if err := comparisonKeys(v, typ); err != nil {
				return err
			}
		}
	case reflect.Slice:
		var values []json.RawMessage
		if err := json.Unmarshal(b, &values); err != nil {
			return err
		}
		for _, v := range values {
			if err := comparisonKeys(v, t.Elem()); err != nil {
				return err
			}
		}
	case reflect.Map:
		var values map[string]json.RawMessage
		if err := json.Unmarshal(b, &values); err != nil {
			return err
		}
		for _, v := range values {
			if err := comparisonKeys(v, t.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeComparisonJSON(w io.Writer, value any) error {
	b, err := json.Marshal(value)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		return io.ErrShortWrite
	}
	return err
}
