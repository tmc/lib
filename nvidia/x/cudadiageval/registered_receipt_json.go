package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
)

// ReadRegisteredReceipt reads one canonical v1 envelope, bounded to 4 MiB.
// It retains no authority and cannot construct an opaque pair. The caller owns
// the reader and interruption of a blocked read.
func ReadRegisteredReceipt(r io.Reader) (RegisteredReceipt, error) {
	var v RegisteredReceipt
	if r == nil {
		return v, errors.New("registered receipt: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, registrationLimit+1))
	if err != nil {
		return v, err
	}
	if len(b) > registrationLimit {
		return v, registrationBudgetError
	}
	if err := pairedDecode(b, &v, true); err != nil {
		return v, err
	}
	return v, validateEnvelope(v)
}

// WriteRegisteredReceipt writes canonical envelope bytes without a newline.
// It checks structure only, and propagates errors and short writes.
func WriteRegisteredReceipt(w io.Writer, v RegisteredReceipt) error {
	if w == nil {
		return errors.New("registered receipt: nil writer")
	}
	if err := validateEnvelope(v); err != nil {
		return err
	}
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if len(b) > registrationLimit {
		return registrationBudgetError
	}
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

// This walk is limited to the paired wire profiles. Presence checks precede
// encoding/json, which otherwise accepts null/missing scalar zero values.
func pairedDecode(b []byte, out any, canonical bool) error {
	if err := registrationTokenBounds(b); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var walk func() error
	walk = func() error {
		t, err := d.Token()
		if err != nil {
			return err
		}
		delim, ok := t.(json.Delim)
		if !ok {
			return nil
		}
		if delim != '{' && delim != '[' {
			return errors.New("paired receipt: invalid JSON")
		}
		seen := map[string]bool{}
		for d.More() {
			if delim == '{' {
				k, err := d.Token()
				if err != nil {
					return err
				}
				key, ok := k.(string)
				if !ok || seen[key] {
					return errors.New("paired receipt: duplicate field")
				}
				seen[key] = true
			}
			if err := walk(); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	}
	if err := walk(); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("paired receipt: trailing value")
	}
	if err := pairedKeys(b, reflect.TypeOf(out).Elem()); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return err
	}
	if canonical {
		c, err := json.Marshal(out)
		if err != nil {
			return err
		}
		if !bytes.Equal(c, b) {
			return errors.New("paired receipt: noncanonical bytes")
		}
	}
	return nil
}

func pairedKeys(b json.RawMessage, t reflect.Type) error {
	null := bytes.Equal(bytes.TrimSpace(b), []byte("null"))
	if t.Kind() == reflect.Pointer {
		if null {
			return nil
		}
		return pairedKeys(b, t.Elem())
	}
	if null && t.Kind() != reflect.Slice && t.Kind() != reflect.Map {
		return errors.New("paired receipt: null scalar or object")
	}
	if reflect.PointerTo(t).Implements(reflect.TypeFor[json.Unmarshaler]()) {
		return nil
	}
	switch t.Kind() {
	case reflect.Struct:
		var m map[string]json.RawMessage
		if err := json.Unmarshal(b, &m); err != nil {
			return err
		}
		fields := map[string]reflect.StructField{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			tag := strings.Split(f.Tag.Get("json"), ",")[0]
			if tag == "-" {
				continue
			}
			if tag == "" {
				tag = f.Name
			}
			fields[tag] = f
			required := !strings.Contains(f.Tag.Get("json"), "omitempty") || t == reflect.TypeFor[ToolReceipt]() && (tag == "synthetic" || tag == "raw_digest" || tag == "task") || t == reflect.TypeFor[PropertyMeasurement]() && tag == "value"
			if required {
				if _, ok := m[tag]; !ok {
					return fmt.Errorf("paired receipt: missing field %s", tag)
				}
			}
		}
		for key, value := range m {
			f, ok := fields[key]
			if !ok {
				return fmt.Errorf("paired receipt: unknown field %s", key)
			}
			if err := pairedKeys(value, f.Type); err != nil {
				return err
			}
		}
	case reflect.Slice:
		var a []json.RawMessage
		if err := json.Unmarshal(b, &a); err != nil {
			return err
		}
		if len(a) > 4096 {
			return errors.New("paired receipt: collection count exceeded")
		}
		for _, v := range a {
			if err := pairedKeys(v, t.Elem()); err != nil {
				return err
			}
		}
	case reflect.Map:
		var m map[string]json.RawMessage
		if err := json.Unmarshal(b, &m); err != nil {
			return err
		}
		if len(m) > 4096 {
			return errors.New("paired receipt: map count exceeded")
		}
		for _, v := range m {
			if err := pairedKeys(v, t.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

func pairedCharge(remaining *int64, n int) error {
	if n < 0 {
		return registrationBudgetError
	}
	*remaining -= int64(n)
	if *remaining < 0 {
		return registrationBudgetError
	}
	return nil
}

func pairedReadRecord(ctx context.Context, r io.Reader, lease *pairLease) ([]byte, error) {
	var b bytes.Buffer
	chunk := make([]byte, 64<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		lease.mu.Lock()
		room := lease.remaining
		lease.mu.Unlock()
		if fileRoom := int64(registrationLimit - b.Len()); fileRoom < room {
			room = fileRoom
		}
		size := len(chunk)
		if room < int64(size) {
			size = int(room) + 1
		}
		if size < 1 {
			size = 1
		}
		n, err := r.Read(chunk[:size])
		if n < 0 || n > size {
			return nil, errors.New("paired receipt: invalid reader count")
		}
		lease.mu.Lock()
		chargeErr := pairedCharge(&lease.remaining, n)
		lease.mu.Unlock()
		if chargeErr != nil {
			return nil, chargeErr
		}
		if b.Len()+n > registrationLimit {
			return nil, registrationBudgetError
		}
		b.Write(chunk[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if n == 0 {
			return nil, io.ErrNoProgress
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// This tighter pair reader charges every returned byte, including the one
// sentinel byte that proves a growth/aggregate overrun. Standalone registration
// retains its accepted reader behavior through the nil-extra snapshot path.
func pairedReadArtifact(ctx context.Context, f *os.File, remaining *int64) ([]byte, error) {
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("paired receipt: nonregular artifact")
	}
	if info.Size() > 16<<20 || info.Size() > *remaining {
		return nil, registrationBudgetError
	}
	var b bytes.Buffer
	chunk := make([]byte, 64<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		room := *remaining
		if fileRoom := int64((16 << 20) - b.Len()); fileRoom < room {
			room = fileRoom
		}
		size := len(chunk)
		if room < int64(size) {
			size = int(room) + 1
		}
		if size < 1 {
			size = 1
		}
		n, err := f.Read(chunk[:size])
		if chargeErr := pairedCharge(remaining, n); chargeErr != nil {
			return nil, chargeErr
		}
		if b.Len()+n > 16<<20 {
			return nil, registrationBudgetError
		}
		b.Write(chunk[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	return b.Bytes(), nil
}
