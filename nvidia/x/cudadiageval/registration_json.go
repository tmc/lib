package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"
)

// ReadRegistration reads at most 4 MiB. It rejects duplicate and noncanonical
// keys, unknown fields, future versions, trailing data and nesting beyond 32.
// The caller owns the reader and interruption of a blocked read.
func ReadRegistration(r io.Reader) (Registration, error) {
	var out Registration
	if r == nil {
		return out, errors.New("read registration: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, registrationLimit+1))
	if err != nil {
		return out, fmt.Errorf("read registration: %w", err)
	}
	if len(b) > registrationLimit {
		return out, errors.New("read registration: size limit exceeded")
	}
	if err := registrationDecode(b, &out); err != nil {
		return out, fmt.Errorf("read registration: %w", err)
	}
	return out, out.Validate()
}

// WriteRegistration validates and writes one canonical JSON record. A short
// write is an error. It does not authenticate policy artifacts or measurements.
func WriteRegistration(w io.Writer, r Registration) error {
	if w == nil {
		return errors.New("write registration: nil writer")
	}
	if err := r.Validate(); err != nil {
		return err
	}
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

// registrationTokenBounds is deliberately scoped to registration inputs. Unlike
// ledgerDecode it bounds nesting and scalar size before a recursive tag check.
func registrationTokenBounds(b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	depth, tokens := 0, 0
	for {
		t, err := d.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		tokens++
		if tokens > 1<<20 {
			return errors.New("registration: token count limit exceeded")
		}
		switch v := t.(type) {
		case json.Delim:
			if v == '{' || v == '[' {
				depth++
				if depth > 32 {
					return errors.New("registration: nesting limit exceeded")
				}
			} else {
				depth--
			}
		case string:
			if len(v) > 4096 {
				return errors.New("registration: string limit exceeded")
			}
		case json.Number:
			if len(v) > 4096 {
				return errors.New("registration: number limit exceeded")
			}
		}
	}
	return nil
}

func registrationDecode(b []byte, out *Registration) error {
	if err := registrationTokenBounds(b); err != nil {
		return err
	}
	// Check duplicate keys before decoding into maps can discard them.
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var walk func() error
	walk = func() error {
		tok, err := d.Token()
		if err != nil {
			return err
		}
		if delim, ok := tok.(json.Delim); ok {
			if delim != '{' && delim != '[' {
				return errors.New("registration: invalid delimiter")
			}
			seen := map[string]bool{}
			for d.More() {
				if delim == '{' {
					key, err := d.Token()
					if err != nil {
						return err
					}
					s, ok := key.(string)
					if !ok || seen[s] {
						return errors.New("registration: duplicate JSON field")
					}
					seen[s] = true
				}
				if err := walk(); err != nil {
					return err
				}
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
		return errors.New("registration: trailing JSON value")
	}
	if err := registrationKeys(b, reflect.TypeOf(*out)); err != nil {
		return err
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	return d.Decode(out)
}

// Pointer and custom JSON values need explicit treatment. The existing ledger
// canonical walker lacks these cases; Protocol includes both Comparison and
// time.Time. Maps retain caller-defined names; Protocol.Validate admits their
// meanings. This helper does not change the ledger's serialized contracts.
func registrationKeys(b json.RawMessage, t reflect.Type) error {
	if t.Kind() == reflect.Pointer {
		if bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
			return nil
		}
		return registrationKeys(b, t.Elem())
	}
	// encoding/json accepts null for scalars without changing their zero value.
	// Only pointers and collections have a nullable registration representation.
	if bytes.Equal(bytes.TrimSpace(b), []byte("null")) && t.Kind() != reflect.Slice && t.Kind() != reflect.Map {
		return errors.New("registration: null nonnullable value")
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
		if m == nil {
			return errors.New("registration: null object")
		}
		fields := map[string]reflect.Type{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			name := strings.Split(f.Tag.Get("json"), ",")[0]
			fields[name] = f.Type
		}
		// New registration schema fields are explicit even when zero or null.
		// Missing clock bounds/warmup/loss may not silently become known zeros.
		if t.PkgPath() == reflect.TypeFor[Registration]().PkgPath() && strings.HasPrefix(t.Name(), "Registration") {
			for name := range fields {
				if _, ok := m[name]; !ok {
					return fmt.Errorf("registration: missing explicit field %q", name)
				}
			}
		}
		for name, value := range m {
			typ, ok := fields[name]
			if !ok {
				return fmt.Errorf("registration: noncanonical field %q", name)
			}
			if err := registrationKeys(value, typ); err != nil {
				return err
			}
		}
	case reflect.Slice:
		var vs []json.RawMessage
		if err := json.Unmarshal(b, &vs); err != nil {
			return err
		}
		if len(vs) > 4096 {
			return errors.New("registration: collection count limit exceeded")
		}
		for _, v := range vs {
			if err := registrationKeys(v, t.Elem()); err != nil {
				return err
			}
		}
	case reflect.Map:
		var vs map[string]json.RawMessage
		if err := json.Unmarshal(b, &vs); err != nil {
			return err
		}
		if len(vs) > 4096 {
			return errors.New("registration: collection count limit exceeded")
		}
		for _, v := range vs {
			if err := registrationKeys(v, t.Elem()); err != nil {
				return err
			}
		}
	}
	return nil
}

// registrationSnapshot acquires each original through a no-follow descriptor,
// hashes bounded bytes, and gives the unchanged ledger verifier private copies.
// Neither snapshot paths nor caller timestamps are acquisition authority.
func registrationSnapshot(ctx context.Context, r Registration, l QualificationLedger, root string) (result string, resultCleanup func() error, outcomes []RegistrationArtifactResult, resultErr error) {
	remaining := int64(64 << 20)
	return registrationSnapshotBudget(ctx, r, l, root, nil, &remaining)
}

// Paired admission supplies its one operation budget and mandatory extra inputs.
func registrationSnapshotBudget(ctx context.Context, r Registration, l QualificationLedger, root string, extra []RegistrationArtifact, remaining *int64) (result string, resultCleanup func() error, outcomes []RegistrationArtifactResult, resultErr error) {
	refs, err := r.artifacts()
	if err != nil {
		return "", nil, nil, err
	}
	refs = append(refs, extra...)
	mandatory := map[string]bool{}
	for _, a := range refs {
		mandatory[a.Path] = true
	}
	cells := map[string]QualificationCell{}
	for _, c := range l.Cells {
		cells[c.ID] = c
	}
	required := map[string]bool{}
	var selectCell func(string)
	selectCell = func(id string) {
		if required[id] {
			return
		}
		required[id] = true
		for _, dep := range cells[id].DependsOn {
			selectCell(dep)
		}
	}
	for _, c := range r.SelectedCells {
		if c.Applicability == "selected" {
			selectCell(c.CellID)
		}
	}
	for _, c := range l.Cells {
		for _, a := range c.Attempts {
			refs = append(refs, RegistrationArtifact{a.Path, a.SHA256}, RegistrationArtifact{a.Evidence.RawPath, a.Evidence.RawSHA256})
			if required[c.ID] {
				mandatory[a.Path] = true
				mandatory[a.Evidence.RawPath] = true
			}
		}
		for _, ref := range c.References {
			if ref.SHA256 != "" {
				refs = append(refs, RegistrationArtifact{ref.Path, ref.SHA256})
			}
		}
	}
	if len(refs) > 4096 {
		return "", nil, nil, errors.New("verify registration: aggregate artifact count exceeded")
	}
	paths := map[string]string{}
	for _, a := range refs {
		if err := registrationArtifacts([]RegistrationArtifact{a}); err != nil {
			return "", nil, nil, err
		}
		if old, ok := paths[a.Path]; ok && old != a.SHA256 {
			return "", nil, nil, errors.New("verify registration: conflicting original artifact identity")
		}
		paths[a.Path] = a.SHA256
	}
	original, err := registrationOpenRoot(root)
	if err != nil {
		return "", nil, nil, err
	}
	defer func() {
		if err := original.close(); err != nil {
			if resultCleanup != nil {
				err = errors.Join(err, resultCleanup())
				result = ""
				resultCleanup = nil
			}
			resultErr = errors.Join(resultErr, err)
		}
	}()
	dir, err := os.MkdirTemp("", "cudadiageval-registration-")
	if err != nil {
		return "", nil, nil, err
	}
	cleanup := func() error { return os.RemoveAll(dir) }
	fail := func(err error) (string, func() error, []RegistrationArtifactResult, error) {
		return "", nil, outcomes, errors.Join(err, cleanup())
	}
	names := make([]string, 0, len(paths))
	for name := range paths {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		v := RegistrationArtifactResult{Artifact: RegistrationArtifact{name, paths[name]}, Required: mandatory[name], Status: "consistent"}
		f, err := original.open(ctx, name)
		var data []byte
		if err == nil {
			if extra == nil {
				data, err = registrationReadArtifact(ctx, f, remaining)
			} else {
				data, err = pairedReadArtifact(ctx, f, remaining)
			}
			closeErr := f.Close()
			if closeErr != nil {
				return fail(errors.Join(err, closeErr))
			}
			if err == nil && registrationSHA(data) != paths[name] {
				err = errors.New("artifact digest mismatch")
				v.Status = "corrupt"
			}
		}
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, registrationBudgetError) {
				return fail(err)
			}
			if mandatory[name] {
				return fail(fmt.Errorf("verify registration artifact %s: %w", name, err))
			}
			if v.Status != "corrupt" {
				v.Status = "unavailable"
				if errors.Is(err, os.ErrNotExist) {
					v.Status = "missing"
				}
			}
			v.Reason = err.Error()
			outcomes = append(outcomes, v)
			continue
		}
		dst := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(dst), 0700); err != nil {
			return fail(err)
		}
		if err := os.WriteFile(dst, data, 0600); err != nil {
			return fail(err)
		}
		outcomes = append(outcomes, v)
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	return dir, cleanup, outcomes, nil
}

var registrationBudgetError = errors.New("verify registration: artifact size or aggregate budget exceeded")

func registrationReadArtifact(ctx context.Context, f *os.File, remaining *int64) ([]byte, error) {
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("verify registration: artifact is not a regular file")
	}
	if info.Size() > 16<<20 || info.Size() > *remaining {
		return nil, registrationBudgetError
	}
	var b bytes.Buffer
	buf := make([]byte, 64<<10)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := f.Read(buf)
		if int64(n) > *remaining || b.Len()+n > 16<<20 {
			return nil, registrationBudgetError
		}
		*remaining -= int64(n)
		b.Write(buf[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	return b.Bytes(), nil
}

// Bound directly constructed values before marshaling, hashing or semantic
// traversal. The size lower bound includes field names and raw string bytes;
// the exact encoded 4 MiB limit is checked again after marshaling.
func registrationShapeBounds(value reflect.Value) error {
	size, tokens := 0, 0
	var walk func(reflect.Value, int) error
	walk = func(v reflect.Value, depth int) error {
		tokens++
		if tokens > 1<<20 || depth > 32 {
			return errors.New("registration: shape count or depth limit exceeded")
		}
		if v.Kind() == reflect.Pointer {
			if v.IsNil() {
				size += 4
			} else {
				return walk(v.Elem(), depth)
			}
			return nil
		}
		if v.Type() == reflect.TypeFor[time.Time]() {
			size += 32
			return nil
		}
		switch v.Kind() {
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				f := v.Type().Field(i)
				if !f.IsExported() {
					continue
				}
				size += len(strings.Split(f.Tag.Get("json"), ",")[0]) + 4
				if err := walk(v.Field(i), depth+1); err != nil {
					return err
				}
			}
		case reflect.Slice, reflect.Map:
			if v.Len() > 4096 {
				return errors.New("registration: collection count limit exceeded")
			}
			if v.Kind() == reflect.Slice {
				for i := 0; i < v.Len(); i++ {
					if err := walk(v.Index(i), depth+1); err != nil {
						return err
					}
				}
			} else {
				for _, key := range v.MapKeys() {
					if err := walk(key, depth+1); err != nil {
						return err
					}
					if err := walk(v.MapIndex(key), depth+1); err != nil {
						return err
					}
				}
			}
		case reflect.String:
			if len(v.String()) > 4096 {
				return errors.New("registration: string limit exceeded")
			}
			size += len(v.String()) + 2
		default:
			size++
		}
		if size > registrationLimit {
			return errors.New("registration: size limit exceeded")
		}
		return nil
	}
	return walk(value, 0)
}
