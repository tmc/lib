package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nsightsystems"
)

// A header-only file is sufficient for process-fault controls: no fake success
// is presented as SQLite integration or canonical schema admission.
func faultReader(t *testing.T, process sqliteProcess) *sqliteReader {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fault.sqlite")
	if err := os.WriteFile(path, []byte("SQLite format 3\x00"), 0600); err != nil {
		t.Fatal(err)
	}
	r, err := newSQLiteReader(context.Background(), path, testNSYSOptions(), process)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(r.dir); err != nil {
			t.Error(err)
		}
	})
	r.window = wholeCapture()
	r.maxKernels = 10
	return r
}

func TestNSYSProcessBounds(t *testing.T) {
	providerErr := errors.New("provider failed")
	for _, tt := range []struct {
		name    string
		process sqliteProcess
		want    error
	}{
		{"stdout", func(ctx context.Context, _ string, _ []string, out, errout io.Writer) error {
			_, err := io.WriteString(out, strings.Repeat("x", 129))
			return errors.Join(providerErr, err)
		}, errSQLiteLimit},
		{"stderr", func(ctx context.Context, _ string, _ []string, out, errout io.Writer) error {
			_, err := io.WriteString(errout, strings.Repeat("x", (64<<10)+1))
			return errors.Join(providerErr, err)
		}, errSQLiteLimit},
		{"provider", func(ctx context.Context, _ string, _ []string, out, errout io.Writer) error {
			io.WriteString(out, "[]")
			return providerErr
		}, providerErr},
		{"deadline", func(ctx context.Context, _ string, _ []string, out, errout io.Writer) error {
			<-ctx.Done()
			return errors.Join(providerErr, ctx.Err())
		}, context.DeadlineExceeded},
	} {
		t.Run(tt.name, func(t *testing.T) {
			acquired := false
			process := func(ctx context.Context, tool string, args []string, out, errout io.Writer) error {
				acquired = true
				return tt.process(ctx, tool, args, out, errout)
			}
			r := faultReader(t, process)
			r.bytesLeft = 128
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
			defer cancel()
			_, err := r.query(ctx, "SELECT 1", 1)
			if !errors.Is(err, tt.want) || errors.Is(err, providerErr) != acquired || (!acquired && tt.name != "deadline") {
				t.Fatalf("lost limit/context/provider identity: %v", err)
			}
		})
	}
	t.Run("cancel after acquisition", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		called := false
		r := faultReader(t, func(ctx context.Context, _ string, _ []string, _ io.Writer, _ io.Writer) error {
			called = true
			cancel()
			return errors.Join(providerErr, ctx.Err())
		})
		_, err := r.query(ctx, "SELECT 1", 1)
		if !called || !errors.Is(err, context.Canceled) || !errors.Is(err, providerErr) {
			t.Fatalf("acquired cancellation: %v called=%t", err, called)
		}
	})
	t.Run("already canceled", func(t *testing.T) {
		called := false
		r := faultReader(t, func(context.Context, string, []string, io.Writer, io.Writer) error { called = true; return nil })
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := r.query(ctx, "SELECT 1", 1)
		if !errors.Is(err, context.Canceled) || called {
			t.Fatalf("canceled query: %v called=%t", err, called)
		}
	})
}

func TestNSYSJSONBounds(t *testing.T) {
	for _, tt := range []struct {
		name, data string
		limit      int
	}{
		{"rows", `[{"x":1},{"x":2}]`, 1},
		{"not array", `{"x":1}`, 2},
		{"trailing", `[] []`, 2},
		{"malformed", `[{`, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := faultReader(t, func(_ context.Context, _ string, _ []string, out, _ io.Writer) error {
				_, err := io.WriteString(out, tt.data)
				return err
			})
			if _, err := r.query(context.Background(), "SELECT 1", tt.limit); err == nil {
				t.Fatal("accepted invalid or excessive JSON")
			}
		})
	}
}

func TestNSYSUnavailableIsNotFailure(t *testing.T) {
	fail := errors.New("SQLite unavailable")
	for _, tt := range []struct {
		name    string
		process sqliteProcess
		want    error
	}{
		{"missing table", func(_ context.Context, _ string, _ []string, out, _ io.Writer) error {
			_, err := io.WriteString(out, "[]")
			return err
		}, nsightsystems.ErrUnavailable},
		{"query failure", func(context.Context, string, []string, io.Writer, io.Writer) error { return fail }, fail},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := faultReader(t, tt.process)
			_, err := r.ReadTable(context.Background(), "StringIds", 10)
			if !errors.Is(err, tt.want) || errors.Is(err, nsightsystems.ErrUnavailable) != (tt.want == nsightsystems.ErrUnavailable) {
				t.Fatalf("table status: %v", err)
			}
		})
	}
}

func TestNSYSConsumerRetainsAcquiredProviderError(t *testing.T) {
	for _, test := range []struct {
		name     string
		canceled bool
	}{
		{name: "provider error"},
		{name: "provider error with cancellation", canceled: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "header.sqlite")
			if err := os.WriteFile(path, []byte("SQLite format 3\x00"), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			providerErr := errors.New("acquired SQLite provider failure")
			calls := 0
			process := func(ctx context.Context, _ string, _ []string, _, _ io.Writer) error {
				calls++
				if test.canceled {
					cancel()
				}
				return errors.Join(providerErr, ctx.Err())
			}
			var output, log bytes.Buffer
			err := runContext(ctx, []string{
				"-nsys", path, "-nsys-release", "2024.2",
				"-nsys-build", "synthetic-error-control", "-nsys-clock", "report-relative",
			}, &output, &log, process)
			if calls != 1 || output.Len() != 0 || !errors.Is(err, providerErr) || errors.Is(err, context.Canceled) != test.canceled {
				t.Fatalf("consumer error custody: calls=%d output=%d provider=%t canceled=%t err=%v", calls, output.Len(), errors.Is(err, providerErr), errors.Is(err, context.Canceled), err)
			}
		})
	}
}

func TestNSYSFixedReadOnlyArguments(t *testing.T) {
	var args []string
	r := faultReader(t, func(_ context.Context, _ string, a []string, out, _ io.Writer) error {
		args = append([]string(nil), a...)
		_, err := io.WriteString(out, "[]")
		return err
	})
	if _, err := r.ReadTable(context.Background(), "StringIds", 10); !errors.Is(err, nsightsystems.ErrUnavailable) {
		t.Fatal(err)
	}
	prefix := []string{"-readonly", "-batch", "-init", "/dev/null", "-json", "-cmd", "PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;", "--"}
	if !reflect.DeepEqual(args[:len(prefix)], prefix) || !strings.HasPrefix(args[8], "file:") || !strings.HasSuffix(args[8], "?mode=ro&immutable=1") || args[9] != `SELECT type,sql FROM sqlite_schema WHERE name='StringIds' LIMIT 2` {
		t.Fatalf("unsafe argv: %q", args)
	}
	if _, err := r.ReadTable(context.Background(), "StringIds'; DROP TABLE StringIds;--", 10); err == nil {
		t.Fatal("accepted report-directed identifier")
	}
}

func TestNSYSSnapshotChanged(t *testing.T) {
	r := faultReader(t, func(context.Context, string, []string, io.Writer, io.Writer) error { return nil })
	if err := os.Chmod(r.path, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(r.path, []byte("different contents"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := r.verify(context.Background()); err == nil {
		t.Fatal("accepted changed snapshot")
	}
}

func TestNSYSProjection(t *testing.T) {
	responses := []string{
		`[{"type":"table","sql":"CREATE TABLE StringIds(id INTEGER,value TEXT)"}]`,
		`[{"name":"id","type":"INTEGER","notnull":1},{"name":"value","type":"TEXT","notnull":1}]`,
		`[{"Full":1,"Selected":1}]`,
		`[{"__profile2trace_rowid":7,"id":42,"value":"literal; .shell inert"}]`,
	}
	calls := 0
	r := faultReader(t, func(_ context.Context, _ string, _ []string, out, _ io.Writer) error {
		if calls >= len(responses) {
			t.Fatal("extra query")
		}
		_, err := io.WriteString(out, responses[calls])
		calls++
		return err
	})
	table, err := r.ReadTable(context.Background(), "StringIds", 10)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]json.RawMessage
	if len(table.Rows) != 1 {
		t.Fatal(table)
	}
	if err := json.Unmarshal(table.Rows[0], &got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || string(got["id"]) != "42" || r.keys["StringIds"][0] != 7 || table.DatabaseSHA256 != r.digest {
		t.Fatalf("projection changed or identity absent: %s %+v", table.Rows[0], r)
	}
}
