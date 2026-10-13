package nsightsystems

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// This is a test-only generic database adapter, not a public capture command.
// Every query name is checked against the fixed documented table allowlist.
// The database contains only the owned finite synthetic fixtures.
type sqliteFixtureReader struct{ tool, database string }

func (s sqliteFixtureReader) query(ctx context.Context, query string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, s.tool, "-readonly", "-batch", "-init", "/dev/null", "-json", s.database, query)
	cmd.WaitDelay = 200 * time.Millisecond
	return cmd.Output()
}

func (s sqliteFixtureReader) ReadTable(ctx context.Context, name string, limit int) (Table, error) {
	known := false
	for _, n := range tableNames {
		if name == n {
			known = true
		}
	}
	if !known || limit < 0 || limit > 262144 {
		return Table{}, fmt.Errorf("invalid fixture query")
	}
	out, err := s.query(ctx, `PRAGMA table_info("`+name+`");`)
	if err != nil {
		return Table{}, err
	}
	if len(out) == 0 {
		return Table{}, ErrUnavailable
	}
	var columns []struct {
		Name    string `json:"name"`
		Type    string `json:"type"`
		NotNull int    `json:"notnull"`
	}
	if err := json.Unmarshal(out, &columns); err != nil {
		return Table{}, err
	}
	if len(columns) == 0 {
		return Table{}, ErrUnavailable
	}
	body, err := os.ReadFile(s.database)
	if err != nil {
		return Table{}, err
	}
	hash := sha256.Sum256(body)
	t := Table{DatabaseSHA256: hex.EncodeToString(hash[:])}
	for _, c := range columns {
		t.Columns = append(t.Columns, Column{Name: c.Name, Type: c.Type, Nullable: c.NotNull == 0})
	}
	out, err = s.query(ctx, fmt.Sprintf(`SELECT * FROM "%s" LIMIT %d;`, name, limit+1))
	if err != nil {
		return Table{}, err
	}
	if len(out) == 0 {
		return t, nil
	}
	if len(out) > 64<<10 {
		return Table{}, fmt.Errorf("fixture output bound exceeded")
	}
	if err := json.Unmarshal(out, &t.Rows); err != nil {
		return Table{}, err
	}
	return t, nil
}

func TestActualSQLiteImport(t *testing.T) {
	tool, err := exec.LookPath("sqlite3")
	if err != nil {
		t.Skip("explicit sqlite3 schema fixture adapter unavailable")
	}
	for _, release := range []string{"2024.2", "2025.3"} {
		t.Run(release, func(t *testing.T) {
			schema, err := os.ReadFile("testdata/schema-" + release + ".sql")
			if err != nil {
				t.Fatal(err)
			}
			rows, err := os.ReadFile("testdata/rows-" + release + ".sql")
			if err != nil {
				t.Fatal(err)
			}
			db := filepath.Join(t.TempDir(), "synthetic.sqlite")
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, tool, "-batch", "-init", "/dev/null", db)
			cmd.Stdin = strings.NewReader(string(schema) + string(rows))
			cmd.WaitDelay = 200 * time.Millisecond
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("synthetic database: %v %s", err, out)
			}
			body, err := os.ReadFile(db)
			if err != nil {
				t.Fatal(err)
			}
			hash := sha256.Sum256(body)
			id := testSource(release)
			id.DatabaseSHA256 = hex.EncodeToString(hash[:])
			r, err := Import(ctx, sqliteFixtureReader{tool: tool, database: db}, id, Limits{})
			if err != nil {
				t.Fatal(err)
			}
			if r.Source.DatabaseSHA256 != id.DatabaseSHA256 || len(r.Events) != 6 || len(r.Joins) != 1 || r.Events[0].Name != "kernel<float>" || r.Events[4].Name != "nccl:AllReduce" || *r.Events[1].Bytes != 64 {
				t.Fatal("actual exported SQLite rows differ")
			}
			if after, err := os.ReadFile(db); err != nil || sha256.Sum256(after) != hash {
				t.Fatal("read-only database changed")
			}
		})
	}
}
