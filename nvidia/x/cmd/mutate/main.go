// Command mutate runs a test command against source mutations and reports
// which ones the tests caught.
//
// Usage:
//
//	mutate [-dir D] [-timeout T] [-json] mutations.json -- command [args...]
//
// The mutations file is a JSON array. Each mutation replaces old with new in
// one file, or applies a list of such edits in order:
//
//	[
//	  {"name": "no_barrier", "file": "tile.go", "old": "__syncthreads();\n", "new": ""},
//	  {"name": "shallow_wait", "edits": [
//	    {"file": "tile.go", "old": "wait<STAGES - 2>", "new": "wait<STAGES - 1>"},
//	    {"file": "wgmma.go", "old": "wait<STAGES - 2>", "new": "wait<STAGES - 1>"}
//	  ], "want": "TestTile"}
//	]
//
// Each old must occur exactly once in its file at the point the edit
// applies; anything else is a setup error, because a mutation that edits
// nothing reports a survivor that is not one. Paths are relative to -dir,
// where the command also runs.
//
// The command first runs unmutated and must pass. Then, for each mutation,
// mutate writes the edits, runs the command, and restores the original
// bytes. A mutation is caught when the command fails; with "want", only
// when its output also matches that regular expression, so that a
// mutation which merely breaks the build can be told apart from one a
// specific test detects. The pattern is matched against the whole output;
// reports carry only its last 20 lines. A command that exceeds -timeout is killed with its
// process group and counts as caught.
//
// Files are restored after every mutation and on interrupt. If a file
// changes during a run other than by mutate, it is left alone and the run
// stops: something else is editing the tree.
//
// The exit status is 0 if every mutation was caught, 1 if any survived or
// failed without matching its want pattern, and 2 for a setup error.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"text/tabwriter"
	"time"
)

// Status values.
const (
	Caught    = "caught"
	Survived  = "survived"
	Unmatched = "unmatched" // failed, but the output did not match want
	TimedOut  = "timeout"   // killed at -timeout; counts as caught
)

type edit struct {
	File string `json:"file"`
	Old  string `json:"old"`
	New  string `json:"new"`
}

type mutation struct {
	Name  string `json:"name"`
	File  string `json:"file,omitempty"`
	Old   string `json:"old,omitempty"`
	New   string `json:"new,omitempty"`
	Edits []edit `json:"edits,omitempty"`
	Want  string `json:"want,omitempty"`

	want *regexp.Regexp
}

type result struct {
	Name     string  `json:"name"`
	Status   string  `json:"status"`
	ExitCode int     `json:"exit_code"`
	Seconds  float64 `json:"seconds"`
	Output   string  `json:"output,omitempty"` // last lines of combined output

	full string // all of the combined output, for matching want
}

type report struct {
	Command   []string `json:"command"`
	Dir       string   `json:"dir"`
	Baseline  result   `json:"baseline"`
	Mutations []result `json:"mutations"`
	Caught    int      `json:"caught"`
	Survived  int      `json:"survived"`
	Unmatched int      `json:"unmatched"`
}

func main() {
	code, err := run(os.Args[1:], os.Stdout, os.Stderr)
	if err != nil {
		fmt.Fprintln(os.Stderr, "mutate:", err)
	}
	os.Exit(code)
}

func run(args []string, stdout, stderr io.Writer) (int, error) {
	fs := flag.NewFlagSet("mutate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", ".", "directory the paths are relative to and the command runs in")
	timeout := fs.Duration("timeout", 10*time.Minute, "limit for each run of the command")
	asJSON := fs.Bool("json", false, "print JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: mutate [flags] mutations.json -- command [args...]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0, nil
		}
		return 2, nil
	}
	rest := fs.Args()
	if len(rest) < 3 || rest[1] != "--" || *timeout <= 0 {
		fs.Usage()
		return 2, nil
	}
	muts, err := load(rest[0])
	if err != nil {
		return 2, err
	}
	absDir, err := filepath.Abs(*dir)
	if err != nil {
		return 2, err
	}
	r := &runner{dir: absDir, argv: rest[2:], timeout: *timeout}
	rep, err := r.all(muts)
	if rep != nil {
		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			enc.Encode(rep)
		} else {
			printText(stdout, rep)
		}
	}
	if err != nil {
		return 2, err
	}
	if rep.Survived > 0 || rep.Unmatched > 0 {
		return 1, nil
	}
	return 0, nil
}

// load reads and checks the mutations file.
func load(path string) ([]*mutation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var muts []*mutation
	if err := dec.Decode(&muts); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if len(muts) == 0 {
		return nil, fmt.Errorf("%s: no mutations", path)
	}
	seen := map[string]bool{}
	for i, m := range muts {
		if m.Name == "" {
			return nil, fmt.Errorf("mutation %d: no name", i)
		}
		if seen[m.Name] {
			return nil, fmt.Errorf("mutation %s: duplicate name", m.Name)
		}
		seen[m.Name] = true
		if m.File != "" || m.Old != "" || m.New != "" {
			if len(m.Edits) > 0 {
				return nil, fmt.Errorf("mutation %s: both file/old/new and edits", m.Name)
			}
			m.Edits = []edit{{File: m.File, Old: m.Old, New: m.New}}
		}
		if len(m.Edits) == 0 {
			return nil, fmt.Errorf("mutation %s: no edits", m.Name)
		}
		for j, e := range m.Edits {
			if e.File == "" || e.Old == "" || e.Old == e.New {
				return nil, fmt.Errorf("mutation %s edit %d: want a file, a nonempty old, and a new that differs", m.Name, j)
			}
		}
		if m.Want != "" {
			if m.want, err = regexp.Compile(m.Want); err != nil {
				return nil, fmt.Errorf("mutation %s: want: %w", m.Name, err)
			}
		}
	}
	return muts, nil
}

type runner struct {
	dir     string
	argv    []string
	timeout time.Duration

	mu      sync.Mutex
	pending map[string]restore // files currently mutated
}

// A restore is what a mutated file must go back to, and what mutate wrote,
// so that a change by someone else can be told from mutate's own.
type restore struct {
	orig, wrote []byte
	mode        os.FileMode
}

// all checks every mutation, runs the baseline, and runs each mutation.
func (r *runner) all(muts []*mutation) (*report, error) {
	// Check every edit applies before running anything.
	for _, m := range muts {
		if _, err := r.apply(m); err != nil {
			return nil, err
		}
	}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(stop)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	interrupted := make(chan struct{})
	go func() {
		select {
		case <-stop:
			cancel()
			close(interrupted)
		case <-ctx.Done():
		}
	}()

	rep := &report{Command: r.argv, Dir: r.dir}
	rep.Baseline = r.exec(ctx, "baseline")
	if rep.Baseline.ExitCode != 0 {
		return rep, fmt.Errorf("baseline failed (%s, exit %d); mutations cannot be judged", rep.Baseline.Status, rep.Baseline.ExitCode)
	}
	for _, m := range muts {
		files, err := r.apply(m)
		if err != nil {
			return rep, err
		}
		if err := r.write(files); err != nil {
			r.restoreAll()
			return rep, err
		}
		res := r.exec(ctx, m.Name)
		if err := r.restoreAll(); err != nil {
			return rep, err
		}
		select {
		case <-interrupted:
			return rep, errors.New("interrupted; files restored")
		default:
		}
		switch {
		case res.Status == TimedOut:
			rep.Caught++
		case res.ExitCode == 0:
			res.Status = Survived
			rep.Survived++
		case m.want != nil && !m.want.MatchString(res.full):
			res.Status = Unmatched
			rep.Unmatched++
		default:
			res.Status = Caught
			rep.Caught++
		}
		if res.Status == Caught {
			res.Output = ""
		}
		rep.Mutations = append(rep.Mutations, res)
	}
	return rep, nil
}

// apply computes the mutated contents of each file m edits, from the files
// as they are now, without writing anything.
func (r *runner) apply(m *mutation) (map[string]restore, error) {
	files := map[string]restore{}
	cur := map[string][]byte{}
	for j, e := range m.Edits {
		path := filepath.Join(r.dir, e.File)
		data, ok := cur[path]
		if !ok {
			info, err := os.Stat(path)
			if err != nil {
				return nil, fmt.Errorf("mutation %s: %w", m.Name, err)
			}
			if data, err = os.ReadFile(path); err != nil {
				return nil, fmt.Errorf("mutation %s: %w", m.Name, err)
			}
			files[path] = restore{orig: data, mode: info.Mode().Perm()}
		}
		if n := bytes.Count(data, []byte(e.Old)); n != 1 {
			return nil, fmt.Errorf("mutation %s edit %d: old text occurs %d times in %s, want 1", m.Name, j, n, e.File)
		}
		cur[path] = bytes.Replace(data, []byte(e.Old), []byte(e.New), 1)
	}
	changed := false
	for path, data := range cur {
		f := files[path]
		f.wrote = data
		files[path] = f
		changed = changed || !bytes.Equal(f.orig, data)
	}
	if !changed {
		return nil, fmt.Errorf("mutation %s: edits leave every file unchanged", m.Name)
	}
	return files, nil
}

func (r *runner) write(files map[string]restore) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pending = files
	for path, f := range files {
		if err := os.WriteFile(path, f.wrote, f.mode); err != nil {
			return err
		}
	}
	return nil
}

// restoreAll puts every mutated file back. A file whose contents are no
// longer what mutate wrote was changed by someone else and is not touched.
func (r *runner) restoreAll() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	var errs []error
	for path, f := range r.pending {
		now, err := os.ReadFile(path)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if !bytes.Equal(now, f.wrote) {
			errs = append(errs, fmt.Errorf("%s changed during the run; left as found, original not restored", path))
			continue
		}
		if err := os.WriteFile(path, f.orig, f.mode); err != nil {
			errs = append(errs, err)
			continue
		}
		if back, err := os.ReadFile(path); err != nil || !bytes.Equal(back, f.orig) {
			errs = append(errs, fmt.Errorf("%s: restore did not take", path))
		}
	}
	r.pending = nil
	return errors.Join(errs...)
}

// exec runs the command once.
func (r *runner) exec(ctx context.Context, name string) result {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, r.argv[0], r.argv[1:]...)
	cmd.Dir = r.dir
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	cmd.WaitDelay = 5 * time.Second
	setProcessGroup(cmd)
	start := time.Now()
	err := cmd.Run()
	res := result{Name: name, Seconds: time.Since(start).Seconds(), Output: tail(out.String(), 20), full: out.String()}
	var ee *exec.ExitError
	switch {
	case ctx.Err() == context.DeadlineExceeded:
		res.Status, res.ExitCode = TimedOut, -1
	case err == nil:
		res.Status = "ok"
	case errors.As(err, &ee):
		res.Status, res.ExitCode = "failed", ee.ExitCode()
	default:
		res.Status, res.ExitCode = "failed", -1
		res.Output = err.Error() + "\n" + res.Output
	}
	return res
}

func tail(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

func printText(w io.Writer, rep *report) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "mutation\tstatus\texit\tseconds")
	fmt.Fprintf(tw, "(baseline)\t%s\t%d\t%.1f\n", rep.Baseline.Status, rep.Baseline.ExitCode, rep.Baseline.Seconds)
	for _, m := range rep.Mutations {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%.1f\n", m.Name, m.Status, m.ExitCode, m.Seconds)
	}
	tw.Flush()
	for _, m := range rep.Mutations {
		if m.Status != Caught && m.Status != TimedOut {
			fmt.Fprintf(w, "\n--- %s %s; last output:\n%s\n", m.Name, m.Status, m.Output)
		}
	}
	fmt.Fprintf(w, "\n%d caught, %d survived, %d unmatched of %d\n", rep.Caught, rep.Survived, rep.Unmatched, len(rep.Mutations))
}
