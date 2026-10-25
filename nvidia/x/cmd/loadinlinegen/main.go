// Command loadinlinegen writes a single-file torch load_inline submission
// from a template and CUDA sources.
//
// Usage:
//
//	loadinlinegen [-o file] [-name prefix] [-D NAME=VALUE]... [-set KEY=VALUE]... template source...
//
// The template is a Go text/template of the Python file. A source is a
// CUDA file, or FILE.py:NAME for the top-level string assignment NAME in
// another submission, such as a kernel another owner ships. The template
// sees:
//
//	.CUDA  the -D defines, then each source after a "// ---- source ----" line
//	.Tag   the first 10 hex digits of the SHA-256 of the template, .CUDA and the -set values
//	.Name  prefix_Tag, an extension name that changes whenever the code does,
//	       so torch never reuses a stale build of an older kernel
//	.KEY   each -set value
//
// A template embeds .CUDA in a raw triple-quoted string, either kind:
//
//	CUDA_SRC = r"""
//	{{.CUDA}}
//	"""
//
// so the command refuses sources that contain """ or ”' or end in a
// backslash. Referring to a field the command did not set is an error.
// After writing the output, loadinlinegen parses it with Python (-python)
// and fails unless one of its string constants holds .CUDA exactly: the
// check that the embedding survived Python's quoting.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"text/template"
)

//go:embed pyast.py
var pyAST []byte

type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, ",") }
func (l *listFlag) Set(v string) error { *l = append(*l, v); return nil }

type config struct {
	out, name, python string
	defines, sets     []string
	tmpl              string
	sources           []string
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("loadinlinegen: ")
	var c config
	flag.StringVar(&c.out, "o", "submission.py", "output `file`")
	flag.StringVar(&c.name, "name", "ext", "extension name `prefix`")
	flag.StringVar(&c.python, "python", "python3", "Python used to read FILE.py:NAME sources and check the output")
	flag.Var((*listFlag)(&c.defines), "D", "add `NAME=VALUE` as a #define ahead of the sources (repeatable)")
	flag.Var((*listFlag)(&c.sets), "set", "set template field `KEY=VALUE` (repeatable)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: loadinlinegen [flags] template source...\n")
		flag.PrintDefaults()
		os.Exit(2)
	}
	flag.Parse()
	if flag.NArg() < 2 {
		flag.Usage()
	}
	c.tmpl, c.sources = flag.Arg(0), flag.Args()[1:]
	name, err := run(context.Background(), c)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("wrote %s (extension %s)\n", c.out, name)
}

func run(ctx context.Context, c config) (string, error) {
	tmplText, err := os.ReadFile(c.tmpl)
	if err != nil {
		return "", err
	}
	data := map[string]any{}
	for _, s := range c.sets {
		k, v, ok := strings.Cut(s, "=")
		if !ok || k == "" {
			return "", fmt.Errorf("-set %q: want KEY=VALUE", s)
		}
		if k == "CUDA" || k == "Tag" || k == "Name" {
			return "", fmt.Errorf("-set %s: the command sets that field", k)
		}
		data[k] = v
	}
	var cuda strings.Builder
	for _, d := range c.defines {
		k, v, _ := strings.Cut(d, "=")
		if k == "" {
			return "", fmt.Errorf("-D %q: want NAME=VALUE", d)
		}
		fmt.Fprintf(&cuda, "#define %s %s\n", k, v)
	}
	for _, src := range c.sources {
		text, err := readSource(ctx, c.python, src)
		if err != nil {
			return "", err
		}
		if err := embeddable(text); err != nil {
			return "", fmt.Errorf("%s: %w", src, err)
		}
		fmt.Fprintf(&cuda, "// ---- %s ----\n%s", filepath.Base(src), text)
		if !strings.HasSuffix(text, "\n") {
			cuda.WriteString("\n")
		}
	}
	data["CUDA"] = strings.TrimSuffix(cuda.String(), "\n")

	h := sha256.New()
	h.Write(tmplText)
	h.Write([]byte(cuda.String()))
	sets := slices.Clone(c.sets)
	slices.Sort(sets)
	for _, s := range sets {
		fmt.Fprintf(h, "\x00%s", s)
	}
	tag := fmt.Sprintf("%x", h.Sum(nil))[:10]
	data["Tag"], data["Name"] = tag, c.name+"_"+tag

	t, err := template.New(filepath.Base(c.tmpl)).Option("missingkey=error").Parse(string(tmplText))
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		return "", err
	}
	if err := os.WriteFile(c.out, buf.Bytes(), 0o644); err != nil {
		return "", err
	}
	if err := check(ctx, c.python, c.out, data["CUDA"].(string)); err != nil {
		return "", err
	}
	return data["Name"].(string), nil
}

// embeddable reports whether text can sit inside a raw triple-quoted
// Python string of either kind and come out unchanged.
func embeddable(text string) error {
	for _, q := range []string{`"""`, `'''`} {
		if i := strings.Index(text, q); i >= 0 {
			return fmt.Errorf("line %d: contains %s, which would end the Python string", 1+strings.Count(text[:i], "\n"), q)
		}
	}
	if strings.HasSuffix(strings.TrimRight(text, "\n"), `\`) {
		return errors.New("ends in a backslash, which would escape the closing quote")
	}
	return nil
}

// readSource returns a CUDA file's text, or for FILE.py:NAME the value of
// that top-level string assignment.
func readSource(ctx context.Context, python, src string) (string, error) {
	file, name, ok := strings.Cut(src, ".py:")
	if !ok {
		b, err := os.ReadFile(src)
		return string(b), err
	}
	out, err := runPython(ctx, python, nil, "extract", file+".py", name)
	if err != nil {
		return "", fmt.Errorf("%s: %w", src, err)
	}
	return out, nil
}

// check parses the written file and requires a string constant that
// holds cuda exactly.
func check(ctx context.Context, python, path, cuda string) error {
	_, err := runPython(ctx, python, strings.NewReader(cuda), "check", path)
	var ee *exec.ExitError
	if errors.As(err, &ee) && ee.ExitCode() == 3 {
		return fmt.Errorf("%s: no Python string in it holds the CUDA source unchanged; embed {{.CUDA}} in a raw triple-quoted string", path)
	}
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func runPython(ctx context.Context, python string, stdin *strings.Reader, args ...string) (string, error) {
	script, err := os.CreateTemp("", "loadinlinegen-*.py")
	if err != nil {
		return "", err
	}
	defer os.Remove(script.Name())
	if _, err := script.Write(pyAST); err != nil {
		return "", err
	}
	if err := script.Close(); err != nil {
		return "", err
	}
	cmd := exec.CommandContext(ctx, python, append([]string{script.Name()}, args...)...)
	if stdin != nil {
		cmd.Stdin = stdin
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return "", fmt.Errorf("%w: %s", err, msg)
		}
		return "", err
	}
	return stdout.String(), nil
}
