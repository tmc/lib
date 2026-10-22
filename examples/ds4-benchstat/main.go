package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
)

func main() {
	var (
		receipts = flag.Bool("receipt", false, "read oracle RESULT.json receipts instead of CSV")
		backend  = flag.String("backend", "", "backend the CSV was measured on (CSV mode)")
		host     = flag.String("host", "", "host the CSV was measured on (CSV mode)")
	)
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: ds4-benchstat [-backend name] [-host name] [file...]")
		fmt.Fprintln(os.Stderr, "       ds4-benchstat -receipt RESULT.json...")
		flag.PrintDefaults()
	}
	flag.Parse()
	if err := run(*receipts, *backend, *host, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "ds4-benchstat:", err)
		os.Exit(1)
	}
}

func run(receipts bool, backend, host string, args []string) error {
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	if receipts {
		if len(args) == 0 {
			return fmt.Errorf("-receipt needs at least one RESULT.json")
		}
		for _, path := range args {
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			cfg, rows, err := parseReceipt(data)
			if err != nil {
				return fmt.Errorf("%s: %w", path, err)
			}
			if err := emit(out, cfg, rows); err != nil {
				return err
			}
		}
		return nil
	}

	cfg := []config{{"pkg", "ds4.c"}}
	if backend != "" {
		cfg = append(cfg, config{"backend", backend})
	}
	if host != "" {
		cfg = append(cfg, config{"host", host})
	}
	if len(args) == 0 {
		rows, err := parseCSV(os.Stdin)
		if err != nil {
			return err
		}
		return emit(out, cfg, rows)
	}
	for _, path := range args {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		rows, err := parseCSV(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := emit(out, cfg, rows); err != nil {
			return err
		}
	}
	return nil
}

// emit writes one group and reports on stderr what it left out, so a caller
// never has to infer coverage from the line count.
func emit(w *bufio.Writer, cfg []config, rows []row) error {
	written, skipped, err := write(w, cfg, rows)
	if err != nil {
		return err
	}
	if skipped > 0 {
		fmt.Fprintf(os.Stderr, "ds4-benchstat: %d benchmarks written, %d measurements omitted as unreported\n", written, skipped)
	}
	fmt.Fprintf(os.Stderr, "ds4-benchstat: compare files with: benchstat %s a.txt b.txt\n", ignoreFlag())
	return w.Flush()
}
