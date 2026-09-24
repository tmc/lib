package main

import (
	"errors"
	"flag"
	"log"
	"os"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("trainstat: ")
	if err := run(os.Args[1:]); err != nil {
		var exitErr *exitCodeError
		if errors.As(err, &exitErr) {
			if !exitErr.Printed {
				log.Print(exitErr)
			}
			os.Exit(exitErr.Code)
		}
		log.Fatal(err)
	}
}

type exitCodeError struct {
	Code    int
	Err     error
	Printed bool // Err has already been printed
}

func (e *exitCodeError) Error() string { return e.Err.Error() }
func (e *exitCodeError) Unwrap() error { return e.Err }

// usageError marks err as a usage error, which exits with status 2.
func usageError(err error) error {
	return &exitCodeError{Code: 2, Err: err}
}

// parseFlags parses args with fs, which must use flag.ContinueOnError.
// On failure fs has printed the error and usage, so the returned usage
// error is not printed again.
func parseFlags(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		return &exitCodeError{Code: 2, Err: err, Printed: true}
	}
	return nil
}

func run(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "jsonl":
			return runJSONL(args[1:])
		case "repro":
			return runRepro(args[1:], os.Stdout)
		case "gate":
			return runGate(args[1:], os.Stdout)
		}
	}
	return runSummary(args, os.Stdout)
}
