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
			if !errors.Is(exitErr.Err, flag.ErrHelp) {
				log.Print(exitErr)
			}
			os.Exit(exitErr.Code)
		}
		log.Fatal(err)
	}
}

type exitCodeError struct {
	Code int
	Err  error
}

func (e *exitCodeError) Error() string { return e.Err.Error() }
func (e *exitCodeError) Unwrap() error { return e.Err }

// usageError marks err as a usage error, which exits with status 2.
func usageError(err error) error {
	return &exitCodeError{Code: 2, Err: err}
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
