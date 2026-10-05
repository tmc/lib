package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudahotswap/httpapi"
)

func isClientCommand(args []string) bool {
	if len(args) == 0 {
		return false
	}
	switch args[0] {
	case "push", "list", "revert", "status", "tail", "demo":
		return true
	}
	for _, arg := range args {
		if arg == "-addr" || strings.HasPrefix(arg, "-addr=") {
			return true
		}
	}
	return false
}

func runClient(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return errors.New("client command required")
	}
	command := args[0]
	if command == "demo" {
		return runDemo(ctx, args[1:], stdout, stderr)
	}
	if strings.HasPrefix(command, "-") {
		command = "watch"
	} else {
		args = args[1:]
	}
	f := flag.NewFlagSet("cudahotswap "+command, flag.ContinueOnError)
	f.SetOutput(stderr)
	address := f.String("addr", "unix:/var/run/cudahotswap.sock", "unix:path or tcp:host:port")
	name := f.String("name", "kernel", "registered kernel name")
	label := f.String("label", "client", "caller label")
	file := f.String("source", "", "CUDA source file")
	entry := f.String("entry", "replay", "kernel entry")
	watch := f.String("watch", "", "local CUDA file to watch")
	since := f.Uint64("since", 0, "report cursor")
	if err := f.Parse(args); err != nil {
		return err
	}
	if f.NArg() != 0 {
		return errors.New("unexpected positional arguments")
	}
	client, err := httpapi.NewClient(*address)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(stdout)
	if command == "tail" {
		stream, err := client.Events(ctx, *since)
		if err != nil {
			return err
		}
		defer stream.Close()
		for {
			frame, err := stream.Read()
			if err != nil {
				return err
			}
			if err := enc.Encode(frame); err != nil {
				return err
			}
		}
	}
	q := httpapi.Request{Name: *name, Label: *label}
	push := func(path string) (string, error) {
		b, err := readClientSource(path)
		if err != nil {
			return "", err
		}
		q.Source = cudahotswap.Source{Code: string(b), Filename: path, Entry: *entry}
		res, err := client.Call(ctx, "push", q)
		if err != nil {
			return "", err
		}
		if err := enc.Encode(res); err != nil {
			return "", err
		}
		if res.Failure != nil {
			return string(b), errors.New(res.Failure.Message)
		}
		return string(b), nil
	}
	if command == "watch" || *watch != "" {
		path := *watch
		if path == "" {
			return errors.New("-watch requires a local file")
		}
		last := ""
		have := false
		timer := time.NewTicker(100 * time.Millisecond)
		defer timer.Stop()
		for {
			b, err := readClientSource(path)
			if err != nil {
				return err
			}
			if !have || last != string(b) {
				_, err = push(path)
				if err != nil {
					fmt.Fprintln(stderr, err)
				}
				last = string(b)
				have = true
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-timer.C:
			}
		}
	}
	if command == "push" {
		_, err := push(*file)
		return err
	}
	if command != "list" && command != "revert" && command != "status" {
		return errors.New("unsupported client command")
	}
	res, err := client.Call(ctx, command, q)
	if err != nil {
		return err
	}
	if err := enc.Encode(res); err != nil {
		return err
	}
	if res.Failure != nil {
		return errors.New(res.Failure.Message)
	}
	return nil
}

func readClientSource(path string) ([]byte, error) {
	f, err := openClientSource(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("source must be a regular file")
	}
	b, err := io.ReadAll(io.LimitReader(f, (8<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(b) > 8<<20 {
		return nil, errors.New("source exceeds 8 MiB")
	}
	return b, nil
}
