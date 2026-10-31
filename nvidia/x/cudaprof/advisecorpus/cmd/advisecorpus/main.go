// Command advisecorpus consumes retained trajectory files and an optional exact
// advice sidecar. Without a sidecar it reports NOT_EVALUATED, never a missed lead.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cudaprof/advisecorpus"
)

func run() error {
	root := flag.String("root", ".", "base directory for relative artifact paths")
	linkName := flag.String("link", "", "existing corpus-advise-link/v1 JSON sidecar")
	flag.Parse()
	if flag.NArg() == 0 {
		return fmt.Errorf("provide retained NDJSON trajectory files")
	}
	var link *advisecorpus.Link
	if *linkName != "" {
		f, err := os.Open(*linkName)
		if err != nil {
			return err
		}
		link, err = advisecorpus.DecodeLink(f)
		f.Close()
		if err != nil {
			return err
		}
	}
	var results []advisecorpus.Result
	matched := false
	seen := map[string]bool{}
	for _, name := range flag.Args() {
		f, err := os.Open(name)
		if err != nil {
			return err
		}
		records, err := advisecorpus.Decode(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		for _, record := range records {
			var id string
			if err := json.Unmarshal(record["id"], &id); err != nil {
				return err
			}
			if seen[id] {
				return fmt.Errorf("duplicate record id %q", id)
			}
			seen[id] = true
			var selected *advisecorpus.Link
			if link != nil && id == link.RecordID {
				selected = link
				matched = true
			}
			result, err := advisecorpus.Consume(*root, record, selected)
			if err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			results = append(results, result)
		}
	}
	if link != nil && !matched {
		return fmt.Errorf("advice link record %q not found", link.RecordID)
	}
	encoder := json.NewEncoder(os.Stdout)
	for _, result := range results {
		if err := encoder.Encode(result); err != nil {
			return err
		}
	}
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
