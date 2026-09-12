package main

import (
	"strings"
	"testing"
)

func TestReadCycleResults(t *testing.T) {
	const good = `{"Kind":"cycle","Cycle":1,"Result":{"Comparisons":8}}
{"Kind":"cycle","Cycle":2,"Result":{"Comparisons":8}}
{"Kind":"summary","Requested":2,"Completed":2,"Complete":true,"Error":""}`
	rows, err := readCycleResults(strings.NewReader(good), 2)
	if err != nil || len(rows) != 2 || rows[1].Comparisons != 8 {
		t.Fatalf("%v %v", rows, err)
	}
	for name, input := range map[string]string{
		"missing summary": strings.Split(good, `{"Kind":"summary"`)[0],
		"wrong order":     strings.Replace(good, `"Cycle":2`, `"Cycle":1`, 1),
		"missing result":  strings.Replace(good, `"Result":{"Comparisons":8}`, `"Result":null`, 1),
		"failed cleanup":  strings.Replace(good, `"Error":""`, `"Error":"cleanup failed"`, 1),
		"incomplete":      strings.Replace(good, `"Complete":true`, `"Complete":false`, 1),
		"wrong count":     strings.Replace(good, `"Completed":2`, `"Completed":1`, 1),
		"trailing record": good + ` {}`,
		"trailing junk":   good + ` bad`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := readCycleResults(strings.NewReader(input), 2); err == nil {
				t.Fatal("accepted invalid cycle stream")
			}
		})
	}
}

func TestReadSweepResults(t *testing.T) {
	const good = `{"Kind":"sample","Cycle":1,"SizeIndex":1,"Result":{"Case":{"Bytes":16}}}
{"Kind":"sample","Cycle":1,"SizeIndex":2,"Result":{"Case":{"Bytes":64}}}
{"Kind":"sample","Cycle":2,"SizeIndex":1,"Result":{"Case":{"Bytes":16}}}
{"Kind":"sample","Cycle":2,"SizeIndex":2,"Result":{"Case":{"Bytes":64}}}
{"Kind":"summary","RequestedCycles":2,"RequestedSamples":4,"CompletedSamples":4,"Complete":true,"Error":""}`
	rows, err := readSweepResults(strings.NewReader(good), []int{16, 64}, 2)
	if err != nil || len(rows) != 4 {
		t.Fatal(rows, err)
	}
	for name, input := range map[string]string{
		"wrong size":     strings.Replace(good, `"Bytes":64`, `"Bytes":16`, 1),
		"wrong cycle":    strings.Replace(good, `"Cycle":2`, `"Cycle":1`, 1),
		"wrong index":    strings.Replace(good, `"SizeIndex":2`, `"SizeIndex":1`, 1),
		"short summary":  strings.Replace(good, `"CompletedSamples":4`, `"CompletedSamples":3`, 1),
		"failed cleanup": strings.Replace(good, `"Error":""`, `"Error":"cleanup failed"`, 1),
		"truncated":      strings.Split(good, `{"Kind":"summary"`)[0],
		"extra":          good + ` {}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := readSweepResults(strings.NewReader(input), []int{16, 64}, 2); err == nil {
				t.Fatal("accepted invalid sweep")
			}
		})
	}
}
