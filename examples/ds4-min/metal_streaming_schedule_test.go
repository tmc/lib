package main

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func TestMetalStreamingLayerSchedule(t *testing.T) {
	for _, failure := range []string{"", "front", "stage", "consume", "release"} {
		t.Run(failure, func(t *testing.T) {
			var got []string
			active := false
			count := 0
			fault := errors.New("fault")
			visit := func(name string, layer, row int) error {
				got = append(got, fmt.Sprintf("%s:%d:%d", name, layer, row))
				if name == failure {
					return fault
				}
				return nil
			}
			err := runMetalStreamingRows(2, 3, func(layer, row int) error {
				if active {
					t.Fatal("frontier overlaps consumer lease")
				}
				count++
				return visit("front", layer, row)
			}, func(layer int) error {
				if count != 3 {
					t.Fatal("stage before complete routes")
				}
				active = true
				return visit("stage", layer, -1)
			}, func(layer, row int) error {
				if !active {
					t.Fatal("consumer without lease")
				}
				return visit("consume", layer, row)
			}, func(layer int) error {
				if !active {
					t.Fatal("release without consumer")
				}
				if err := visit("release", layer, -1); err != nil {
					return err
				}
				active, count = false, 0
				return nil
			})
			if failure != "" {
				if !errors.Is(err, fault) || len(got) == 0 || got[len(got)-1][:len(failure)] != failure {
					t.Fatal("fault lost or continued work", err, got)
				}
				return
			}
			want := []string{"front:0:0", "front:0:1", "front:0:2", "stage:0:-1", "consume:0:0", "consume:0:1", "consume:0:2", "release:0:-1", "front:1:0", "front:1:1", "front:1:2", "stage:1:-1", "consume:1:0", "consume:1:1", "consume:1:2", "release:1:-1"}
			if err != nil || active || !reflect.DeepEqual(got, want) {
				t.Fatal("layer-major schedule", err, active, got)
			}
		})
	}
}
