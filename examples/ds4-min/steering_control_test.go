package main

import (
	"errors"
	"reflect"
	"testing"
)

func TestSteeringControlGraphTransition(t *testing.T) {
	for _, tt := range []struct {
		name      string
		old, next float32
		failure   string
		want      []string
	}{
		{"zero-enable", 0, 1, "", []string{"sync", "invalidate", "publish"}},
		{"change", 1, -1, "", []string{"sync", "invalidate", "publish"}},
		{"disable", 1, 0, "", []string{"sync", "invalidate", "publish"}},
		{"same", 1, 1, "", []string{"sync", "publish"}},
		{"sync-failure", 0, 1, "sync", []string{"sync"}},
		{"graph-failure", 0, 1, "invalidate", []string{"sync", "invalidate"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var events []string
			value := tt.old
			failure := errors.New("fixture failure")
			callback := func(name string) func() error {
				return func() error {
					events = append(events, name)
					if tt.failure == name {
						return failure
					}
					return nil
				}
			}
			err := changeSteeringFFN(tt.old, tt.next, callback("sync"), callback("invalidate"), func(v float32) { events = append(events, "publish"); value = v })
			if (err != nil) != (tt.failure != "") || !reflect.DeepEqual(events, tt.want) {
				t.Fatalf("events %v error %v", events, err)
			}
			if tt.failure != "" && value != tt.old {
				t.Fatal("failed control transition published scale")
			}
		})
	}
}
