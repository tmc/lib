package cudahotswap

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

func TestServiceErr(t *testing.T) {
	cause := errors.New("device fault")
	ready := settingsExampleService()
	t.Cleanup(func() { ready.Close(context.Background()) })
	for _, tt := range []struct {
		name     string
		s        *Service
		wantErr  bool
		poisoned bool
	}{
		{"nil", nil, true, false},
		{"zero", &Service{}, true, false},
		{"available", ready, false, false},
		{"closed", &Service{closed: true}, true, false},
		{"poisoned", &Service{poison: cause}, true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.s.Err()
			if (err != nil) != tt.wantErr || errors.Is(err, ErrPoisoned) != tt.poisoned {
				t.Fatalf("Err = %v, want error %t, poisoned %t", err, tt.wantErr, tt.poisoned)
			}
			if tt.poisoned && !errors.Is(err, cause) {
				t.Fatal("original device error lost")
			}
		})
	}
	s := &Service{poison: cause}
	_, _, _, err := s.watchSnapshot()
	if !errors.Is(err, ErrPoisoned) || !errors.Is(err, cause) {
		t.Fatalf("watch error lost classification or cause: %v", err)
	}
}

func ExampleService_Err() {
	s := settingsExampleService()
	fmt.Println(s.Err())
	s.Close(context.Background())
	fmt.Println(s.Err())
	// Output:
	// <nil>
	// service is closed
}
