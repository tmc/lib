package continuousbatch

import (
	"context"
	"fmt"
	"testing"
	"time"
)

func BenchmarkServing(b *testing.B) {
	for _, concurrency := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("serial/c%d", concurrency), func(b *testing.B) {
			backend := &fakeBackend{prefill: 50 * time.Microsecond, decode: 100 * time.Microsecond}
			for n := 0; n < b.N; n++ {
				for request := 0; request < concurrency; request++ {
					state := State{ID: fmt.Sprint(request), Limit: 8}
					token, err := backend.Prefill(context.Background(), &state)
					if err != nil {
						b.Fatal(err)
					}
					state.Current = token.ID
					for generated := 1; generated < state.Limit; generated++ {
						result := backend.DecodeBatch(context.Background(), []*State{&state})[0]
						if result.Err != nil {
							b.Fatal(result.Err)
						}
						state.Current = result.Token.ID
					}
				}
			}
		})
		b.Run(fmt.Sprintf("scheduler/c%d", concurrency), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				backend := &fakeBackend{prefill: 50 * time.Microsecond, decode: 100 * time.Microsecond}
				s, err := New(backend, Config{MaxActive: concurrency, MaxQueue: concurrency, MaxBatch: concurrency, BatchWait: 50 * time.Microsecond, TokenBuffer: 8, PrefillWorkers: concurrency})
				if err != nil {
					b.Fatal(err)
				}
				responses := make([]*Response, concurrency)
				for request := range responses {
					responses[request], err = s.Submit(context.Background(), Request{ID: fmt.Sprintf("%d-%d", n, request), MaxTokens: 8})
					if err != nil {
						b.Fatal(err)
					}
				}
				for _, response := range responses {
					var tokens []Token
					for token := range response.Tokens {
						tokens = append(tokens, token)
					}
					if err := <-response.Done; err != nil || len(tokens) != 8 {
						b.Fatalf("tokens=%d err=%v", len(tokens), err)
					}
				}
				if err := s.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
