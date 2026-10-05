package cudahotswap

import (
	"context"
	"fmt"
)

func ExampleService_Subscribe() {
	s := settingsExampleService() // Host fake loader and lease.
	defer s.Close(context.Background())
	sub, err := s.Subscribe(context.Background(), 0)
	fmt.Println(err)
	if err == nil {
		defer sub.Close()
	}
	// Output: <nil>
}

func ExampleSubscription() {
	var sub Subscription
	for range sub.Events() {
	}
	fmt.Println(sub.Err())
	// Output: <nil>
}

func ExampleSubscription_Events() {
	var sub Subscription
	_, open := <-sub.Events()
	fmt.Println(open)
	// Output: false
}

func ExampleSubscription_Err() {
	var sub Subscription
	fmt.Println(sub.Err())
	// Output: <nil>
}

func ExampleSubscription_Close() {
	var sub Subscription
	fmt.Println(sub.Close())
	// Output: <nil>
}
