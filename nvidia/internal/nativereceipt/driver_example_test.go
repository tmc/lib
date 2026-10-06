package nativereceipt

import "fmt"

func ExampleSession_WriteDriver() {
	// A receipt requires a session created by New in the observed process.
	var session *Session
	fmt.Println(session.WriteDriver("/receipt.json"))
	// Output: native receipt: nil session
}

func ExampleReceipt_CheckDriver() {
	// A zero receipt cannot prove a producer or a prepared image.
	var receipt Receipt
	fmt.Println(receipt.CheckDriver(Process{}, "", ""))
	// Output: native receipt: producer process identity disagrees
}
