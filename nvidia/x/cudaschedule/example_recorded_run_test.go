package cudaschedule

import (
	"context"
	"fmt"
)

func ExampleRunRecorded() {
	_, err := RunRecorded(context.Background(), 0, "", RecordedRunOptions{})
	fmt.Println(err != nil)
	// Output: true
}
func ExampleRunRecordedHelper() {
	err := RunRecordedHelper(context.Background(), nil)
	fmt.Println(err)
	// Output: recorded helper: expected keeper or supervisor, config path and digest
}
