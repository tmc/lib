package cudaplan

import "context"

// ExportNewHostCompletion exports newHostCompletion for package-external unit testing.
func ExportNewHostCompletion(waitFn func(context.Context) error, destroyFn func() error, releases ...func()) *Completion {
	return newHostCompletion(waitFn, destroyFn, releases...)
}
