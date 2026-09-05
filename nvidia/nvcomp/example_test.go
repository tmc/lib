package nvcomp_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvcomp"
)

// A zero options value is the vendor default, so an operation that wants LZ4's
// defaults does not have to name a field.
func ExampleCompressOpts() {
	var defaults nvcomp.CompressOpts
	fmt.Println(defaults.DataType, defaults.BitshuffleMode)

	// Bitshuffling is described by the vendor as a transform applied before
	// compression; it needs the element type of the data to be shuffled.
	shuffled := nvcomp.CompressOpts{
		DataType:       nvcomp.DataTypeFloat16,
		BitshuffleMode: nvcomp.BitshuffleMsbFirst,
	}
	fmt.Println(shuffled.DataType, shuffled.BitshuffleMode)

	// The same value has to be given to every call in one operation: the
	// temporary size, the alignments and the maximum output size are all
	// derived from it.
	// Output:
	// DataTypeChar BitshuffleNone
	// DataTypeFloat16 BitshuffleMsbFirst
}

// A batch reports each chunk's outcome only in the status array it writes in
// device memory. Copy that array back after the stream completes, then check
// every element: the error returned by the asynchronous call describes the
// launch, not the work.
func ExampleCheckStatuses() {
	// As copied back from a DeviceStatusArray of four chunks.
	statuses := []nvcomp.Result{
		nvcomp.Success,
		nvcomp.Success,
		nvcomp.ErrCannotDecompress,
		nvcomp.Success,
	}
	err := nvcomp.CheckStatuses(statuses)
	fmt.Println(err)

	var chunkErr *nvcomp.ChunkError
	if ok := asChunkError(err, &chunkErr); ok {
		fmt.Println("first bad chunk:", chunkErr.Chunk)
	}
	// Output:
	// nvcomp: 1 of 4 chunks failed; chunk 2: nvcomp: ErrCannotDecompress: the input could not be decompressed
	// first bad chunk: 2
}

// asChunkError is errors.As, spelled out so the example above reads as one
// thing rather than two.
func asChunkError(err error, target **nvcomp.ChunkError) bool {
	e, ok := err.(*nvcomp.ChunkError)
	if ok {
		*target = e
	}
	return ok
}

// Example shows the order of calls in one batched compression, and where each
// buffer has to live.
//
// It has no output because it cannot run here: every argument with a Device
// prefix is an address in device memory, so the sequence needs a CUDA
// allocator and a GPU. It compiles, which is what fixes the argument order and
// the types; it is not evidence that the sequence works. See pipeline_test.go
// for the same sequence written as a test, with its prerequisites named.
func Example() {
	// deviceAlloc stands in for whichever CUDA binding the program already
	// uses: this package neither allocates nor frees device memory. Every
	// address it returns must be freed by the caller, and must stay live until
	// the stream has completed.
	deviceAlloc := func(bytes int) uintptr { panic("supplied by the caller's CUDA binding") }
	deviceFree := func(uintptr) {}
	streamSynchronize := func(nvcomp.Stream) error { panic("supplied by the caller's CUDA binding") }
	copyToDevice := func(dst uintptr, src any) {}
	copyToHost := func(dst any, src uintptr) {}

	const (
		numChunks     = 8
		maxChunkBytes = 65536
	)
	stream := nvcomp.Stream(0) // the CUDA default stream
	opts := nvcomp.CompressOpts{}

	// 1. Ask for the alignments this options value requires, and honor them in
	//    every allocation below. The constants RequiredCompressionAlignment
	//    and friends only bound what this call can report.
	align, err := nvcomp.CompressGetRequiredAlignments(opts)
	if err != nil {
		panic(err)
	}
	_ = align

	// 2. Size the output buffers and the temporary workspace. Both queries
	//    take the same options value the compression call will take.
	maxOut, err := nvcomp.CompressGetMaxOutputChunkSize(maxChunkBytes, opts)
	if err != nil {
		panic(err)
	}
	tempBytes, err := nvcomp.CompressGetTempSizeAsync(numChunks, maxChunkBytes, opts, numChunks*maxChunkBytes)
	if err != nil {
		panic(err)
	}

	// 3. Allocate on the device. The two pointer arrays are the subtle part:
	//    each is an array of device addresses that itself lives in device
	//    memory, which is why they have their own type.
	inputPtrs := deviceAlloc(numChunks * nvcomp.SizeofDevicePointer)
	inputBytes := deviceAlloc(numChunks * nvcomp.SizeofDeviceSize)
	outputPtrs := deviceAlloc(numChunks * nvcomp.SizeofDevicePointer)
	outputBytes := deviceAlloc(numChunks * nvcomp.SizeofDeviceSize)
	statusArray := deviceAlloc(numChunks * nvcomp.SizeofDeviceStatus)
	temp := deviceAlloc(int(tempBytes))
	defer func() {
		for _, p := range []uintptr{inputPtrs, inputBytes, outputPtrs, outputBytes, statusArray, temp} {
			deviceFree(p)
		}
	}()

	// Fill the two input arrays with the device addresses and sizes of the
	// chunks, and the output array with the addresses of buffers of at least
	// maxOut bytes each.
	_ = maxOut
	copyToDevice(inputPtrs, []uintptr{})
	copyToDevice(inputBytes, []uint{})
	copyToDevice(outputPtrs, []uintptr{})

	// 4. Launch. A nil error here means the launch was accepted and nothing
	//    more. Everything allocated above must stay live and unmodified until
	//    the stream completes.
	err = nvcomp.CompressAsync(
		nvcomp.DevicePointerArray(inputPtrs),
		nvcomp.DeviceSizeArray(inputBytes),
		maxChunkBytes,
		numChunks,
		nvcomp.DeviceBuffer(temp),
		tempBytes,
		nvcomp.DevicePointerArray(outputPtrs),
		nvcomp.DeviceSizeArray(outputBytes),
		opts,
		nvcomp.DeviceStatusArray(statusArray),
		stream,
	)
	if err != nil {
		panic(err)
	}

	// 5. Wait for the stream, then read the per-chunk statuses. Skipping this
	//    turns a failed chunk into output that is silently wrong.
	if err := streamSynchronize(stream); err != nil {
		panic(err)
	}
	statuses := make([]nvcomp.Result, numChunks)
	copyToHost(statuses, statusArray)
	if err := nvcomp.CheckStatuses(statuses); err != nil {
		panic(err)
	}

	// 6. Read the compressed size of each chunk from outputBytes. The batched
	//    API records nothing else: the chunk count, the uncompressed sizes and
	//    the options all have to be kept by the caller to decompress later.
	sizes := make([]uint, numChunks)
	copyToHost(sizes, outputBytes)
}
