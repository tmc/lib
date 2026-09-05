package nvjpeg2k

import (
	"errors"
	"runtime"
)

// Parse parses a JPEG 2000 bitstream held in host memory into bs.
//
// data is read during the call and is not retained: the parsed metadata
// belongs to bs and stays valid until bs is parsed again or destroyed. The
// length the library receives is derived from data, so the pointer and the
// length cannot disagree.
//
// Parse wraps nvjpeg2kStreamParse. That entry point also takes save_metadata
// and save_stream, which the header documents as reserved and requires to be
// 0; they are not exposed and are always passed as 0.
func (h Handle) Parse(data []byte, bs Bitstream) error {
	if len(data) == 0 {
		return errors.New("nvjpeg2k: Parse: empty bitstream buffer")
	}
	if bs == 0 {
		return errors.New("nvjpeg2k: Parse: zero Bitstream handle")
	}
	if err := checkLibSym("nvjpeg2kStreamParse"); err != nil {
		return err
	}
	err := toError(fnStreamParse(h, &data[0], uint(len(data)), 0, 0, bs))
	runtime.KeepAlive(data)
	return err
}
