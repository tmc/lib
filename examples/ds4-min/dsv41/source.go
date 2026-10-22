package dsv41

import (
	"fmt"
	"io"
)

// VerifiedSource returns a range-checked read-only view of the exact source
// retained by BindVerified and its file size. It does not reverify an externally
// modified file. The caller must keep the original source immutable and open
// until all model consumers close. The returned view exposes no Close method.
func (m *Model) VerifiedSource() (io.ReaderAt, int64, error) {
	if m == nil || m.assetSHA == ([32]byte{}) || m.boundSource == nil || m.boundSize <= 0 {
		return nil, 0, fmt.Errorf("V4.1 source is not verified")
	}
	return verifiedSource{m.boundSource, m.boundSize}, m.boundSize, nil
}

type verifiedSource struct {
	source io.ReaderAt
	size   int64
}

func (s verifiedSource) ReadAt(p []byte, offset int64) (int, error) {
	if offset < 0 || offset > s.size || int64(len(p)) > s.size-offset {
		return 0, io.ErrUnexpectedEOF
	}
	if len(p) == 0 {
		return 0, nil
	}
	return s.source.ReadAt(p, offset)
}
