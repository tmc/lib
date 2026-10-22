package dsv41

import (
	"crypto/sha256"
	"fmt"
	"io"

	"github.com/tmc/lib/gguf"
)

// BindVerified binds the model and verifies the SHA-256 of the complete file
// before admitting portable payloads. The expected identity must come from an
// independent asset receipt. This reads the entire file, but makes no copy
// proportional to file size. The source remains borrowed and immutable.
func BindVerified(source io.ReaderAt, size int64, g *gguf.File, expected [32]byte) (*Model, error) {
	if expected == ([32]byte{}) {
		return nil, fmt.Errorf("missing V4.1 asset identity")
	}
	m, err := Bind(source, size, g)
	if err != nil {
		return nil, err
	}
	digest, err := verifyAsset(source, size, expected)
	if err != nil {
		return nil, err
	}
	m.assetSHA = digest
	return m, nil
}

func verifyAsset(source io.ReaderAt, size int64, expected [32]byte) ([32]byte, error) {
	digest, err := hashAsset(source, size)
	if err != nil {
		return [32]byte{}, err
	}
	if expected == ([32]byte{}) || digest != expected {
		return [32]byte{}, fmt.Errorf("V4.1 asset identity mismatch")
	}
	return digest, nil
}

func hashAsset(source io.ReaderAt, size int64) ([32]byte, error) {
	var digest [32]byte
	if source == nil || size < 0 {
		return digest, fmt.Errorf("invalid V4.1 asset source")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.NewSectionReader(source, 0, size))
	if err != nil {
		return digest, fmt.Errorf("hash V4.1 asset: %w", err)
	}
	if n != size {
		return digest, fmt.Errorf("hash V4.1 asset: %w", io.ErrUnexpectedEOF)
	}
	copy(digest[:], h.Sum(nil))
	return digest, nil
}

// AssetSHA256 returns the verified complete-file identity. An unverified Bind
// returns zero and cannot write or restore portable payloads.
func (m *Model) AssetSHA256() [32]byte {
	if m == nil {
		return [32]byte{}
	}
	return m.assetSHA
}
