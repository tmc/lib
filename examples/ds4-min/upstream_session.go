package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4kvstore"
)

// upstreamSessionContract is a caller-provided, independently qualified
// receipt. It is never inferred from the KVC that it is supposed to authorize.
type upstreamSessionContract struct {
	ModelSHA256    string
	TokenIDsSHA256 string
	PayloadSHA256  string
	Contract       dsv4kvstore.UpstreamRestoreContract
}

type upstreamStateLoader interface {
	LoadUpstream(dsv4kvstore.UpstreamRecord, dsv4kvstore.UpstreamRestoreContract) error
}

func readUpstreamSessionContract(r io.Reader) (upstreamSessionContract, error) {
	var c upstreamSessionContract
	data, err := io.ReadAll(io.LimitReader(r, (1<<20)+1))
	if err != nil {
		return c, err
	}
	if len(data) > 1<<20 {
		return c, errors.New("read upstream contract: too large")
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return c, fmt.Errorf("read upstream contract: %w", err)
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return c, errors.New("read upstream contract: trailing data")
	}
	for _, field := range []struct {
		text string
		dst  *[sha256.Size]byte
	}{{c.ModelSHA256, &c.Contract.ModelSHA256}, {c.TokenIDsSHA256, &c.Contract.TokenIDsSHA256}} {
		v, err := hex.DecodeString(field.text)
		if err != nil || len(v) != sha256.Size {
			return c, errors.New("read upstream contract: full model and token hashes required")
		}
		copy(field.dst[:], v)
	}
	if v, err := hex.DecodeString(c.PayloadSHA256); err != nil || len(v) != sha256.Size {
		return c, errors.New("read upstream contract: full payload hash required")
	}
	return c, nil
}

func loadUpstreamSession(s *session, record dsv4kvstore.UpstreamRecord, c upstreamSessionContract) error {
	if s == nil || s.backend == nil {
		return dsv4.ErrClosed
	}
	loader, ok := s.backend.(upstreamStateLoader)
	if !ok {
		return fmt.Errorf("load upstream session: %w", dsv4.ErrUnimplemented)
	}
	layout, err := dsv4kvstore.ValidateUpstreamRecordLayout(record, c.Contract.Ratios)
	if err != nil {
		return err
	}
	if err := dsv4kvstore.ValidateUpstreamRestoreContract(layout, c.Contract); err != nil {
		return err
	}
	if record.Metadata.QuantBits != c.Contract.QuantBits || len(record.Trailer) != 0 {
		return errors.New("load upstream session: unsupported quantization or trailer")
	}
	fingerprint, err := hex.DecodeString(c.PayloadSHA256)
	if err != nil || !bytes.Equal(fingerprint, record.PayloadSHA256[:]) {
		return errors.New("load upstream session: payload provenance mismatch")
	}
	h, err := dsv4kvstore.HashUpstreamPayloadTokenIDs(bytes.NewReader(record.Payload), layout)
	if err != nil || h != c.Contract.TokenIDsSHA256 {
		return errors.New("load upstream session: token provenance mismatch")
	}
	if layout.Metadata.Context != uint32(s.context) || layout.Metadata.Tokens >= uint32(s.context) {
		return errors.New("load upstream session: context mismatch or exhausted")
	}
	tokens := make([]int, layout.Metadata.Tokens)
	for i := range tokens {
		v := binary.LittleEndian.Uint32(record.Payload[52+4*i:])
		if v >= layout.Metadata.Vocab {
			return errors.New("load upstream session: invalid token")
		}
		tokens[i] = int(v)
	}
	start := 52 + 4*len(tokens)
	logits := make([]float32, layout.Metadata.Vocab)
	for i := range logits {
		logits[i] = math.Float32frombits(binary.LittleEndian.Uint32(record.Payload[start+4*i:]))
		if math.IsNaN(float64(logits[i])) || math.IsInf(float64(logits[i]), 0) {
			return errors.New("load upstream session: nonfinite logits")
		}
	}
	if err := loader.LoadUpstream(record, c.Contract); err != nil {
		return err
	}
	s.tokens, s.logits = tokens, logits
	return nil
}

func loadUpstreamSessionPaths(s *session, path, contractPath string) error {
	f, err := os.Open(contractPath)
	if err != nil {
		return fmt.Errorf("open upstream contract: %w", err)
	}
	c, readErr := readUpstreamSessionContract(f)
	closeErr := f.Close()
	if err := errors.Join(readErr, closeErr); err != nil {
		return err
	}
	record, err := dsv4kvstore.ReadUpstreamPath(path, dsv4kvstore.UpstreamInspectOptions{MaxPayloadBytes: maxLocalSessionBytes, MaxFileBytes: maxLocalSessionBytes})
	if err != nil {
		return err
	}
	return loadUpstreamSession(s, record, c)
}
