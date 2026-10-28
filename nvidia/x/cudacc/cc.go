package cudacc

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/sha512"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// CCMode is the confidential-computing mode of a GPU or NVLink domain.
type CCMode uint8

const (
	CCModeOff   CCMode = iota
	CCModeOn           // single-GPU TEE with encrypted copy-engine bounce buffers
	CCModePPCIE        // multi-GPU protected PCIe/NVLink domain
)

func (m CCMode) String() string {
	switch m {
	case CCModeOff:
		return "off"
	case CCModeOn:
		return "on"
	case CCModePPCIE:
		return "ppcie"
	}
	return fmt.Sprintf("CCMode(%d)", uint8(m))
}

// Errors reported by [VerifyFabricAttestation] and [EncryptedBounceRing].
var (
	ErrBadSignature        = errors.New("attestation signature invalid")
	ErrUnknownDevice       = errors.New("no key for device")
	ErrDuplicateDevice     = errors.New("duplicate device")
	ErrModeMismatch        = errors.New("confidential computing mode mismatch")
	ErrNonceMismatch       = errors.New("challenge nonce mismatch")
	ErrMeasurementMismatch = errors.New("measurement mismatch")
	ErrDeviceCount         = errors.New("wrong number of devices")

	ErrAuth            = errors.New("bounce slot authentication failed")
	ErrReplay          = errors.New("bounce slot replayed or too old")
	ErrSessionMismatch = errors.New("bounce slot from another session")
	ErrSeqExhausted    = errors.New("bounce ring sequence numbers exhausted")
)

// AttestationReport is the measurement report of one GPU or NVSwitch.
//
// Countersignature is an ASN.1 ECDSA P-384 signature, by the device's
// identity key, over the SHA-384 digest of [AttestationReport.Transcript].
type AttestationReport struct {
	DeviceUUID       string
	Arch             string
	Mode             CCMode
	FirmwareRIMHash  [48]byte // SHA-384 reference integrity manifest digest
	VBIOSHash        [48]byte
	ChallengeNonce   [32]byte
	Countersignature []byte
}

const reportMagic = "NVCC\x01"

// Transcript returns the signed part of the report's binary encoding:
//
//	"NVCC" 0x01 | mode u8 | len u16 | uuid | len u16 | arch |
//	rim [48] | vbios [48] | nonce [32]
//
// Lengths are little-endian.
func (r AttestationReport) Transcript() ([]byte, error) {
	if len(r.DeviceUUID) > math.MaxUint16 || len(r.Arch) > math.MaxUint16 {
		return nil, errors.New("attestation report: field too long")
	}
	b := make([]byte, 0, len(reportMagic)+1+4+len(r.DeviceUUID)+len(r.Arch)+48+48+32)
	b = append(b, reportMagic...)
	b = append(b, byte(r.Mode))
	b = binary.LittleEndian.AppendUint16(b, uint16(len(r.DeviceUUID)))
	b = append(b, r.DeviceUUID...)
	b = binary.LittleEndian.AppendUint16(b, uint16(len(r.Arch)))
	b = append(b, r.Arch...)
	b = append(b, r.FirmwareRIMHash[:]...)
	b = append(b, r.VBIOSHash[:]...)
	b = append(b, r.ChallengeNonce[:]...)
	return b, nil
}

// MarshalBinary encodes the report as its transcript followed by a u16
// signature length and the signature.
func (r AttestationReport) MarshalBinary() ([]byte, error) {
	b, err := r.Transcript()
	if err != nil {
		return nil, err
	}
	if len(r.Countersignature) > math.MaxUint16 {
		return nil, errors.New("attestation report: signature too long")
	}
	b = binary.LittleEndian.AppendUint16(b, uint16(len(r.Countersignature)))
	return append(b, r.Countersignature...), nil
}

// UnmarshalBinary decodes a report written by MarshalBinary.
func (r *AttestationReport) UnmarshalBinary(b []byte) error {
	if len(b) < len(reportMagic) || string(b[:len(reportMagic)]) != reportMagic {
		return errors.New("attestation report: bad magic")
	}
	b = b[len(reportMagic):]
	var out AttestationReport
	if len(b) < 1 {
		return errors.New("attestation report: truncated")
	}
	out.Mode, b = CCMode(b[0]), b[1:]
	str := func() (string, bool) {
		if len(b) < 2 {
			return "", false
		}
		n := int(binary.LittleEndian.Uint16(b))
		if len(b) < 2+n {
			return "", false
		}
		s := string(b[2 : 2+n])
		b = b[2+n:]
		return s, true
	}
	var ok bool
	if out.DeviceUUID, ok = str(); !ok {
		return errors.New("attestation report: truncated uuid")
	}
	if out.Arch, ok = str(); !ok {
		return errors.New("attestation report: truncated arch")
	}
	if len(b) < 48+48+32+2 {
		return errors.New("attestation report: truncated measurements")
	}
	b = b[copy(out.FirmwareRIMHash[:], b):]
	b = b[copy(out.VBIOSHash[:], b):]
	b = b[copy(out.ChallengeNonce[:], b):]
	n := int(binary.LittleEndian.Uint16(b))
	b = b[2:]
	if len(b) != n {
		return fmt.Errorf("attestation report: signature length %d, have %d bytes", n, len(b))
	}
	out.Countersignature = append([]byte(nil), b...)
	*r = out
	return nil
}

func (r AttestationReport) digest() ([]byte, error) {
	t, err := r.Transcript()
	if err != nil {
		return nil, err
	}
	d := sha512.Sum384(t)
	return d[:], nil
}

// Sign sets r.Countersignature using key, which must be a P-384 key. It is
// for simulated devices and tests; real reports are signed by the device.
func (r *AttestationReport) Sign(key *ecdsa.PrivateKey) error {
	d, err := r.digest()
	if err != nil {
		return err
	}
	sig, err := ecdsa.SignASN1(rand.Reader, key, d)
	if err != nil {
		return fmt.Errorf("sign attestation report: %w", err)
	}
	r.Countersignature = sig
	return nil
}

// FabricPolicy is what [VerifyFabricAttestation] requires of a domain.
type FabricPolicy struct {
	Mode          CCMode // CCModeOn or CCModePPCIE
	ExpectedNonce [32]byte
	ExpectedRIM   [48]byte
	ExpectedVBIOS [48]byte
	// Keys maps each DeviceUUID to its identity key. The caller is
	// responsible for having verified the certificate chain behind each key.
	Keys map[string]*ecdsa.PublicKey
	// Devices is the number of reports required; 0 means len(Keys).
	Devices int
}

// VerifyFabricAttestation checks every report against p and returns nil
// only if all pass, so peer mappings may be enabled for the whole domain.
//
// Each report must name a distinct device with a key in p.Keys and carry a
// valid signature by that key over its transcript; only then are its
// fields trusted and compared with p (nonce and digests in constant time).
// The returned error wraps one of the package's Err values and names the
// first failing device.
func VerifyFabricAttestation(reports []AttestationReport, p FabricPolicy) error {
	if p.Mode != CCModeOn && p.Mode != CCModePPCIE {
		return fmt.Errorf("verify fabric attestation: policy mode %v: %w", p.Mode, ErrModeMismatch)
	}
	want := p.Devices
	if want == 0 {
		want = len(p.Keys)
	}
	if want == 0 || len(reports) != want {
		return fmt.Errorf("verify fabric attestation: %d reports, want %d: %w", len(reports), want, ErrDeviceCount)
	}
	if p.Mode == CCModeOn && len(reports) != 1 {
		return fmt.Errorf("verify fabric attestation: CC-On covers one GPU, have %d reports: %w", len(reports), ErrModeMismatch)
	}
	seen := make(map[string]bool, len(reports))
	for _, r := range reports {
		if err := verifyReport(r, p, seen); err != nil {
			return fmt.Errorf("verify fabric attestation: device %q: %w", r.DeviceUUID, err)
		}
	}
	return nil
}

func verifyReport(r AttestationReport, p FabricPolicy, seen map[string]bool) error {
	if seen[r.DeviceUUID] {
		return ErrDuplicateDevice
	}
	seen[r.DeviceUUID] = true
	key := p.Keys[r.DeviceUUID]
	if key == nil {
		return ErrUnknownDevice
	}
	d, err := r.digest()
	if err != nil {
		return err
	}
	if !ecdsa.VerifyASN1(key, d, r.Countersignature) {
		return ErrBadSignature
	}
	if r.Mode != p.Mode {
		return fmt.Errorf("%w: report %v, policy %v", ErrModeMismatch, r.Mode, p.Mode)
	}
	if subtle.ConstantTimeCompare(r.ChallengeNonce[:], p.ExpectedNonce[:]) != 1 {
		return ErrNonceMismatch
	}
	rim := subtle.ConstantTimeCompare(r.FirmwareRIMHash[:], p.ExpectedRIM[:])
	vbios := subtle.ConstantTimeCompare(r.VBIOSHash[:], p.ExpectedVBIOS[:])
	if rim&vbios != 1 {
		return ErrMeasurementMismatch
	}
	return nil
}

// replayWindow is the number of sequence numbers below the highest one
// received that OpenBounceSlot still accepts once.
const replayWindow = 64

// EncryptedBounceRing seals and opens bounce-buffer slots with AES-GCM-256.
//
// The 96-bit IV of each slot is SessionID[0:4] followed by the big-endian
// 64-bit sequence number, which increases by one per sealed slot, so an IV
// never repeats under a key. The receiving side accepts each sequence
// number at most once and rejects numbers more than 64 below the highest
// accepted. A ring may both send and receive; the two directions should
// use different keys.
//
// An EncryptedBounceRing is not safe for concurrent use.
type EncryptedBounceRing struct {
	NumSlots  int
	SlotBytes int
	SessionID [16]byte
	NextSeqNo uint64 // sequence number of the next sealed slot

	recvAny  bool
	recvHigh uint64
	recvMask uint64 // bit i: recvHigh-i accepted
}

// NewEncryptedBounceRing returns a ring of numSlots slots of slotBytes
// plaintext bytes each.
func NewEncryptedBounceRing(numSlots, slotBytes int, session [16]byte) (*EncryptedBounceRing, error) {
	if numSlots <= 0 || slotBytes <= 0 {
		return nil, fmt.Errorf("bounce ring: need positive slots and slot size, have %d x %d", numSlots, slotBytes)
	}
	return &EncryptedBounceRing{NumSlots: numSlots, SlotBytes: slotBytes, SessionID: session}, nil
}

// Slot returns the ring slot that carries sequence number seq.
func (r *EncryptedBounceRing) Slot(seq uint64) int { return int(seq % uint64(r.NumSlots)) }

func gcm(key [32]byte) (cipher.AEAD, error) {
	b, err := aes.NewCipher(key[:])
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(b)
}

// SealBounceSlot encrypts plaintext and authenticates it with aad under the
// next sequence number, returning the IV and the ciphertext with its
// 16-byte tag appended.
func (r *EncryptedBounceRing) SealBounceSlot(key [32]byte, plaintext, aad []byte) (iv [12]byte, ciphertextWithTag []byte, err error) {
	if len(plaintext) > r.SlotBytes {
		return iv, nil, fmt.Errorf("seal bounce slot: %d bytes exceed slot size %d", len(plaintext), r.SlotBytes)
	}
	if r.NextSeqNo == math.MaxUint64 {
		return iv, nil, ErrSeqExhausted
	}
	a, err := gcm(key)
	if err != nil {
		return iv, nil, fmt.Errorf("seal bounce slot: %w", err)
	}
	copy(iv[:4], r.SessionID[:4])
	binary.BigEndian.PutUint64(iv[4:], r.NextSeqNo)
	r.NextSeqNo++
	return iv, a.Seal(nil, iv[:], plaintext, aad), nil
}

// OpenBounceSlot checks the IV's session and sequence number, verifies the
// tag and returns the plaintext. On any error it returns no plaintext, and
// only a slot that authenticates advances the replay window.
func (r *EncryptedBounceRing) OpenBounceSlot(key [32]byte, iv [12]byte, ciphertextWithTag, aad []byte) ([]byte, error) {
	if subtle.ConstantTimeCompare(iv[:4], r.SessionID[:4]) != 1 {
		return nil, ErrSessionMismatch
	}
	seq := binary.BigEndian.Uint64(iv[4:])
	if !r.fresh(seq) {
		return nil, fmt.Errorf("open bounce slot %d: %w", seq, ErrReplay)
	}
	a, err := gcm(key)
	if err != nil {
		return nil, fmt.Errorf("open bounce slot: %w", err)
	}
	if len(ciphertextWithTag) > r.SlotBytes+a.Overhead() {
		return nil, fmt.Errorf("open bounce slot: %d bytes exceed slot size", len(ciphertextWithTag))
	}
	pt, err := a.Open(nil, iv[:], ciphertextWithTag, aad)
	if err != nil {
		return nil, fmt.Errorf("open bounce slot %d: %w", seq, ErrAuth)
	}
	r.accept(seq)
	return pt, nil
}

func (r *EncryptedBounceRing) fresh(seq uint64) bool {
	if !r.recvAny || seq > r.recvHigh {
		return true
	}
	d := r.recvHigh - seq
	return d < replayWindow && r.recvMask&(1<<d) == 0
}

func (r *EncryptedBounceRing) accept(seq uint64) {
	switch {
	case !r.recvAny:
		r.recvAny, r.recvHigh, r.recvMask = true, seq, 1
	case seq > r.recvHigh:
		shift := seq - r.recvHigh
		if shift >= replayWindow {
			r.recvMask = 0
		} else {
			r.recvMask <<= shift
		}
		r.recvMask |= 1
		r.recvHigh = seq
	default:
		r.recvMask |= 1 << (r.recvHigh - seq)
	}
}

// BounceTransferPlan models one encrypted transfer through a bounce ring.
type BounceTransferPlan struct {
	Chunks       int
	ChunkUs      [3]float64 // encrypt, DMA, decrypt time per full chunk
	IntervalUs   float64    // steady-state time between chunk completions
	TotalUs      float64
	EffectiveGBs float64
}

// PlanBounceTransfer models moving bytes through a ring of numSlots slots
// of slotBytes each, in three stages per chunk: encrypt at encryptGBps, DMA
// at linkGBps, decrypt at decryptGBps. A slot is held from encryption to
// decryption, so at most numSlots chunks are in flight and the steady
// interval is max(slowest stage, sum of stages / numSlots).
func PlanBounceTransfer(bytes int64, numSlots, slotBytes int, encryptGBps, linkGBps, decryptGBps float64) (BounceTransferPlan, error) {
	if bytes <= 0 || numSlots <= 0 || slotBytes <= 0 {
		return BounceTransferPlan{}, fmt.Errorf("plan bounce transfer: bytes %d, slots %d, slot size %d must be > 0", bytes, numSlots, slotBytes)
	}
	if encryptGBps <= 0 || linkGBps <= 0 || decryptGBps <= 0 {
		return BounceTransferPlan{}, errors.New("plan bounce transfer: bandwidths must be > 0")
	}
	var p BounceTransferPlan
	p.Chunks = int((bytes + int64(slotBytes) - 1) / int64(slotBytes))
	// GB/s is bytes per nanosecond, so bytes / (GBps*1e3) is microseconds.
	us := func(gbps float64) float64 { return float64(slotBytes) / (gbps * 1e3) }
	p.ChunkUs = [3]float64{us(encryptGBps), us(linkGBps), us(decryptGBps)}
	sum := p.ChunkUs[0] + p.ChunkUs[1] + p.ChunkUs[2]
	slowest := max(p.ChunkUs[0], p.ChunkUs[1], p.ChunkUs[2])
	p.IntervalUs = max(slowest, sum/float64(numSlots))
	p.TotalUs = sum + float64(p.Chunks-1)*p.IntervalUs
	p.EffectiveGBs = float64(bytes) / (p.TotalUs * 1e3)
	return p, nil
}
