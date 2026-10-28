package cudacc

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"errors"
	"fmt"
	"math"
	"testing"
)

type testFabric struct {
	reports []AttestationReport
	keys    map[string]*ecdsa.PrivateKey
	policy  FabricPolicy
}

func newTestFabric(t *testing.T, gpus, switches int) testFabric {
	t.Helper()
	f := testFabric{keys: make(map[string]*ecdsa.PrivateKey)}
	f.policy = FabricPolicy{Mode: CCModePPCIE, Keys: make(map[string]*ecdsa.PublicKey)}
	for i := range f.policy.ExpectedNonce {
		f.policy.ExpectedNonce[i] = byte(i * 7)
	}
	for i := range f.policy.ExpectedRIM {
		f.policy.ExpectedRIM[i] = byte(0xa0 + i)
		f.policy.ExpectedVBIOS[i] = byte(0x10 + i)
	}
	add := func(uuid, arch string) {
		key, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
		if err != nil {
			t.Fatal(err)
		}
		r := AttestationReport{
			DeviceUUID:      uuid,
			Arch:            arch,
			Mode:            CCModePPCIE,
			FirmwareRIMHash: f.policy.ExpectedRIM,
			VBIOSHash:       f.policy.ExpectedVBIOS,
			ChallengeNonce:  f.policy.ExpectedNonce,
		}
		if err := r.Sign(key); err != nil {
			t.Fatal(err)
		}
		f.reports = append(f.reports, r)
		f.keys[uuid] = key
		f.policy.Keys[uuid] = &key.PublicKey
	}
	for i := range gpus {
		add(fmt.Sprintf("GPU-%04d", i), "sm_100a")
	}
	for i := range switches {
		add(fmt.Sprintf("NVSW-%02d", i), "nvswitch5")
	}
	return f
}

func TestVerifyFabricAttestation(t *testing.T) {
	base := newTestFabric(t, 4, 2)
	if err := VerifyFabricAttestation(base.reports, base.policy); err != nil {
		t.Fatalf("valid fabric: %v", err)
	}
	resign := func(f *testFabric, i int) {
		if err := f.reports[i].Sign(f.keys[f.reports[i].DeviceUUID]); err != nil {
			t.Fatal(err)
		}
	}
	tests := []struct {
		name   string
		mutate func(f *testFabric)
		want   error
	}{
		{"nonce_bit_unsigned", func(f *testFabric) { f.reports[1].ChallengeNonce[5] ^= 1 }, ErrBadSignature},
		{"rim_bit_unsigned", func(f *testFabric) { f.reports[2].FirmwareRIMHash[47] ^= 0x80 }, ErrBadSignature},
		{"mode_unsigned", func(f *testFabric) { f.reports[0].Mode = CCModeOn }, ErrBadSignature},
		{"signature_bit", func(f *testFabric) { f.reports[3].Countersignature[10] ^= 1 }, ErrBadSignature},
		{"stale_nonce_signed", func(f *testFabric) {
			f.reports[1].ChallengeNonce[0] ^= 1
			resign(f, 1)
		}, ErrNonceMismatch},
		{"rim_signed", func(f *testFabric) {
			f.reports[4].FirmwareRIMHash[0] ^= 1
			resign(f, 4)
		}, ErrMeasurementMismatch},
		{"vbios_signed", func(f *testFabric) {
			f.reports[5].VBIOSHash[20] ^= 1
			resign(f, 5)
		}, ErrMeasurementMismatch},
		{"mode_off_signed", func(f *testFabric) {
			f.reports[0].Mode = CCModeOff
			resign(f, 0)
		}, ErrModeMismatch},
		{"wrong_key", func(f *testFabric) {
			if err := f.reports[2].Sign(f.keys[f.reports[3].DeviceUUID]); err != nil {
				t.Fatal(err)
			}
		}, ErrBadSignature},
		{"duplicate", func(f *testFabric) { f.reports[1] = f.reports[0] }, ErrDuplicateDevice},
		{"unknown", func(f *testFabric) {
			delete(f.policy.Keys, f.reports[2].DeviceUUID)
			f.policy.Devices = len(f.reports)
		}, ErrUnknownDevice},
		{"missing_report", func(f *testFabric) { f.reports = f.reports[:5] }, ErrDeviceCount},
		{"policy_off", func(f *testFabric) { f.policy.Mode = CCModeOff }, ErrModeMismatch},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newTestFabric(t, 4, 2)
			tc.mutate(&f)
			err := VerifyFabricAttestation(f.reports, f.policy)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestAttestationReportBinary(t *testing.T) {
	f := newTestFabric(t, 1, 0)
	f.policy.Mode = CCModeOn
	for i := range f.reports {
		f.reports[i].Mode = CCModeOn
		if err := f.reports[i].Sign(f.keys[f.reports[i].DeviceUUID]); err != nil {
			t.Fatal(err)
		}
	}
	b, err := f.reports[0].MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	var got AttestationReport
	if err := got.UnmarshalBinary(b); err != nil {
		t.Fatal(err)
	}
	if err := VerifyFabricAttestation([]AttestationReport{got}, f.policy); err != nil {
		t.Fatalf("decoded report: %v", err)
	}
	for n := range len(b) {
		if err := new(AttestationReport).UnmarshalBinary(b[:n]); err == nil {
			t.Fatalf("truncated to %d bytes: no error", n)
		}
	}
	// Flipping any bit of the encoding must fail to decode or to verify.
	for i := range len(b) {
		c := bytes.Clone(b)
		c[i] ^= 1
		var r AttestationReport
		if r.UnmarshalBinary(c) != nil {
			continue
		}
		if VerifyFabricAttestation([]AttestationReport{r}, f.policy) == nil {
			t.Fatalf("bit flip at byte %d verified", i)
		}
	}
}

func testKey(b byte) (k [32]byte) {
	for i := range k {
		k[i] = b + byte(i)
	}
	return k
}

func TestBounceRingRoundTrip(t *testing.T) {
	session := [16]byte{0xde, 0xad, 0xbe, 0xef}
	tx, err := NewEncryptedBounceRing(4, 64, session)
	if err != nil {
		t.Fatal(err)
	}
	rx, _ := NewEncryptedBounceRing(4, 64, session)
	key := testKey(1)
	seen := make(map[[12]byte]bool)
	for i := range 200 {
		pt := bytes.Repeat([]byte{byte(i)}, 1+i%64)
		aad := []byte(fmt.Sprintf("slot%d", tx.Slot(tx.NextSeqNo)))
		iv, ct, err := tx.SealBounceSlot(key, pt, aad)
		if err != nil {
			t.Fatal(err)
		}
		if seen[iv] {
			t.Fatalf("IV %x reused", iv)
		}
		seen[iv] = true
		got, err := rx.OpenBounceSlot(key, iv, ct, aad)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, pt) {
			t.Fatalf("seq %d: plaintext mismatch", i)
		}
	}
	if tx.NextSeqNo != 200 {
		t.Errorf("NextSeqNo = %d, want 200", tx.NextSeqNo)
	}
	if _, _, err := tx.SealBounceSlot(key, make([]byte, 65), nil); err == nil {
		t.Error("oversized plaintext: no error")
	}
}

func TestBounceRingTamper(t *testing.T) {
	session := [16]byte{1, 2, 3, 4}
	key := testKey(9)
	tx, _ := NewEncryptedBounceRing(2, 32, session)
	pt := []byte("activation tile 0123456789abcdef")
	aad := []byte("hdr")
	iv, ct, err := tx.SealBounceSlot(key, pt, aad)
	if err != nil {
		t.Fatal(err)
	}
	open := func(key [32]byte, iv [12]byte, ct, aad []byte) ([]byte, error) {
		rx, _ := NewEncryptedBounceRing(2, 32, session)
		return rx.OpenBounceSlot(key, iv, ct, aad)
	}
	for i := range len(ct) * 8 {
		c := bytes.Clone(ct)
		c[i/8] ^= 1 << (i % 8)
		if got, err := open(key, iv, c, aad); !errors.Is(err, ErrAuth) || got != nil {
			t.Fatalf("ciphertext bit %d: got %q, %v", i, got, err)
		}
	}
	for i := range len(aad) * 8 {
		a := bytes.Clone(aad)
		a[i/8] ^= 1 << (i % 8)
		if got, err := open(key, iv, ct, a); !errors.Is(err, ErrAuth) || got != nil {
			t.Fatalf("aad bit %d: got %q, %v", i, got, err)
		}
	}
	badSeq := iv
	badSeq[11] ^= 1
	if _, err := open(key, badSeq, ct, aad); !errors.Is(err, ErrAuth) {
		t.Errorf("sequence bit: %v", err)
	}
	badSalt := iv
	badSalt[0] ^= 1
	if _, err := open(key, badSalt, ct, aad); !errors.Is(err, ErrSessionMismatch) {
		t.Errorf("salt bit: %v", err)
	}
	if _, err := open(testKey(10), iv, ct, aad); !errors.Is(err, ErrAuth) {
		t.Errorf("wrong key: %v", err)
	}
}

func TestBounceRingReplay(t *testing.T) {
	session := [16]byte{7}
	key := testKey(3)
	tx, _ := NewEncryptedBounceRing(8, 16, session)
	type slot struct {
		iv [12]byte
		ct []byte
	}
	var slots []slot
	for range 100 {
		iv, ct, err := tx.SealBounceSlot(key, []byte("x"), nil)
		if err != nil {
			t.Fatal(err)
		}
		slots = append(slots, slot{iv, ct})
	}
	rx, _ := NewEncryptedBounceRing(8, 16, session)
	mustOpen := func(i int) {
		t.Helper()
		if _, err := rx.OpenBounceSlot(key, slots[i].iv, slots[i].ct, nil); err != nil {
			t.Fatalf("open %d: %v", i, err)
		}
	}
	mustReject := func(i int) {
		t.Helper()
		if _, err := rx.OpenBounceSlot(key, slots[i].iv, slots[i].ct, nil); !errors.Is(err, ErrReplay) {
			t.Fatalf("open %d: err = %v, want ErrReplay", i, err)
		}
	}
	mustOpen(10)
	mustReject(10)
	mustOpen(5) // out of order inside the window
	mustReject(5)
	// A forged slot with a high sequence number must not advance the window.
	forged := slots[90].iv
	if _, err := rx.OpenBounceSlot(key, forged, slots[11].ct, nil); !errors.Is(err, ErrAuth) {
		t.Fatalf("forged slot: %v", err)
	}
	mustOpen(11)
	mustOpen(90)
	mustReject(20) // 70 below the highest: outside the window
	mustOpen(27)   // 63 below: inside
	mustReject(90)
}

func TestBounceRingExhausted(t *testing.T) {
	tx, _ := NewEncryptedBounceRing(1, 8, [16]byte{})
	tx.NextSeqNo = math.MaxUint64
	if _, _, err := tx.SealBounceSlot(testKey(0), nil, nil); !errors.Is(err, ErrSeqExhausted) {
		t.Fatalf("err = %v, want ErrSeqExhausted", err)
	}
}

func TestPlanBounceTransfer(t *testing.T) {
	const size = 1 << 30
	prev := math.Inf(1)
	for _, slots := range []int{1, 2, 3, 4, 8} {
		p, err := PlanBounceTransfer(size, slots, 2<<20, 40, 50, 60)
		if err != nil {
			t.Fatal(err)
		}
		sum := p.ChunkUs[0] + p.ChunkUs[1] + p.ChunkUs[2]
		switch {
		case slots == 1 && math.Abs(p.IntervalUs-sum) > 1e-9:
			t.Errorf("1 slot: interval %v, want serial %v", p.IntervalUs, sum)
		case slots >= 3 && p.IntervalUs != p.ChunkUs[0]:
			t.Errorf("%d slots: interval %v, want encrypt-bound %v", slots, p.IntervalUs, p.ChunkUs[0])
		}
		if p.TotalUs > prev {
			t.Errorf("%d slots: total %v > %v with fewer slots", slots, p.TotalUs, prev)
		}
		prev = p.TotalUs
		if p.EffectiveGBs > 40 {
			t.Errorf("%d slots: %v GB/s exceeds the slowest stage", slots, p.EffectiveGBs)
		}
	}
	if _, err := PlanBounceTransfer(0, 1, 1, 1, 1, 1); err == nil {
		t.Error("zero bytes: no error")
	}
}

func ExampleEncryptedBounceRing() {
	session := [16]byte{0x01, 0x02, 0x03, 0x04}
	var key [32]byte
	tx, _ := NewEncryptedBounceRing(2, 64, session)
	rx, _ := NewEncryptedBounceRing(2, 64, session)

	iv, ct, _ := tx.SealBounceSlot(key, []byte("kv page 17"), []byte("slot 0"))
	pt, err := rx.OpenBounceSlot(key, iv, ct, []byte("slot 0"))
	fmt.Printf("%x %q %v\n", iv, pt, err)
	_, err = rx.OpenBounceSlot(key, iv, ct, []byte("slot 0"))
	fmt.Println(errors.Is(err, ErrReplay))
	// Output:
	// 010203040000000000000000 "kv page 17" <nil>
	// true
}
