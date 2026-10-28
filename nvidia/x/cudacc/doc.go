// Package cudacc plans and checks the host side of NVIDIA Confidential
// Computing: single-GPU CC-On and multi-GPU protected PCIe/NVLink (PPCIE).
//
// It provides:
//
//   - [AttestationReport], a signed measurement report for one GPU or
//     NVSwitch, with a binary encoding ([AttestationReport.MarshalBinary])
//     and an ECDSA P-384 countersignature over its transcript;
//   - [VerifyFabricAttestation], which admits a domain only if every
//     report is signed by its device's key, is in the required mode, echoes
//     the challenge nonce and matches the expected firmware and VBIOS
//     digests (compared in constant time);
//   - [EncryptedBounceRing], AES-GCM-256 sealing and opening of bounce
//     buffer slots with a salt‖sequence IV, strictly increasing sequence
//     numbers on send and a sliding replay window on receive;
//   - [PlanBounceTransfer], a pipeline model of an encrypted transfer
//     through a bounce ring.
//
// The report encoding and IV layout are this package's own; they are not
// the DMTF SPDM message format or NVIDIA's copy-engine wire format, and
// nothing here talks to a GPU, a root of trust or the NVIDIA attestation
// service. Verifying the device certificate chain that vouches for each
// device key is left to the caller.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package cudacc
