// Package dsv41engram implements DeepSeek V4.1 Engram hashes and bounded table
// reads. Tables use 256 E4M3 bytes followed by eight E8M0 scale bytes per row.
// ReadRows returns the reference's BF16-rounded values in float32 storage;
// ReadEncodedRows preserves the serialized row for backend-specific decoding.
// The package does not allocate or map a complete table.
package dsv41engram
