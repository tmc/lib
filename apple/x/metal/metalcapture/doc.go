// Package metalcapture provides pure Go portable recording, serialization, and
// inspection of Metal profiling captures.
//
// Captures are stored as standard pprof profiles containing an embedded,
// versioned payload with source files, intermediate representations (AIR),
// GPU assembly where collected, and structured measurement metadata.
//
// The package has no dependencies on Apple private frameworks or GPU hardware,
// and can be compiled and run on any operating system.
package metalcapture
