# Physical decode fixture

`TestPhysicalH264` requires `NVDEC_FIXTURE_DIR` with `three.h264` and
`three.nv12`. Generate them outside the repository with CPU FFmpeg:

    ffmpeg -f lavfi -i testsrc2=size=128x96:rate=1 -frames:v 3 \
      -c:v libx264 -qp 0 -pix_fmt yuv420p -x264-params keyint=1 \
      -f h264 three.h264
    ffmpeg -i three.h264 -pix_fmt nv12 -f rawvideo three.nv12

Run with `go test -tags nvdec_physical ./nvdec`. The tag requires a working
CUDA driver and NVDEC library; missing prerequisites fail the test.

The test compares three complete frames against the CPU decoder output,
using the pitch returned by NVDEC to copy each row. It checks map/unmap and
parser/decoder/context cleanup results. This does not measure leaks or cover
resolution changes, interlacing, other codecs, or malformed streams.

Keep fixture hashes and FFmpeg version with each runtime receipt. Binary
fixtures are not committed.
