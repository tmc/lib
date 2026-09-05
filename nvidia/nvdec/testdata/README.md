# Physical decode fixtures

TestPhysicalDecode requires NVDEC_FIXTURE_DIR containing three 256x256
frames per codec and CPU-decoded NV12 references. Generate outside git:

    ffmpeg -f lavfi -i testsrc2=size=256x256:rate=1 -frames:v 3 -c:v libx264 -threads 2 -pix_fmt yuv420p -f h264 three.h264
    ffmpeg -f lavfi -i testsrc2=size=256x256:rate=1 -frames:v 3 -c:v libx265 -threads 2 -pix_fmt yuv420p -f hevc three.hevc
    ffmpeg -f lavfi -i testsrc2=size=256x256:rate=1 -frames:v 3 -c:v libaom-av1 -threads 2 -pix_fmt yuv420p -f ivf three.ivf
    for codec in h264 hevc ivf; do
      ffmpeg -i three.$codec -pix_fmt nv12 -f rawvideo three.$codec.nv12
    done

Run with `go test -tags nvdec_physical ./nvdec`. Missing fixtures or native
prerequisites fail the test. Record hashes and FFmpeg version with receipts.
Binary fixtures are not committed.

The comparison uses decoded CPU output, so the input need not be lossless.
256x256 exceeds the minimum dimensions reported for all three codecs on the
qualified GB10. AV1 IVF frame payloads are submitted separately; the IVF
container headers are not passed to NVDEC.

The test compares complete frames using the pitch returned by NVDEC and
checks map/unmap and parser/decoder/context cleanup results. It does not
measure leaks or cover resolution changes, interlacing, higher bit depth,
or malformed compressed streams.
