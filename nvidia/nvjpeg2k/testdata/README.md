# nvjpeg2k test data

Two kinds of file are described here.

`exported_symbols.txt` is committed. It is the dynamic symbol table of the
pinned `libnvjpeg2k`, in text form, and `abi_test.go` checks the package's
symbol lookups against it. Its header records the archive and sha256 it came
from.

The decode fixtures are **not committed**. They are binary images, several of
them third-party with their own licenses, and this repository does not carry
binary test data. `physical_test.go` reads them from `$NVJPEG2K_FIXTURE_DIR`
and the reference decodes from `$NVJPEG2K_REFERENCE_DIR`. Nothing in the
package downloads them.

## What each fixture has to be

The physical test makes an **exact** pixel claim, so every fixture must be
**lossless**: reversible 5/3 wavelet, no quantization. A lossy fixture only
supports a tolerance comparison, and a tolerance hides exactly the class of
bug — a plane sized from the wrong dimensions, a pitch off by one row — that
this test exists to catch.

| Fixture | Codestream | Purpose |
| --- | --- | --- |
| `lossless_rgb_8bit.j2k` | reversible 5/3, RGB 4:4:4, 8-bit, untiled | the baseline exact comparison |
| `lossless_ycc420_8bit.jp2` | reversible 5/3, sYCC 4:2:0, 8-bit, JP2 container | the per-component trap: chroma planes are half the luma width and height, so a destination sized from the image dimensions decodes without an error and produces garbage |
| `lossless_gray_12bit.j2k` | reversible 5/3, single component, 12-bit | precision above 8 bits, so the destination type must come from the component precision |
| `lossless_rgb_8bit_tiled.j2k` | reversible 5/3, RGB, 8-bit, tiled (e.g. 256x256 tiles) | tile component dimensions differ from image component dimensions |

Malformed inputs are derived from the first fixture at run time — truncations,
a bare SOC/SIZ prefix, a non-codestream, and a byte flipped three quarters of
the way in — so there is nothing to obtain for those.

## Where to get them

They can be produced from any source image with a JPEG 2000 encoder; produce
them once and keep them outside the repository.

With OpenJPEG (`opj_compress`), from a PNG or PPM source:

    opj_compress -i source.ppm -o lossless_rgb_8bit.j2k -r 1
    opj_compress -i source.ppm -o lossless_ycc420_8bit.jp2 -r 1 -mct 1 -F 8,8,8
    opj_compress -i gray12.pgm -o lossless_gray_12bit.j2k -r 1
    opj_compress -i source.ppm -o lossless_rgb_8bit_tiled.j2k -r 1 -t 256,256

`-r 1` is compression ratio 1, which selects the reversible path. Confirm
losslessness rather than assuming it: decode the result and compare it to the
source before using it as a fixture.

Ready-made conformance streams also exist. The ISO/IEC 15444-4 conformance
files and the OpenJPEG project's `openjpeg-data` repository both carry
reversible codestreams with the properties above; either is a defensible
source, and whichever is used should be recorded here with its exact origin
and sha256 once the fixtures are pinned.

## Reference decodes

`$NVJPEG2K_REFERENCE_DIR` holds one file per fixture, named `<fixture>.raw`:
the fixture decoded by an **independent** decoder and written as raw planar
samples, components concatenated in codestream order, each component's rows
packed at exactly `width * bytes_per_sample` with no padding. That is the
layout `physical_test.go` assembles from the device buffers before comparing.

The independence is the point. Comparing nvJPEG2000 against nvJPEG2000 tests
nothing. Use OpenJPEG:

    opj_decompress -i lossless_rgb_8bit.j2k -o out.pgx    # one .pgx per component

and concatenate the `.pgx` payloads in component order, or use Pillow with
openjpeg support and write the planes out with numpy. Record which decoder and
which version produced the references alongside them; a reference decode from
an unknown decoder is not evidence.

## What this cannot check

The comparison shows that the decoded samples match another decoder. It does
not show that the asynchronous contract is respected — a decode that reads its
buffers after the call returned would still compare equal on a synchronized
run. Nothing here checks for device memory leaks, and nothing here exercises
concurrent use of a `Handle` from several host threads.
