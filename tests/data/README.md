# Test data

## `portrait_sample.jpg`

The face-service tests that need a real face (`tests/test_face_service_backend.py`
and `tests/test_face_service_contract.py`) read `tests/data/portrait_sample.jpg`.
It must be:

- a **public-domain or freely licensed** photo from Wikimedia Commons (record the
  file name, author and licence in the list below),
- a Commons thumbnail at **960 px wide** (`Special:FilePath/<file>?width=960`),
  the size the worker detects on,
- showing **exactly one clearly visible, roughly frontal face**.

Until it exists those tests skip locally. CI sets `WIKIVISAGE_REQUIRE_SAMPLE=1`, so
a missing file fails the face-service jobs instead of silently disabling them.
To try a different photo without committing it, set
`WIKIVISAGE_TEST_SAMPLE_IMAGE=/path/to/photo.jpg`.

Do not use a photo of a private individual, and do not commit anything under a
licence that forbids redistribution.

### Provenance

| File | Commons file | Author | Licence |
|---|---|---|---|
| `portrait_sample.jpg` | [File:Joe Biden presidential portrait (3x4 close cropped).jpg](https://commons.wikimedia.org/wiki/File:Joe_Biden_presidential_portrait_(3x4_close_cropped).jpg) | Adam Schultz (White House) | Public domain (US federal government work, 2021-03-03) |

Downloaded as a 960 px-wide Commons thumbnail
(`Special:FilePath/File:Joe_Biden_presidential_portrait_(3x4_close_cropped).jpg?width=960`,
244 KB, 960x1268).

The close crop is deliberate: the face is ~514x756 px at 960 px width, well past
the ~430 px at which SCRFD-500M starts missing faces at full scale, so the
backend's half-scale pass is part of the path under test. The fixture is also
what `test_real_portrait_has_exactly_one_confident_face_and_a_usable_embedding`
pins: exactly one detection, score > 0.6, and an L2-normalised embedding.
