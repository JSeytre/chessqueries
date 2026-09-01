"""Video manifest: id parsing, validated entries, relay-mapping, persistence."""

import pytest

from chessqueries.annotate.manifest import Manifest, VideoEntry, video_id_from_url


def test_video_id_from_url_variants():
    assert video_id_from_url("https://www.youtube.com/watch?v=hunt9gfNW48") == "hunt9gfNW48"
    assert video_id_from_url("https://youtu.be/hunt9gfNW48") == "hunt9gfNW48"
    assert video_id_from_url("hunt9gfNW48") == "hunt9gfNW48"  # bare id
    with pytest.raises(ValueError):
        video_id_from_url("not a youtube link")


def test_entry_requires_valid_id():
    with pytest.raises(ValueError):
        VideoEntry(video_id="too-short")


def test_has_relay_mapping():
    assert not VideoEntry("hunt9gfNW48").has_relay_mapping
    assert VideoEntry("hunt9gfNW48", tournaments=["80nj8Ryn"]).has_relay_mapping
    assert VideoEntry("hunt9gfNW48", round_ids=["uJ5zaYTD"]).has_relay_mapping


def test_legacy_singular_tournament_key():
    """An older entry carrying ``tournament`` (singular) reads into ``tournaments``."""
    e = VideoEntry.from_dict("hunt9gfNW48", {"tournament": "80nj8Ryn", "round_ids": ["uJ5zaYTD"]})
    assert e.tournaments == ["80nj8Ryn"]
    assert e.round_ids == ["uJ5zaYTD"]


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "videos.json"
    m = Manifest(path=path, entries={})
    assert m.add(VideoEntry("hunt9gfNW48", title="Day 3", tournaments=["80nj8Ryn"]))
    assert not m.add(VideoEntry("hunt9gfNW48"))  # duplicate is a no-op
    m.save()

    back = Manifest.load(path)
    assert "hunt9gfNW48" in back
    assert back["hunt9gfNW48"].tournaments == ["80nj8Ryn"]
    assert len(back) == 1


def test_rounds_for_video(tmp_path):
    """A video's round ids resolve from the manifest (raw input -> rounds)."""
    import json

    import pytest

    from chessqueries.annotate.manifest import rounds_for_video

    p = tmp_path / "videos.json"
    p.write_text(json.dumps({"abcdefghijk": {"round_ids": ["r7", "r8", "r9"]}}))
    assert rounds_for_video("abcdefghijk", videos_path=p) == ["r7", "r8", "r9"]
    with pytest.raises(KeyError):
        rounds_for_video("stuvwxyz012", videos_path=p)
