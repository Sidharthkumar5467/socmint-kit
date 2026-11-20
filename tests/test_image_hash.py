"""analyze.image_hash: download + perceptual-hash avatars/thumbnails for coordination's
reused-image signal (the same image under a different URL). Downloads are injected, so the
test does no network."""

from __future__ import annotations

import io

import pytest

from socmint.analyze import image_hash as ih
from socmint.analyze.coordination import phash_distance
from socmint.schema import Media, Post

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("imagehash")


def _png(seed: int, size: int = 64) -> bytes:
    """A deterministic non-uniform image (a diagonal split), distinct per seed."""
    im = Image.new("RGB", (size, size), (seed % 256, (seed * 7) % 256, (seed * 13) % 256))
    for y in range(size):
        for x in range(y):
            im.putpixel((x, y), ((seed * 3) % 256, (seed * 5) % 256, (seed * 11) % 256))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _post(pid: str, author: str, avatar: str | None = None, media_url: str | None = None) -> Post:
    return Post(post_id=pid, source="x", kind="post", text="t", url=f"https://x.test/{pid}",
                author_id=author, author_handle=f"@{author}", author_avatar_url=avatar,
                collected_at="2026-11-20T00:00:00Z",
                media=[Media(type="photo", url=media_url)] if media_url else [])


def test_same_image_different_urls_hashes_equal():
    same_a = "https://cdn.a/pic_a.jpg"
    same_b = "https://cdn.b/pic_b.jpg"   # different URL, SAME image bytes
    other = "https://cdn.c/pic_c.jpg"
    blob = {same_a: _png(1), same_b: _png(1), other: _png(99)}
    posts = [_post("x:1", "alice", avatar=same_a), _post("x:2", "bob", avatar=same_b),
             _post("x:3", "carol", avatar=other)]

    hashes = ih.compute_image_hashes(posts, fetch=lambda url, _t: blob.get(url))
    assert set(hashes) == {same_a, same_b, other}
    # the re-uploaded identical image hashes the same under both URLs
    assert phash_distance(hashes[same_a], hashes[same_b]) == 0
    # a different image is far away
    assert phash_distance(hashes[same_a], hashes[other]) > 8


def test_candidate_urls_avatars_first_dedup_and_skip_placeholders():
    posts = [
        _post("x:1", "a", avatar="https://cdn/default_profile_images/x.png"),  # placeholder
        _post("x:2", "b", avatar="https://cdn/real_avatar.png", media_url="https://cdn/m1.jpg"),
        _post("x:3", "b", avatar="https://cdn/real_avatar.png"),               # dup avatar
        _post("x:4", "c", media_url="https://cdn/m2.jpg"),
    ]
    urls = ih.candidate_urls(posts)
    assert "https://cdn/default_profile_images/x.png" not in urls  # placeholder skipped
    assert urls.count("https://cdn/real_avatar.png") == 1          # deduped
    # avatars come before media
    assert urls.index("https://cdn/real_avatar.png") < urls.index("https://cdn/m1.jpg")
    assert set(urls) == {"https://cdn/real_avatar.png", "https://cdn/m1.jpg",
                         "https://cdn/m2.jpg"}


def test_cap_limits_the_number_hashed():
    posts = [_post(f"x:{i}", f"u{i}", avatar=f"https://cdn/a{i}.png") for i in range(10)]
    seen: list[str] = []

    def fetch(url, _t):
        seen.append(url)
        return _png(hash(url) % 1000)

    hashes = ih.compute_image_hashes(posts, cap=3, fetch=fetch)
    assert len(seen) == 3 and len(hashes) == 3


def test_fetch_failures_and_non_images_are_skipped_not_fatal():
    posts = [_post("x:1", "a", avatar="https://cdn/ok.png"),
             _post("x:2", "b", avatar="https://cdn/dead.png"),
             _post("x:3", "c", avatar="https://cdn/notimage.png")]

    def fetch(url, _t):
        if url.endswith("ok.png"):
            return _png(5)
        if url.endswith("dead.png"):
            return None            # download failed
        return b"this is not an image"

    hashes = ih.compute_image_hashes(posts, fetch=fetch)
    assert set(hashes) == {"https://cdn/ok.png"}  # only the decodable image is hashed
