import html
from urllib.parse import urlparse
from typing import Iterator, Dict, Any, Optional

def _clean(u: str) -> str:
    return html.unescape(u.strip()) if isinstance(u, str) else u

def _host(u: str) -> str:
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def _is_video(u: str) -> bool:
    return "v.redd.it" in _host(u)

def smart_variants(u: str) -> Iterator[str]:
    """
    Yield original URL first, then fallback guesses for i.redd.it/<id> (no extension).
    Skip obvious video hosts.
    """
    if not isinstance(u, str): return
    u = _clean(u)
    if not u: return
    if _is_video(u): return
    yield u
    pu = urlparse(u)
    fname = pu.path.split("/")[-1]
    if "i.redd.it" in _host(u) and "." not in fname:
        yield u + ".jpg"
        yield u + ".png"

def _from_media_metadata(post: Dict[str, Any]) -> Optional[str]:
    mm = post.get("media_metadata")
    gd = post.get("gallery_data", {})
    items = gd.get("items") if isinstance(gd, dict) else None
    if not isinstance(mm, dict) or not isinstance(items, list):
        return None
    for it in items:
        mid = it.get("media_id")
        meta = mm.get(mid) if mid else None
        if isinstance(meta, dict):
            s = meta.get("s")
            if isinstance(s, dict) and s.get("u"):
                return _clean(s["u"])
            p = meta.get("p")
            if isinstance(p, list) and p and isinstance(p[0], dict) and p[0].get("u"):
                return _clean(p[0]["u"])
    return None

def _from_preview(post: Dict[str, Any]) -> Optional[str]:
    prev = post.get("preview")
    if isinstance(prev, dict):
        imgs = prev.get("images")
        if isinstance(imgs, list) and imgs:
            src = imgs[0].get("source")
            if isinstance(src, dict) and src.get("url"):
                return _clean(src["url"])
    return None

def _from_crosspost(post: Dict[str, Any]) -> Optional[str]:
    cpl = post.get("crosspost_parent_list")
    if isinstance(cpl, list):
        for cp in cpl:
            for fn in (_from_media_metadata, _from_preview):
                u = fn(cp)
                if u: return u
            if isinstance(cp.get("imageUrls"), list):
                for cand in cp["imageUrls"]:
                    if isinstance(cand, str) and cand.strip():
                        return _clean(cand)
            for key in ("url", "externalLink"):
                u = cp.get(key)
                if isinstance(u, str) and u.strip():
                    return _clean(u)
    return None

def candidates_for_post(post: Dict[str, Any]) -> Iterator[str]:
    """
    Priority: gallery (media_metadata) -> preview -> crosspost -> imageUrls -> url/externalLink
    """
    for fn in (_from_media_metadata, _from_preview, _from_crosspost):
        u = fn(post)
        if u:
            yield _clean(u)
    if isinstance(post.get("imageUrls"), list):
        for u in post["imageUrls"]:
            if isinstance(u, str) and u.strip():
                yield _clean(u)
    for key in ("url", "externalLink"):
        u = post.get(key)
        if isinstance(u, str) and u.strip():
            yield _clean(u)
