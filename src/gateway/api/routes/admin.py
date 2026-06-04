"""Self-hosted operator dashboard for standalone gateway mode."""

from functools import lru_cache
from importlib.resources import files

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse, Response

router = APIRouter(tags=["admin"])

_ASSET_MEDIA_TYPES = {
    "styles.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "api.js": "text/javascript; charset=utf-8",
    "dom.js": "text/javascript; charset=utf-8",
    "format.js": "text/javascript; charset=utf-8",
    "modal.js": "text/javascript; charset=utf-8",
    "render.js": "text/javascript; charset=utf-8",
}


@lru_cache(maxsize=1)
def _admin_index_html() -> str:
    return files("gateway.admin").joinpath("index.html").read_text(encoding="utf-8")


@lru_cache(maxsize=len(_ASSET_MEDIA_TYPES))
def _admin_asset_bytes(asset_name: str) -> bytes:
    return files("gateway.admin").joinpath(asset_name).read_bytes()


@router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
@router.get("/admin/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard() -> str:
    """Return the standalone gateway operator dashboard."""
    return _admin_index_html()


@router.get("/admin/assets/{asset_name}", include_in_schema=False)
async def admin_asset(asset_name: str) -> Response:
    """Return a packaged admin frontend asset."""
    media_type = _ASSET_MEDIA_TYPES.get(asset_name)
    if media_type is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin asset not found")
    return Response(content=_admin_asset_bytes(asset_name), media_type=media_type)
