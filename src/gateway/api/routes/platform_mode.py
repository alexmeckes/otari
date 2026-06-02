from fastapi import APIRouter, HTTPException, status

_DISABLED_DETAIL = "This endpoint is not available in platform mode. Manage this resource via the platform UI."

_DISABLED_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

router = APIRouter(tags=["platform-mode"])


@router.api_route("/v1/users/{path:path}", methods=_DISABLED_METHODS)
@router.api_route("/v1/users", methods=_DISABLED_METHODS)
@router.api_route("/v1/keys/{path:path}", methods=_DISABLED_METHODS)
@router.api_route("/v1/keys", methods=_DISABLED_METHODS)
@router.api_route("/v1/budgets/{path:path}", methods=_DISABLED_METHODS)
@router.api_route("/v1/budgets", methods=_DISABLED_METHODS)
@router.api_route("/v1/spend/{path:path}", methods=_DISABLED_METHODS)
@router.api_route("/v1/spend", methods=_DISABLED_METHODS)
async def local_management_disabled() -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_DISABLED_DETAIL)
