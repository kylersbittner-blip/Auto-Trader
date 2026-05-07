from fastapi import Header, HTTPException, status

from config import get_settings


async def require_control_key(x_control_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.control_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CONTROL_API_KEY is required before broker-control endpoints can be used.",
        )

    if x_control_key != settings.control_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing control key.",
        )
