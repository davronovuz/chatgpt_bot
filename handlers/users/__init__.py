from handlers.users.echo import echo_router
from handlers.users.start import user_router

routers_list = [
    user_router,
    echo_router,
]

__all__ = [
    "routers_list",
]
