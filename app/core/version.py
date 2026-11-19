"""Single source of truth for application version."""

__version__ = "1.2.4"
__version_info__ = tuple(int(part) for part in __version__.split("."))


def get_version() -> str:
    return __version__
