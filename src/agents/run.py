"""Generic experiment entrypoint used by the serial queue."""

from .launcher import cli

if __name__ == "__main__":
    raise SystemExit(cli())
