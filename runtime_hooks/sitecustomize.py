"""Load the opt-in optimizer update probe in every Python and Ray worker."""

from src.agents.update_probe import install_from_environment

install_from_environment()
