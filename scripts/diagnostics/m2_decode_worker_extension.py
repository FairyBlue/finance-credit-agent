"""Named public worker extension; delegates only to read-only observer hooks."""

from m2_decode_observer import install_observer, remove_observer


class DecodeObserverExtension:
    def fca_install_decode_observer(self, path):
        return install_observer(self, path)

    def fca_remove_decode_observer(self):
        return remove_observer(self)
