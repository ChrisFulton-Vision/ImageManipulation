import logging

LOG = logging.getLogger("superCalibrate")

if not LOG.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    LOG.addHandler(handler)

    LOG.setLevel(logging.INFO)
    # LOG.setLevel(logging.DEBUG)
    # LOG.setLevel(logging.WARNING)