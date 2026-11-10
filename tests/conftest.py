"""Suite-wide settings."""
import os

# Issue #164: every exception the plugin catches must fail the suite
# instead of degrading to an unchecked result. NUMTY_DEBUG=1 makes
# `_swallowed` re-raise, which mypy reports as an INTERNAL ERROR. Set at
# import time so the in-process mypy batch and every subprocess (which
# inherit this environment) run with it. A test that needs the
# production behaviour removes the variable for its own subprocess.
os.environ["NUMTY_DEBUG"] = "1"
