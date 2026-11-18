import os
import subprocess
import sys

# Results must not depend on Python's per-process string-hash randomisation (set ordering).
# Re-launch once with a fixed seed so every run of the same inputs gives the same output.
# (subprocess rather than os.execv: execv mangles paths with spaces on Windows)
if os.environ.get("PYTHONHASHSEED") != "0":
    env = dict(os.environ, PYTHONHASHSEED="0")
    sys.exit(subprocess.call([sys.executable, "-m", "spendtool"] + sys.argv[1:], env=env))

from .cli import main  # noqa: E402

sys.exit(main())
