"""Stand-in for a run that hangs by itself the first time, and finishes once restarted.
argv[1] is a marker file: missing means first launch (print, then hang), present means second launch (finish)."""
import pathlib
import sys
import time

flag = pathlib.Path(sys.argv[1])
if flag.exists():
    print("generation 2 done", flush=True)
    print("search finished", flush=True)
else:
    flag.write_text("launched once")
    print("generation 1 done", flush=True)
    time.sleep(3600)
