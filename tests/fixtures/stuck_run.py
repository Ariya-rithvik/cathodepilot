"""Stand-in for an XtalOpt run whose optimiser job hangs: prints generations, then at generation 3 starts a
child job that never returns. If the child is killed, the run marks that structure failed and carries on."""
import subprocess
import sys
import time

for gen in range(1, 7):
    if gen == 3:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        code = child.wait()
        print(f"generation {gen}: optimiser job exited with {code}, structure marked failed", flush=True)
    print(f"generation {gen} done", flush=True)
    time.sleep(0.4)
print("search finished", flush=True)
