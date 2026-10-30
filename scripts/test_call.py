"""Place ONE real CALL-E call to SUPERVISOR_PHONE (rings a real phone, spends credit; needs DRY_RUN=0)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cathodepilot.agent import load_env  # noqa: E402
from cathodepilot.phone import phone_supervisor  # noqa: E402

load_env(Path(__file__).resolve().parents[1] / ".env")
print(phone_supervisor()(question="This is a test call. Can you hear me?",
                         proposed_action="nothing, this is only a test", reason="checking the phone line works"))
