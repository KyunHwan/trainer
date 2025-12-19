import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

POLICY = os.path.join(ROOT, "policy_constructor")
if POLICY not in sys.path:
    sys.path.insert(0, POLICY)
