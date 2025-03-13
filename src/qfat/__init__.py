import os
import sys

from qfat.constants import PROJECT_PATH

# Add relay-policy-learning to the system path
sys.path.append(os.path.abspath(str(PROJECT_PATH / "relay-policy-learning")))
