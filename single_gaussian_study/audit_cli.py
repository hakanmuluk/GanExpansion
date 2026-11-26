"""Supply a missing JSON logging import without editing frozen audit source."""
import json
from . import audit
def main():
    # audit_cell and the complete invariant checks already run unchanged.
    # Only audit.main's final JSON progress print lacked this global import.
    audit.json=json
    audit.main()
if __name__=='__main__':main()
