# Recovery preparation delivery

Preparation commit: d7e4fe2b948fd37fbe96d8fe458a9302e3802ca9.
File-permission-only follow-up: e69c127261c6d1f4c6bc411611702cfa8d0d0ffb.
Scientific inputs, v4 archive/executable and all historical artifact bytes are
unchanged; CPU verification is recorded in the checks/delivery directories.

Native WSL's git push stalled without updating the remote and was terminated.
A local Git bundle containing only the two public preparation commits transferred
them to Windows. Windows staged content was checked against every incoming blob
before advancing its checkout; only file-mode differences were allowed. Windows
then pushed the existing m5-gpu-transformer branch successfully. No force push,
historical artifact replacement, credential export or new PR occurred.

Both native Ubuntu-24.04 and Windows checkouts were clean at e69c127, tracking
the same origin branch. GitHub PR #5 remains OPEN/DRAFT with the original base
m4-factorization-ablation and its updated blocker/validation description.

Final read-only Colab inventory again returned no active sessions, balance
167.55 CU, rate 0.00/hour and zero assignments. There have been zero new GPU
requests and zero allocated GPU seconds in this task; the single recovery
authorization remains unused. Browser OAuth using the shared rclone client
completed, but independent artifact upload/download/SHA verification is blocked
by that client's Google API quota. No dedicated client JSON path has yet been
provided. No GPU/model/weight-recovery or research-quality result is claimed.

The required next input is the local path to the user's Desktop OAuth-client
JSON, outside the repository. No credential/token text belongs in chat or Git.
After that authentication, the independent fixture must pass before sealing
PREFLIGHT_READY.json and allocating the one bounded L4. Full training is locked.

Current operational state: M5_BLOCKED — INFRASTRUCTURE.
