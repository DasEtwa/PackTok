# Storage and recovery

Independent WSL rclone transport remains the supported backup path. It does not
depend on a Colab mount or allocation. Secrets reside under the dedicated WSL
user's protected configuration, outside the repository; see [OAuth](oauth.md).

The original workflow is
[drive-backup.py](../../experiments/m5-gpu/scripts/drive-backup.py).
It creates a unique namespace, uses immutable copy (never sync/delete), downloads
the complete payload, verifies size/SHA-256 and rehashes the source, then uploads
and downloads the completion manifest. A failed/partial candidate remains
INCOMPLETE. Only completed, trusted receipts identify restorable artifacts.

[External artifact manifest](../maintenance/2026-10-08/external-artifacts.json)
records known completion hashes, sizes and payload hashes. It contains PackTok
research identity and scoped rclone names, no authenticated Drive links or secrets.
Historical bundle hashes and locally retained evidence remain in
[RECOVERY_STORAGE](../../experiments/m5-gpu/provenance/RECOVERY_STORAGE.md).
The 2026-10-09 paired A/C pilot results archive (logs, metrics, checkpoints;
322,912,151 bytes, SHA-256 `d7da7a90…d1c25b4`) was uploaded to Drive and fully
read back with a matching hash; its receipt is
[DRIVE_BACKUP.json](../../experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T154700Z/DRIVE_BACKUP.json).
That receipt uses a different schema and the `PackTok/M5/pilot/` namespace and is
not listed in the external artifact manifest, so `restore-drive-artifact.py`
below does not accept it yet; restore it manually with rclone and verify the
recorded SHA-256.

## Restore in a fresh authorized environment

Install the documented rclone version or a compatible official release, restore
protected configuration or reauthorize the same Desktop client/account, then:

```sh
python3 experiments/m5-gpu/scripts/restore-drive-artifact.py \
  docs/maintenance/2026-10-08/external-artifacts.json RUN_ID \
  /home/dasetwa/packtok-restored/UNIQUE_DIRECTORY
```

Select an actual run_id from the manifest; do not type the placeholders literally.
The tool requires a new destination directory, downloads the trusted completion
manifest and entire payload, and verifies manifest SHA-256, identity, status, size
and payload SHA-256. It never lists unrelated Drive files, broadens scope or
deletes existing files. Reject a mismatch; do not load partial weights.
The maintenance restore uses a new directory and fresh CLI processes while
retaining the existing protected account/client; it is not a reinstall/new-account
authorization test.

For an M5 model checkpoint, additionally verify configuration and
corpus/tokenizer identity. Two checkpoint kinds exist (see [FORMAT](../../FORMAT.md)).
Weights-only preflight checkpoints can be checked with the CPU `recover_weights`
example described in
[RECOVERY_STORAGE](../../experiments/m5-gpu/provenance/RECOVERY_STORAGE.md); they
cannot resume training exactly. Resumable pilot checkpoints
(`latest.resume.safetensors`) also store AdamW moments, optimizer step, sampling
RNG and data cursor, and are read by `ResumableAdamW::load_checkpoint`; the
pilot's A run was reloaded this way from its step-2000 checkpoint for final
evaluation without retraining. `recover_weights` does not accept them.

## Protected configuration recovery

Keep encrypted offline copies of rclone configuration and the Desktop client JSON
in user-controlled storage outside both checkouts and public Drive namespaces.
Restore files only into the dedicated WSL user's private directories, with
directory 0700/file 0600, or use fresh same-client authorization.
A copied Windows checkout is not a backup of these protected WSL credentials.
No private configuration backup is created or uploaded by this maintenance task.

Expired/revoked tokens do not delete existing backup payloads. Reconnect the same
client/account, verify a known receipt, and then test a new immutable upload.
If the original client is lost, use explicit user-selected file access or manual
Drive download; do not switch to full Drive scope for convenience. Detailed
expiration, rotation and migration limits are in [OAuth](oauth.md).
