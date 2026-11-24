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
The later [paired pilot](../../experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T154700Z/PILOT_RESULT.md)
recovered A's full training checkpoint and verified the final A/C results archive
by complete Drive readback.

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

For a future actual M5 model checkpoint, additionally verify configuration,
corpus/tokenizer identity and use the CPU `recover_weights` example described in
[RECOVERY_STORAGE](../../experiments/m5-gpu/provenance/RECOVERY_STORAGE.md).
Use that example for weights-only initialization. The M5 extended runner's
`latest.resume.safetensors` contains AdamW and sampler state. New M6 runs use
immutable `.ptc` generations. See [checkpoint formats and recovery](../milestones/M6.md).

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
