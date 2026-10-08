# Dedicated Drive OAuth: longevity and approval boundary

## Actual configuration observed on 2026-10-08

Google Cloud project PackTok has consent branding named **PackTok** and one
Desktop client named **PackTok WSL Backup**. These are distinct names. The audience
is **External / Testing**, with one test user. The active WSL remote is
`packtok-drive-own`; protected local checks confirm backend Drive, scope
`drive.file`, a refresh token, and the same client as the protected Desktop JSON.
No client ID, secret or token is recorded here.

Data Access initially declared no scopes even though rclone requested drive.file.
Maintenance declared exactly `https://www.googleapis.com/auth/drive.file`; Google
confirmed the save in the non-sensitive list. No additional grant, full Drive
access, new client or service account was introduced. The existing working
storage implementation remains in use.

Audience currently disables **Publish app** and states that Branding configuration
must be completed. Existing app name, support/developer contacts are populated;
homepage/privacy/terms/logo are empty. The interface does not identify the precise
missing field. No contact, domain, policy or verification claim was invented.
This is a publishing prerequisite, separate from working rclone authorization.

## Selected long-term transition, pending human approval

Retain the current Desktop client and drive.file. Move the existing External
audience to **In production** through Google Cloud's supported interface after
the Branding prerequisite is resolved. Obtain explicit human approval before
publishing: Production allows other Google Account users to authorize the app,
whereas Testing restricts authorization to listed test users. This does not
make existing Drive backups public; each account still grants access separately.
The original maintenance request explicitly requires this publishing approval.

After approved publication, perform one fresh rclone authorization for the same
remote/client/account to replace the Testing-issued refresh token, then another
harmless upload/full-download/size/SHA-256/completion-manifest test. Do not assume
a status change extends a token issued in Testing. Never distribute the client
credentials or submit verification materials without permission.

| Configuration | Suitability |
|---|---|
| External / Testing | Current; Drive consent and refresh tokens expire after seven days. Unsuitable for unattended multi-day storage. |
| External / In production | Selected route removes the Testing-specific limit for new authorization; revocation and other expiration rules remain. Human publishing approval required. |
| Internal | Requires an eligible organization and restricts users to it; not selected for this personal project. |
| Service account / full Drive scope | Unnecessary for existing scoped backups; not selected. |

Google classifies drive.file as non-sensitive and recommends it for per-file
access. Google states that app verification is not mandatory when only
non-sensitive scopes are used; displaying the app name/logo can need the lighter
brand-verification process. Production is not synonymous with verified, and
publishing alone does not guarantee name/logo display or eligibility for every
policy exception. Use the actual Verification Center and requested scope set
when determining requirements; no policy exemption was assumed for this app.
No sensitive or restricted scope is needed by this workflow.

Sources checked during maintenance:
[Google Audience](https://support.google.com/cloud/answer/15549945?hl=en),
[Drive scopes](https://developers.google.com/workspace/drive/api/guides/api-specific-auth),
[OAuth verification FAQ](https://support.google.com/cloud/answer/9110914?hl=en).

## Token lifetime and account isolation

Refresh tokens are not permanent. Google lists revocation, six months of
non-use, excessive token issuance, time-limited consent and administrator policy
as invalidation causes. The limit is 100 live refresh tokens per account/client;
issuing another can invalidate the oldest. Password changes affect tokens with
Gmail scopes, which this workflow does not request. Access-token refresh and
refresh-token lifetime are different. rclone stores updated token data locally;
do not repeatedly reconnect to mask an unresolved error or promise automatic
recovery from revoked consent.

Use the intended Google account and same Desktop client. A different account
has a different Drive; a new OAuth client may not see the old client's files
under drive.file. On migration, preserve that app identity or explicitly select/
download the existing files through Google Drive. Do not widen scope to recover
visibility. See [Google token rules](https://developers.google.com/identity/protocols/oauth2)
and [OAuth best practices](https://developers.google.com/identity/protocols/oauth2/resources/best-practices).

The Desktop flow uses the supported browser authorization and loopback redirect
(`127.0.0.1` locally, rclone's port 53682), with offline access. It is not an
out-of-band code flow or a web-client arbitrary redirect. No public callback,
domain or additional redirect is installed. See
[Google Desktop OAuth](https://developers.google.com/identity/protocols/oauth2/native-app)
and [rclone Drive/client setup](https://rclone.org/drive/#making-your-own-client-id).

## Detect, reauthorize and migrate

A nonzero scoped rclone read/upload or invalid_grant/token-revocation error requires
reauthorization, not a new GPU. The backup tool records INCOMPLETE and refuses a
completion claim. Raw auth errors and browser URLs remain private. Distinguish
HTTP quota failures from token invalidation.

1. In dedicated Ubuntu-24.04, protect a local copy of the current rclone config
   outside Git before a reconnect. Preserve it through encrypted offline storage
   under user control; a 0600 file is not an encrypted backup.
2. Reconnect interactively with `rclone config reconnect packtok-drive-own:`,
   selecting the same account/client and checking that consent stays drive.file.
   Raw OAuth output must not enter public logs. Do not revoke the working grant
   unless rotation/revocation is intended.
3. Confirm scope and identity without dumping configuration or tokens. Run the
   documented harmless integrity test before depending on unattended backup.
4. On Windows/WSL migration, restore the protected config and client JSON into
   the dedicated user's private directories (0700, files 0600), or reauthorize
   the same client. Do not place secrets on GitHub or inside the checkout.
5. In a fresh environment, use the public external-artifact manifest's namespace
   and checksums with the same authorized client/account, download completion and
   payload, then verify both hashes and size before loading weights. Restoring
   weights is not exact training resume.

[Storage and recovery](storage-and-recovery.md) contains executable restore
commands, external manifests and the remaining GPU checkpoint limitation.
A successful maintenance readback proves current integrity, not seven-day or
multi-month authentication. OAuth longevity remains pending publishing readiness
and explicit approval; no long unattended run is declared ready.
