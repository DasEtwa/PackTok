The first maintenance Python subprocess launches inherited the non-login WSL PATH, which omitted ~/.cargo/bin. cargo was not found before any check ran. Empty first fmt logs are retained. The corrected v2 checks explicitly prepend the dedicated user toolchain/bin paths; no host configuration changed.

The initial CLI smoke check used the package name packtok-cli as executable filename; the declared binary is packtok. No CLI command ran in that failed invocation. The corrected smoke uses target/release/packtok.
