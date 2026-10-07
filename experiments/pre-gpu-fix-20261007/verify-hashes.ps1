$ErrorActionPreference = 'Stop'
$fixDir = Join-Path (Get-Location) 'experiments/pre-gpu-fix-20261007'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$records = [System.Collections.Generic.List[string]]::new()
foreach ($manifest in @(
    'experiments/m3-model/runs/final-20261007/SHA256SUMS.txt',
    'experiments/m3-model/runs/audit-20261007/SHA256SUMS.txt',
    'experiments/m4-ablation/SHA256SUMS.txt',
    'experiments/pre-gpu-review-20261007/SHA256SUMS.txt'
)) {
    $count = 0
    foreach ($line in Get-Content -LiteralPath $manifest) {
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { throw "Malformed hash entry in $manifest" }
        $expected = $Matches[1].ToLowerInvariant()
        $path = $Matches[2]
        $checkedPath = if ($path -eq 'PRE_GPU_CODE_REVIEW.md') { Join-Path $fixDir 'PRE_GPU_CODE_REVIEW-before.md' } else { $path }
        $actual = (Get-FileHash -LiteralPath $checkedPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) { throw "Historical hash changed: $path" }
        $count++
        $records.Add("OK $expected  $path")
    }
    $records.Add("manifest=$manifest matched=$count mismatches=0")
}
$before = @(Get-Content -LiteralPath (Join-Path $fixDir 'compatibility-before-windows-v2.txt') | Where-Object { $_ -match '^(golden |model |tokenizer |historical |complete )' })
$after = @(Get-Content -LiteralPath (Join-Path $fixDir 'windows-final/compatibility.txt') | Where-Object { $_ -match '^(golden |model |tokenizer |historical |complete )' })
if ($before.Count -ne $after.Count -or ($before -join "`n") -cne ($after -join "`n")) { throw 'Before/after compatibility or valid generation changed' }
$records.Add("before_after_windows_lines=$($before.Count) identical=true")
$modelHash = (Get-FileHash -LiteralPath crates/packtok-model/src/lib.rs -Algorithm SHA256).Hash.ToLowerInvariant()
$recordedModelHash = (Get-Content -LiteralPath (Join-Path $fixDir 'model-source-sha256.txt')).Split(' ')[0]
if ($modelHash -ne $recordedModelHash) { throw 'Verified model source changed after source snapshot' }
$records.Add("verified_model_source=$modelHash unchanged=true")
[IO.File]::WriteAllText((Join-Path $fixDir 'historical-hashes.txt'), ($records -join "`n") + "`n", $utf8)
$records | Where-Object { $_ -match '^(manifest=|before_after|verified_model)' }
