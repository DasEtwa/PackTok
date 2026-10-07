$ErrorActionPreference = 'Stop'
$fixDir = Join-Path (Get-Location) 'experiments/pre-gpu-fix-20261007'
$logDir = Join-Path $fixDir 'windows-final'
if (Test-Path -LiteralPath $logDir) { throw 'Refusing to overwrite verification logs' }
New-Item -ItemType Directory -Path $logDir | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
$environment = @(& git rev-parse HEAD; & git status --short; & rustc --version; & cargo --version)
[IO.File]::WriteAllText((Join-Path $logDir 'environment.txt'), ($environment -join "`n") + "`n", $utf8)
$commands = @(
    @('fmt','--all','--check'),
    @('clippy','--workspace','--all-targets','--all-features','--','-D','warnings'),
    @('test','--workspace'),
    @('test','--workspace','--release'),
    @('build','--release','--workspace'),
    @('test','-p','packtok-model','serialization_'),
    @('test','-p','packtok-model','generation_validation_'),
    @('test','-p','packtok-model','recurrent_and_both_head_gradients_match_finite_differences'),
    @('test','-p','packtok-model','adam_'),
    @('run','--release','--manifest-path','experiments/pre-gpu-fix-20261007/Cargo.toml','--target-dir','target/pre-gpu-fix-compatibility')
)
$labels = @('fmt','clippy','tests-debug','tests-release','build-release','serialization','generation','gradients','optimizer','compatibility')
for ($index = 0; $index -lt $commands.Count; $index++) {
    $arguments = $commands[$index]
    # Native Cargo diagnostics use stderr even on success.
    $ErrorActionPreference = 'Continue'
    $output = & cargo @arguments 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    [IO.File]::WriteAllText((Join-Path $logDir ($labels[$index] + '.txt')), (($output | ForEach-Object { $_.ToString() }) -join "`n") + "`n", $utf8)
    Write-Output "$($labels[$index]): exit=$code"
    if ($code -ne 0) { exit $code }
}
