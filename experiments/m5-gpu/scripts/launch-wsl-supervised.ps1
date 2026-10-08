param(
    [ValidateSet('recovery', 'cpu-mock')][string]$Mode = 'recovery',
    [string]$Fixture,
    [string]$MockMode = 'pass',
    [int]$MockSeconds = 310,
    [int]$WorkSeconds = 340,
    [int]$RuntimeSeconds = 360,
    [Parameter(Mandatory = $true)][string]$Receipt
)
$ErrorActionPreference = 'Stop'
$Receipt = [System.IO.Path]::GetFullPath($Receipt)
if ((Test-Path -LiteralPath $Receipt) -or
    -not (Test-Path -LiteralPath (Split-Path -Parent $Receipt) -PathType Container)) {
    throw 'A new receipt in an existing directory is required before dispatch'
}
$arguments = @('-d', 'Ubuntu-24.04', '-u', 'dasetwa', '--', 'bash',
    '/home/dasetwa/projects/PackTok/experiments/m5-gpu/scripts/launch-l4-recovery.sh')
if ($Mode -eq 'cpu-mock') {
    if ($Fixture -notmatch '^/home/dasetwa/projects/PackTok/docs/maintenance/[A-Za-z0-9_/-]+$' -or
        $MockMode -notmatch '^[a-z]+$' -or $MockSeconds -lt 0 -or $MockSeconds -gt 1200 -or
        $WorkSeconds -lt 1 -or $WorkSeconds -gt 1620 -or
        $RuntimeSeconds -lt 1 -or $RuntimeSeconds -gt 1680) {
        throw 'Invalid bounded CPU mock configuration'
    }
    $arguments += @('--cpu-mock', $Fixture, $MockMode, $MockSeconds, $WorkSeconds, $RuntimeSeconds)
}
# Hidden independent Windows client owns only this bounded WSL service. Its
# lifetime does not depend on the invoking Codex tool or PowerShell process.
# No permanent daemon, WSL setting, scheduled task or linger is installed.
$client = Start-Process -FilePath "$env:SystemRoot/System32/wsl.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru
@{ host_client_pid = $client.Id; initiating_pid = $PID; mode = $Mode;
   start_utc = [DateTime]::UtcNow.ToString('o'); distribution = 'Ubuntu-24.04';
   fixture = $Fixture; runtime_seconds = $(if ($Mode -eq 'cpu-mock') {$RuntimeSeconds} else {1680});
   max_stop_seconds = $(if ($Mode -eq 'cpu-mock') {5} else {120})
} | ConvertTo-Json | Set-Content -LiteralPath $Receipt -Encoding utf8
Write-Output "Bounded WSL client PID $($client.Id); receipt $Receipt"
