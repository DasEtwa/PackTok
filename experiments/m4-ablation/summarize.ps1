param(
    [Parameter(Mandatory=$true)][string]$InputDirectory,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
$lines = Get-Content -LiteralPath (Join-Path $InputDirectory 'run-summary.txt')
$rows = @($lines | Where-Object {$_ -like 'result *'} | ForEach-Object {
    $row = @{}
    foreach($m in [regex]::Matches($_,'([A-Za-z_]+)=([^ ]+)')) { $row[$m.Groups[1].Value] = $m.Groups[2].Value }
    $row
})
if($rows.Count -ne 24) { throw 'Expected all 24 completed seed/regime/variant results' }
function Number($row, $key) { [double]::Parse($row[$key],$culture) }
function FormatNumber($value,$digits=6) { $value.ToString("F$digits",$culture) }
function Mean($values) { ($values | Measure-Object -Average).Average }
function Summary($values,$digits=6) {
    $avg = Mean $values
    $variance = Mean @($values | ForEach-Object { ($_-$avg)*($_-$avg) })
    "$(FormatNumber $avg $digits) ± $(FormatNumber ([math]::Sqrt($variance)) $digits)"
}
$out = [Collections.Generic.List[string]]::new()
$out.Add('# Derived M4 tables')
$out.Add('')
$out.Add("Input: $InputDirectory/run-summary.txt. Arithmetic means and population SD; rounded log values are used. Raw logs and Rust summaries retain additional precision. Peak memory is the shared process high-water mark at each result, not isolated model memory. MACs exclude non-matrix work.")
foreach($regime in @('schedule','mac')) {
    $out.Add('');$out.Add("## $regime regime");$out.Add('')
    $out.Add('| Variant | Test bits/byte ± SD | Parameters | Updates per seed | MACs ± SD | Train ms ± SD | Test ms ± SD | Process peak B range |')
    $out.Add('|---|---:|---:|---|---:|---:|---:|---|')
    foreach($v in @('A','B','C','D')) {
        $selected=@($rows | Where-Object {$_.regime -eq $regime -and $_.variant -eq $v})
        if($selected.Count -ne 3) { throw 'Missing corresponding seed' }
        $bits=Summary @($selected|ForEach-Object{ Number $_ 'test_bits_byte' })
        $macs=Summary @($selected|ForEach-Object{ Number $_ 'macs' }) 1
        $train=Summary @($selected|ForEach-Object{ Number $_ 'train_ms' }) 3
        $test=Summary @($selected|ForEach-Object{ Number $_ 'test_ms' }) 3
        $peaks=@($selected|ForEach-Object{ if($_.peak_process_bytes -match '^Ok\((\d+)\)$') {[long]$Matches[1]} else {throw 'Peak unavailable'} })
        $range="$(( $peaks|Measure-Object -Minimum).Minimum)–$(( $peaks|Measure-Object -Maximum).Maximum)"
        $updates=($selected|ForEach-Object{$_.updates}) -join ', '
        $out.Add("| $v | $bits | $($selected[0].params) | $updates | $macs | $train | $test | $range |")
    }
    $out.Add('');$out.Add('| Seed | B-A | C-A | D-A | (D-C)-(B-A) |');$out.Add('|---|---:|---:|---:|---:|')
    $contrasts=@()
    foreach($seed in @('20261007','20261008','20261009')) {
        $selected=@($rows|Where-Object{$_.regime -eq $regime -and $_.seed -eq $seed})
        $val=@{};foreach($r in $selected){$val[$r.variant]=Number $r 'test_bits_byte'}
        $diff=@(($val.B-$val.A),($val.C-$val.A),($val.D-$val.A),(($val.D-$val.C)-($val.B-$val.A)))
        $contrasts+= ,$diff
        $cells=($diff|ForEach-Object{FormatNumber $_ 8}) -join ' | '
        $out.Add("| $seed | $cells |")
    }
    $cells=0..3|ForEach-Object{ $i=$_;Summary @($contrasts|ForEach-Object{$_[$i]}) 8 }
    $out.Add('| mean ± population SD | '+($cells -join ' | ')+' |')
}
$out.Add('');$out.Add('## Per-seed measurements');$out.Add('')
$out.Add('| Regime | Variant | Seed | Test bits/byte | NLL/byte | Loss/token | B/token | Updates | Targets | Train target bytes | MACs | Train ms | Curve validation ms | Final validation ms | Test ms | Model B | Head params | Parameter/Adam/gradient B |')
$out.Add('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
foreach($r in $rows) {
    $keys=@('regime','variant','seed','test_bits_byte','test_nll_byte','test_loss_token','bytes_token','updates','train_targets','train_target_bytes','macs','train_ms','curve_validation_ms','final_validation_ms','test_ms','model_bytes','head_params','parameter_adam_gradient_bytes')
    $out.Add('| '+(($keys|ForEach-Object{$r[$_]}) -join ' | ')+' |')
}
$out.Add('');$out.Add('## Raw tokenizer, head and generation observations');$out.Add('')
$out.Add('| Regime | Variant | NLL/byte mean ± SD | Loss/token mean ± SD | Final validation bits/byte mean ± SD | Final validation ms mean ± SD |')
$out.Add('|---|---|---:|---:|---:|---:|')
foreach($regime in @('schedule','mac')) {
    foreach($v in @('A','B','C','D')) {
        $selected=@($rows|Where-Object{$_.regime -eq $regime -and $_.variant -eq $v})
        $cells=@('test_nll_byte','test_loss_token','validation_bits_byte','final_validation_ms')|ForEach-Object{ $key=$_; Summary @($selected|ForEach-Object{Number $_ $key}) }
        $out.Add("| $regime | $v | "+($cells -join ' | ')+' |')
    }
}
$out.Add('');$out.Add('| Regime | Factorized variant | Pack accuracy mean ± SD | Local accuracy under gold pack mean ± SD | Active local rows | Total logits/target | Total test logits | Head parameters |')
$out.Add('|---|---|---:|---:|---:|---:|---:|---:|')
foreach($regime in @('schedule','mac')) {
    foreach($v in @('B','D')) {
        $headLines=@($lines|Where-Object{$_ -like "heads regime=$regime variant=$v *"})
        $heads=@($headLines|ForEach-Object{
            $h=@{};foreach($m in [regex]::Matches($_,'([A-Za-z_]+)=([^ ]+)')){$h[$m.Groups[1].Value]=$m.Groups[2].Value};$h
        })
        $pack=Summary @($heads|ForEach-Object{ Number $_ 'pack_accuracy' })
        $local=Summary @($heads|ForEach-Object{ Number $_ 'local_accuracy_given_gold_pack' })
        $packRows=[regex]::Matches($headLines[0],'pack\[\d+\]=\{targets:(\d+),accuracy:[^,]+,local_count:(\d+)\}')
        $totalLocal=0L;$targets=0L
        foreach($m in $packRows){$n=[long]$m.Groups[1].Value;$targets+=$n;$totalLocal+=$n*[long]$m.Groups[2].Value}
        $logits=$totalLocal+$targets*$packRows.Count
        $active=FormatNumber ($totalLocal/[double]$targets)
        $perTarget=FormatNumber ($logits/[double]$targets)
        $param=($rows|Where-Object{$_.regime -eq $regime -and $_.variant -eq $v}|Select-Object -First 1).head_params
        $out.Add("| $regime | $v | $pack | $local | $active | $perTarget | $logits | $param |")
    }
}
$out.Add('')
$out.Add('```text')
$lines|Where-Object{$_ -match '^(tokenizer_training|encode |mapping |heads |generation |process_peak_memory_bytes)'}|ForEach-Object{$out.Add($_)}
$out.Add('```');$out.Add('');$out.Add('## Learning curves');$out.Add('');$out.Add('```text')
$lines|Where-Object{$_ -like 'curve *'}|ForEach-Object{$out.Add($_)}
$out.Add('```')
$stream=[IO.File]::Open($OutputPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
try {
    $writer=[IO.StreamWriter]::new($stream,[Text.UTF8Encoding]::new($false));
    try {$out|ForEach-Object{$writer.WriteLine($_)}} finally {$writer.Dispose()}
} finally {$stream.Dispose()}
