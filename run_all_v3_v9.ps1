# run_all_v3_v9.ps1 — v3~v9 순차 실행 스크립트
# 사용법: conda activate fruit 후 실행
# 예상 소요시간: 버전당 약 8~12분, 총 약 60~90분

$scripts = @("v3_batch128.py", "v4_batch256.py", "v5_lr0005.py", "v6_lr003.py", "v7_deeper_resblock.py", "v8_dropout.py", "v9_adamw.py")
$start_total = Get-Date

foreach ($script in $scripts) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Running: $script" -ForegroundColor Yellow
    Write-Host "  Time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host "========================================" -ForegroundColor Cyan

    $start = Get-Date
    python $script
    $exit_code = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    if ($exit_code -eq 0) {
        Write-Host "  [OK] $script completed in $($elapsed.ToString('mm\:ss'))" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] $script failed (exit code: $exit_code)" -ForegroundColor Red
        Write-Host "  Stopping..." -ForegroundColor Red
        break
    }
}

$total_elapsed = (Get-Date) - $start_total
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All done! Total: $($total_elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "  Submissions saved in: Data/submissions/" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
