# Regenerate figures from results/ and build paper/paper.pdf.
#   powershell -ExecutionPolicy Bypass -File .\make_paper.ps1
# PowerShell does not accept '&&' as a statement separator; use ';' or this.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "[1/3] regenerating figures from results\ ..." -ForegroundColor Cyan
python analyse.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "analyse.py failed. Run the experiments first." -ForegroundColor Red
    exit 1
}

if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) {
    Write-Host "`npdflatex is not installed. The figures ARE now in figs\." -ForegroundColor Yellow
    Write-Host "  - install MiKTeX from https://miktex.org/download and rerun, or"
    Write-Host "  - upload paper\paper.tex and figs\ to overleaf.com"
    exit 0
}

# Delete the old PDF first, so a stale one can never be mistaken for success.
Set-Location (Join-Path $root "paper")
Remove-Item -ErrorAction SilentlyContinue paper.pdf

Write-Host "[2/3] compiling (pass 1 of 2) ..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -halt-on-error paper.tex | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pdflatex failed. See paper\paper.log" -ForegroundColor Red; exit 1
}
Write-Host "[3/3] compiling (pass 2 of 2, resolves references) ..." -ForegroundColor Cyan
pdflatex -interaction=nonstopmode -halt-on-error paper.tex | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "pdflatex failed. See paper\paper.log" -ForegroundColor Red; exit 1
}

$undef = Select-String -Path paper.log -Pattern "Undefined control sequence|LaTeX Warning: Reference"
if ($undef) { Write-Host "`nLaTeX warnings:" -ForegroundColor Yellow; $undef | ForEach-Object { Write-Host "  $_" } }

if (Test-Path "paper.pdf") {
    Remove-Item -ErrorAction SilentlyContinue paper.aux, paper.out
    Write-Host "`nDone -> paper\paper.pdf" -ForegroundColor Green
    Invoke-Item "paper.pdf"
} else {
    Write-Host "No PDF produced. See paper\paper.log" -ForegroundColor Red; exit 1
}
