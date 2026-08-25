param(
    [switch]$SkipDownload,
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is not available on PATH"
}

if (-not $SkipDownload) {
    conda run -n daily --no-capture-output python -u scripts\download_gff.py `
        --root data\gff `
        --components base glofas dem hand era5 s1 `
        --workers 24 `
        --part-mb 16
    if ($LASTEXITCODE -ne 0) { throw "GFF download or verification failed" }
}

conda run -n daily --no-capture-output python scripts\audit_gff.py `
    --root data\gff `
    --output outputs\gff_data_audit.json
if ($LASTEXITCODE -ne 0) { throw "GFF audit failed" }

conda run -n daily --no-capture-output python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed" }

conda run -n daily --no-capture-output python -m src.train_gff `
    --config configs\gff_horizonformer_smoke.yaml
if ($LASTEXITCODE -ne 0) { throw "GFF smoke experiment failed" }

if (-not $SmokeOnly) {
    conda run -n daily --no-capture-output python -m src.train_gff `
        --config configs\gff_horizonformer.yaml
    if ($LASTEXITCODE -ne 0) { throw "GFF subset experiment failed" }

    conda run -n daily --no-capture-output python scripts\select_gff_checkpoint.py `
        --config configs\gff_horizonformer.yaml `
        --checkpoints `
            outputs\gff_horizonformer\epoch1.pt `
            outputs\gff_horizonformer\epoch2.pt `
            outputs\gff_horizonformer\epoch3.pt `
            outputs\gff_horizonformer\epoch4.pt `
            outputs\gff_horizonformer\epoch5.pt `
            outputs\gff_horizonformer\epoch6.pt
    if ($LASTEXITCODE -ne 0) { throw "GFF checkpoint selection failed" }

    conda run -n daily --no-capture-output python scripts\predict_gff_example.py `
        --config configs\gff_horizonformer.yaml `
        --checkpoint outputs\gff_horizonformer\best.pt `
        --horizon 3 `
        --output outputs\gff_horizonformer\prediction_72h.png
    if ($LASTEXITCODE -ne 0) { throw "Prediction rendering failed" }
}

Write-Host "GFF experiment workflow completed."
