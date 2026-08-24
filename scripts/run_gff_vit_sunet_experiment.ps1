param(
    [switch]$SmokeOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is not available on PATH"
}

conda run -n daily --no-capture-output python -m unittest tests.test_gff_pipeline -v
if ($LASTEXITCODE -ne 0) { throw "GFF pipeline tests failed" }

conda run -n daily --no-capture-output python -u -m src.train_gff `
    --config configs\gff_vit_sunet_smoke.yaml
if ($LASTEXITCODE -ne 0) { throw "ViT SU-Net smoke experiment failed" }

if (-not $SmokeOnly) {
    conda run -n daily --no-capture-output python -u -m src.train_gff `
        --config configs\gff_vit_sunet.yaml
    if ($LASTEXITCODE -ne 0) { throw "ViT SU-Net subset experiment failed" }

    conda run -n daily --no-capture-output python -u scripts\select_gff_checkpoint.py `
        --config configs\gff_vit_sunet.yaml `
        --checkpoints `
            outputs\gff_vit_sunet\epoch1.pt `
            outputs\gff_vit_sunet\epoch2.pt `
            outputs\gff_vit_sunet\epoch3.pt `
            outputs\gff_vit_sunet\epoch4.pt
    if ($LASTEXITCODE -ne 0) { throw "ViT SU-Net checkpoint selection failed" }

    conda run -n daily --no-capture-output python scripts\predict_gff_example.py `
        --config configs\gff_vit_sunet.yaml `
        --checkpoint outputs\gff_vit_sunet\best.pt `
        --horizon 3 `
        --output outputs\gff_vit_sunet\prediction_72h.png
    if ($LASTEXITCODE -ne 0) { throw "ViT SU-Net prediction rendering failed" }
}

Write-Host "GFF ViT SU-Net experiment workflow completed."
