param(
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda is not available on PATH"
}

conda run -n daily --no-capture-output python -m unittest tests.test_gff_pipeline -v
if ($LASTEXITCODE -ne 0) { throw "GFF ablation tests failed" }

$experiments = @(
    @{ Config = "configs\gff_vit_ablation_standard.yaml"; Output = "outputs\gff_vit_ablation_standard" },
    @{ Config = "configs\gff_vit_ablation_db.yaml"; Output = "outputs\gff_vit_ablation_db" },
    @{ Config = "configs\gff_vit_ablation_clahe_only.yaml"; Output = "outputs\gff_vit_ablation_clahe_only" }
)

foreach ($experiment in $experiments) {
    if (-not $SkipTraining) {
        conda run -n daily --no-capture-output python -u -m src.train_gff `
            --config $experiment.Config
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $($experiment.Config)"
        }
    }

    $checkpoints = 1..4 | ForEach-Object {
        Join-Path $experiment.Output "epoch$_.pt"
    }
    conda run -n daily --no-capture-output python -u scripts\select_gff_checkpoint.py `
        --config $experiment.Config `
        --checkpoints $checkpoints
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint selection failed for $($experiment.Config)"
    }
}

conda run -n daily --no-capture-output python scripts\summarize_gff_sunet_ablation.py
if ($LASTEXITCODE -ne 0) { throw "Ablation summary failed" }

Write-Host "GFF SU-Net preprocessing ablation completed."
