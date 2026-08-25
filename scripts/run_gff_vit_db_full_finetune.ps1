$ErrorActionPreference = "Stop"

$config = "configs\gff_vit_db_full_finetune.yaml"
$initialCheckpoint = "outputs\gff_vit_ablation_db\best.pt"
$outputDir = "outputs\gff_vit_db_full_finetune"

if (-not (Test-Path -LiteralPath $initialCheckpoint)) {
    throw "Missing selected dB initialization checkpoint: $initialCheckpoint"
}

conda run -n daily --no-capture-output python -m unittest tests.test_gff_pipeline -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

conda run -n daily --no-capture-output python -m src.train_gff `
    --config $config `
    --init-checkpoint $initialCheckpoint
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

conda run -n daily --no-capture-output python scripts\select_gff_checkpoint.py `
    --config $config `
    --checkpoints "$outputDir\epoch1.pt" "$outputDir\epoch2.pt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

conda run -n daily --no-capture-output python scripts\predict_gff_example.py `
    --config $config `
    --checkpoint "$outputDir\best.pt" `
    --horizon 3 `
    --output "$outputDir\prediction_72h.png"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Full-data dB-only ViT fine-tuning completed."
