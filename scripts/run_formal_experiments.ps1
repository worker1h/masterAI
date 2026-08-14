$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH. Open an Anaconda/Miniconda PowerShell first."
}

function Invoke-DailyPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
    & conda run --no-capture-output -n daily python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "daily Python command failed with exit code $LASTEXITCODE"
    }
}

foreach ($Experiment in 0..3) {
    Invoke-DailyPython -m src.train --config "configs\formal_e$Experiment.yaml"
    Invoke-DailyPython scripts\evaluate_checkpoint.py `
        --config "configs\formal_e$Experiment.yaml" `
        --split test `
        --sample-list "data\split\impactmesh_flood_test_holdout.txt" `
        --name test_holdout
}

Invoke-DailyPython scripts\formal_summary.py
Invoke-DailyPython -m unittest discover -v
