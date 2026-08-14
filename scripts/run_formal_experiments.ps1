$ErrorActionPreference = "Stop"
$Python = "C:\Users\worker1h\.conda\envs\daily\python.exe"

foreach ($Experiment in 0..3) {
    & $Python -m src.train --config "configs\formal_e$Experiment.yaml"
    & $Python scripts\evaluate_checkpoint.py `
        --config "configs\formal_e$Experiment.yaml" `
        --split test `
        --sample-list "data\split\impactmesh_flood_test_holdout.txt" `
        --name test_holdout
}

& $Python scripts\formal_summary.py
& $Python -m unittest discover -v
