# Runs one window's queue of feature-DQN training jobs back to back, skipping any job that already finished.
param([Parameter(Mandatory = $true)][ValidateSet("A", "B")][string]$Window)

Set-Location (Join-Path $PSScriptRoot "..")
$python = ".venv\Scripts\python.exe"
$log = "runs\pilot_features\window_$Window.log"
New-Item -ItemType Directory -Force "runs\pilot_features" | Out-Null

$common = @(
    "--agent-type", "dqn", "--num-episodes", "10000", "--gamma", "0.95",
    "--learning-rate", "0.0005", "--target-sync-every-steps", "1000",
    "--epsilon-decay-episodes", "6000", "--checkpoint-every-episodes", "1000",
    "--eval-episodes", "100"
)
$both = @("--include-move-features", "--include-threat-features")

$queues = @{
    A = @(
        @{ seed = 1; name = "pilot_features"; flags = $both; out = "runs\pilot_features\treatment_move_threat_seed1" },
        @{ seed = 3; name = "pilot_features"; flags = $both; out = "runs\pilot_features\treatment_move_threat_seed3" },
        @{ seed = 0; name = "pilot_moves_only"; flags = @("--include-move-features"); out = "runs\pilot_features\moves_only_seed0" }
    )
    B = @(
        @{ seed = 2; name = "pilot_features"; flags = $both; out = "runs\pilot_features\treatment_move_threat_seed2" },
        @{ seed = 4; name = "pilot_features"; flags = $both; out = "runs\pilot_features\treatment_move_threat_seed4" },
        @{ seed = 0; name = "pilot_threats_only"; flags = @("--include-threat-features"); out = "runs\pilot_features\threats_only_seed0" }
    )
}

foreach ($job in $queues[$Window]) {
    $out = $job.out
    if (Test-Path "$out\checkpoints\episode_10000.pt") {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] skipping $out (already finished)"
        continue
    }
    $flags = $job.flags
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] starting $out"
    & $python -m training.quick_run @common --seed $job.seed --run-name $job.name @flags --output-dir $out *>> $log
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] finished $out (exit code $LASTEXITCODE)"
}
