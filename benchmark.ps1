# Fisher-KPP Benchmark Pipeline for Windows
# PowerShell wrapper providing make equivalents for Windows users

param(
    [Parameter(Position=0)]
    [ValidateSet("benchmark-cpu", "benchmark", "benchmark-cluster", "results", "status", "docs", "clean", "help")]
    [string]$Target = "help"
)

function Show-Help {
    Write-Host "Fisher-KPP Benchmark Pipeline (Windows)"
    Write-Host ""
    Write-Host "Usage: ./benchmark.ps1 [target]"
    Write-Host ""
    Write-Host "Targets:"
    Write-Host "  benchmark-cpu       CPU benchmarks only (fast, local)"
    Write-Host "  benchmark           Full pipeline (CPU + cluster GPU)"
    Write-Host "  benchmark-cluster   Submit GPU jobs to Talon"
    Write-Host "  results             Aggregate latest results"
    Write-Host "  docs                Show documentation"
    Write-Host "  status              Check pipeline status"
    Write-Host "  clean               Remove output files"
    Write-Host "  help                Show this message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  ./benchmark.ps1 benchmark-cpu"
    Write-Host "  ./benchmark.ps1 benchmark"
    Write-Host ""
}

function Invoke-BenchmarkCPU {
    Write-Host "=========================================="
    Write-Host "CPU BENCHMARKS (via Talon Cluster)"
    Write-Host "=========================================="
    Write-Host ""
    
    python scripts/cluster_runner.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Benchmarking complete. Check outputs/ for results."
    } else {
        Write-Host ""
        Write-Host "SSH connection failed. To enable automated access:"
        Write-Host "  1. Read: ./SSH_SETUP.md"
        Write-Host "  2. Run: ssh-keygen -t rsa -b 4096 -N """" -f ~/.ssh/id_rsa"
        Write-Host "  3. Copy key: cat ~/.ssh/id_rsa.pub | ssh jayapreethi.mohan@talon.und.edu 'cat >> ~/.ssh/authorized_keys'"
        Write-Host ""
        Write-Host "Or use manual SSH commands - see WINDOWS_SETUP.md"
    }
}

function Invoke-Benchmark {
    Write-Host "=========================================="
    Write-Host "FULL PIPELINE: CPU + CLUSTER GPU"
    Write-Host "=========================================="
    Write-Host ""
    
    python scripts/run_benchmark_suite.py --config config/benchmark_config.yaml
}

function Invoke-BenchmarkCluster {
    Write-Host "=========================================="
    Write-Host "SUBMIT GPU JOBS TO TALON CLUSTER"
    Write-Host "=========================================="
    Write-Host ""
    
    python scripts/run_benchmark_suite.py --cluster-only --config config/benchmark_config.yaml
    
    Write-Host ""
    Write-Host "Monitor jobs: ssh jayapreethi.mohan@talon.und.edu squeue"
}

function Invoke-Results {
    Write-Host "=========================================="
    Write-Host "AGGREGATING LATEST RESULTS"
    Write-Host "=========================================="
    Write-Host ""
    
    $LatestRun = Get-ChildItem outputs/benchmark_*/ -Directory -ErrorAction SilentlyContinue | Sort-Object { $_.LastWriteTime } -Descending | Select-Object -First 1
    
    if ($LatestRun) {
        Write-Host "Processing: $($LatestRun.FullName)"
        python scripts/aggregate_results.py --run-dir $LatestRun.FullName
    }
    else {
        Write-Host "No benchmark results found in outputs/"
    }
}

function Show-Docs {
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host "PIPELINE DOCUMENTATION"
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Host ""
    Write-Host "Quick Start:"
    Write-Host "  docs/PIPELINE_GUIDE.md          Full technical reference"
    Write-Host "  PIPELINE_QUICKSTART.md           30-second tutorials"
    Write-Host ""
    Write-Host "Configuration:"
    Write-Host "  config/benchmark_config.yaml    All benchmark parameters"
    Write-Host ""
    Write-Host "Scripts:"
    Write-Host "  scripts/run_benchmark_suite.py    Master orchestrator"
    Write-Host "  scripts/gpu_benchmark_single.py  GPU benchmark (Talon)"
    Write-Host "  scripts/aggregate_results.py     Result aggregation"
    Write-Host ""
    Write-Host "Output:"
    Write-Host "  outputs/benchmark_*/            Timestamped results"
    Write-Host "  outputs/benchmark_*/BENCHMARK_REPORT.md  Main report"
    Write-Host ""
    Write-Host "View full guide:"
    Write-Host "  Get-Content docs/PIPELINE_GUIDE.md | more"
}

function Show-Status {
    Write-Host "Pipeline Status:"
    Write-Host ""
    
    $Runs = Get-ChildItem outputs/benchmark_*/ -Directory -ErrorAction SilentlyContinue | Sort-Object { $_.LastWriteTime } -Descending
    
    if ($Runs) {
        foreach ($Run in $Runs | Select-Object -First 3) {
            Write-Host "Run: $($Run.Name)"
            
            if (Test-Path "$($Run.FullName)/metadata.json") {
                Write-Host "  [OK] Environment captured"
            }
            if (Test-Path "$($Run.FullName)/cpu_results.json") {
                Write-Host "  [OK] CPU results available"
            }
            if (Test-Path "$($Run.FullName)/BENCHMARK_REPORT.md") {
                Write-Host "  [OK] Report generated"
            }
            Write-Host ""
        }
    }
    else {
        Write-Host "No benchmark results yet. Run: ./benchmark.ps1 benchmark-cpu"
    }
}

function Invoke-Clean {
    Write-Host "Cleaning benchmark outputs..."
    Remove-Item -Path "outputs/benchmark_*" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned"
}

# Route to appropriate function
switch ($Target) {
    "benchmark-cpu" { Invoke-BenchmarkCPU }
    "benchmark" { Invoke-Benchmark }
    "benchmark-cluster" { Invoke-BenchmarkCluster }
    "results" { Invoke-Results }
    "docs" { Show-Docs }
    "status" { Show-Status }
    "clean" { Invoke-Clean }
    "help" { Show-Help }
    default { Show-Help }
}
