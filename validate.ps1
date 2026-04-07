param (
    [Parameter(Mandatory=$true)][string]$PingUrl,
    [string]$RepoDir = "."
)

$ErrorActionPreference = "Continue"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  OpenEnv Submission Validator (Windows)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Repo: $RepoDir"
Write-Host "Ping URL: $PingUrl`n"

# Step 1: Ping HF Space
Write-Host "Step 1/3: Pinging HF Space ($PingUrl/reset) ..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "$PingUrl/reset" -Method Post -ContentType "application/json" -Body "{}" -UseBasicParsing -TimeoutSec 30
    if ($response.StatusCode -eq 200) {
        Write-Host "PASSED -- HF Space is live and responds to /reset`n" -ForegroundColor Green
    } else {
        Write-Host "FAILED -- HF Space /reset returned HTTP $($response.StatusCode) (expected 200)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "FAILED -- HF Space not reachable (connection failed or timed out)" -ForegroundColor Red
    Write-Host "Hint: Try starting your local FastAPI server in another terminal window.`n" -ForegroundColor Yellow
    exit 1
}

# Step 2: Docker build
Write-Host "Step 2/3: Running docker build ..." -ForegroundColor Cyan
if (!(Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "FAILED -- docker command not found" -ForegroundColor Red
    Write-Host "Hint: Ensure Docker Desktop is installed and running.`n" -ForegroundColor Yellow
    exit 1
}

$dockerContext = $RepoDir
if (Test-Path "$RepoDir\server\Dockerfile") {
    $dockerContext = "$RepoDir\server"
} elseif (!(Test-Path "$RepoDir\Dockerfile")) {
    Write-Host "FAILED -- No Dockerfile found in repo root." -ForegroundColor Red
    exit 1
}

Write-Host "  Found Dockerfile in $dockerContext"
$process = Start-Process -FilePath "docker" -ArgumentList "build $dockerContext" -Wait -NoNewWindow -PassThru
if ($process.ExitCode -eq 0) {
    Write-Host "PASSED -- Docker build succeeded`n" -ForegroundColor Green
} else {
    Write-Host "FAILED -- Docker build failed." -ForegroundColor Red
    exit 1
}

# Step 3: OpenEnv validate
Write-Host "Step 3/3: Running openenv validate ..." -ForegroundColor Cyan
if (!(Get-Command "openenv" -ErrorAction SilentlyContinue)) {
    Write-Host "FAILED -- openenv command not found" -ForegroundColor Red
    Write-Host "Hint: Install it via: pip install openenv-core`n" -ForegroundColor Yellow
    exit 1
}

# Temporarily change directory to the repo root to run validation
Push-Location $RepoDir
$process = Start-Process -FilePath "openenv" -ArgumentList "validate" -Wait -NoNewWindow -PassThru
Pop-Location

if ($process.ExitCode -eq 0) {
    Write-Host "`nPASSED -- openenv validate passed" -ForegroundColor Green
} else {
    Write-Host "`nFAILED -- openenv validate failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  All 3/3 checks passed! Ready to submit." -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan