# RTAI Trading System - Production Deployment Script (Windows PowerShell)
# One-command deployment with health checks and rollback capability

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "rollback", "status", "logs", "stop", "restart")]
    [string]$Action = "deploy"
)

# Configuration
$ProjectName = "rtai-trading"
$TargetImageSizeMB = 190
$HealthTimeout = 60
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Color functions
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Blue }
function Write-Success { param($Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

# Pre-deployment checks
function Test-Prerequisites {
    Write-Info "🔍 Checking prerequisites..."
    
    # Check Docker
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed"
        exit 1
    }
    
    # Check Docker Compose
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Compose is not installed" 
        exit 1
    }
    
    # Check .env file
    $EnvFile = Join-Path $ScriptDir ".env"
    if (-not (Test-Path $EnvFile)) {
        Write-Warning ".env file not found"
        Write-Info "Creating .env from .env.example..."
        $EnvExample = Join-Path $ScriptDir ".env.example"
        Copy-Item $EnvExample $EnvFile
        Write-Warning "Please edit .env file with your configuration"
        exit 1
    }
    
    # Verify no matplotlib imports
    Write-Info "🧟 Checking for zombie files..."
    $RtaiDir = Join-Path $ScriptDir "rtai"
    $ZombieCheck = Select-String -Path "$RtaiDir\*.py" -Pattern "import matplotlib|from matplotlib" -Quiet
    if ($ZombieCheck) {
        Write-Error "❌ ZOMBIE DETECTED: matplotlib imports found in rtai/"
        exit 1
    }
    
    Write-Success "✅ Prerequisites check passed"
}

# Build optimized Docker image
function Build-Image {
    Write-Info "🔨 Building optimized Docker image..."
    
    # Build with target runtime
    $BuildArgs = @(
        "build",
        "--target", "runtime",
        "--tag", "${ProjectName}:latest",
        "--tag", "${ProjectName}:$(Get-Date -Format 'yyyyMMdd-HHmmss')",
        $ScriptDir
    )
    
    & docker @BuildArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed"
        exit 1
    }
    
    # Check image size
    $ImageInfo = docker images "${ProjectName}:latest" --format "table {{.Size}}" | Select-Object -Skip 1 -First 1
    Write-Info "📦 Image size: $ImageInfo"
    
    # Basic size validation (if it contains GB, it's too big)
    if ($ImageInfo -like "*GB*") {
        Write-Error "❌ Image size exceeds target (<${TargetImageSizeMB}MB)"
        exit 1
    }
    
    Write-Success "✅ Docker image built successfully"
}

# Deploy services
function Deploy-Services {
    Write-Info "🚀 Deploying services..."
    
    # Change to script directory
    Push-Location $ScriptDir
    
    try {
        # Stop existing containers
        docker-compose down --remove-orphans 2>$null
        
        # Start services
        docker-compose up -d
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to start services"
            exit 1
        }
        
        Write-Success "✅ Services deployed"
    }
    finally {
        Pop-Location
    }
}

# Health check
function Test-Health {
    Write-Info "🏥 Waiting for health check..."
    
    $Attempts = 0
    $MaxAttempts = [math]::Floor($HealthTimeout / 5)
    
    while ($Attempts -lt $MaxAttempts) {
        try {
            $Response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5 -UseBasicParsing
            if ($Response.StatusCode -eq 200) {
                Write-Success "✅ Health check passed"
                return $true
            }
        }
        catch {
            # Continue trying
        }
        
        Write-Info "⏳ Waiting for service to be healthy... (attempt $($Attempts + 1)/$MaxAttempts)"
        Start-Sleep 5
        $Attempts++
    }
    
    Write-Error "❌ Health check failed after ${HealthTimeout}s"
    return $false
}

# Show service status
function Show-Status {
    Write-Info "📊 Service Status:"
    Push-Location $ScriptDir
    
    try {
        docker-compose ps
        
        Write-Host ""
        Write-Info "🏥 Health Status:"
        try {
            $HealthResponse = Invoke-WebRequest -Uri "http://localhost:8080/health/status" -UseBasicParsing
            $HealthResponse.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
        }
        catch {
            Write-Host "Health endpoint not available"
        }
        
        Write-Host ""
        Write-Info "📈 Available Endpoints:"
        Write-Host "  Health: http://localhost:8080/health"
        Write-Host "  Status: http://localhost:8080/health/status"
        Write-Host "  Metrics: http://localhost:8080/metrics"
        
        Write-Host ""
        Write-Info "📋 Logs:"
        Write-Host "  View logs: docker-compose logs -f rtai"
        Write-Host "  View metrics: Invoke-WebRequest http://localhost:8080/metrics"
    }
    finally {
        Pop-Location
    }
}

# Rollback function
function Invoke-Rollback {
    Write-Warning "🔄 Rolling back deployment..."
    
    Push-Location $ScriptDir
    
    try {
        # Stop current deployment
        docker-compose down
        
        # Find previous image
        $PreviousImages = docker images $ProjectName --format "table {{.Tag}}" | 
                         Select-Object -Skip 1 | 
                         Where-Object { $_ -match '\d{8}-\d{6}' } | 
                         Select-Object -First 1
        
        if ($PreviousImages) {
            Write-Info "Rolling back to ${ProjectName}:$PreviousImages"
            docker tag "${ProjectName}:$PreviousImages" "${ProjectName}:latest"
            docker-compose up -d
            Write-Success "✅ Rollback completed"
        }
        else {
            Write-Error "❌ No previous image found for rollback"
            exit 1
        }
    }
    finally {
        Pop-Location
    }
}

# Main deployment logic
function Main {
    Write-Info "🎯 Starting RTAI Trading System deployment..."
    
    switch ($Action) {
        "deploy" {
            Test-Prerequisites
            Build-Image
            Deploy-Services
            if (Test-Health) {
                Show-Status
                Write-Success "🎉 Deployment completed successfully!"
            }
            else {
                Write-Error "💥 Deployment failed health check"
                Write-Info "Use '.\deploy.ps1 rollback' to rollback"
                exit 1
            }
        }
        "rollback" {
            Invoke-Rollback
        }
        "status" {
            Show-Status
        }
        "logs" {
            Push-Location $ScriptDir
            try { docker-compose logs -f rtai }
            finally { Pop-Location }
        }
        "stop" {
            Write-Info "🛑 Stopping services..."
            Push-Location $ScriptDir
            try {
                docker-compose down
                Write-Success "✅ Services stopped"
            }
            finally { Pop-Location }
        }
        "restart" {
            Write-Info "🔄 Restarting services..."
            Push-Location $ScriptDir
            try {
                docker-compose restart
                if (Test-Health) { Show-Status }
            }
            finally { Pop-Location }
        }
    }
}

# Run main function
try {
    Main
}
catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    exit 1
}
