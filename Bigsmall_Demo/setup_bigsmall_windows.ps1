$ErrorActionPreference = 'Stop'

$environmentName = 'bigsmall-rppg'
$environmentExists = conda env list | Select-String -SimpleMatch $environmentName

if (-not $environmentExists) {
    conda create -n $environmentName python=3.10 pip -y
}

conda run -n $environmentName python -m pip install `
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 `
    --index-url https://download.pytorch.org/whl/cu128
conda run -n $environmentName python -m pip install -r requirements-bigsmall-windows.txt

$faceModelDirectory = Join-Path $PSScriptRoot 'assets\models'
$faceModelPath = Join-Path $faceModelDirectory 'face_landmarker.task'
if (-not (Test-Path -LiteralPath $faceModelPath)) {
    New-Item -ItemType Directory -Force -Path $faceModelDirectory | Out-Null
    Invoke-WebRequest `
        -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' `
        -OutFile $faceModelPath
}

conda run -n $environmentName python tools/run_bigsmall_smoke.py
