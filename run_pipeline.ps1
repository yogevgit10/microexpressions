param(
  [string]$VideoDir = "C:\Users\gitya\Downloads\Micro Exp Papers",
  [string]$OutDir = ".\results",
  [double]$EverySeconds = 5.0,
  [switch]$DeepFace,
  [switch]$ExtractAll
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

Push-Location $ScriptDir
try {
  if ($ExtractAll) {
    $ArgsList = @(
      "-m", "microexp.cli", "extract-all",
      "--video-dir", $VideoDir,
      "--out-dir", $OutDir,
      "--target-fps", "100"
    )
  }
  else {
    $ArgsList = @(
      "-m", "microexp.cli", "run-all",
      "--video-dir", $VideoDir,
      "--out-dir", $OutDir,
      "--every-s", "$EverySeconds"
    )
  }
  if ($DeepFace) {
    $ArgsList += "--deepface"
  }
  & $Python @ArgsList
}
finally {
  Pop-Location
}
