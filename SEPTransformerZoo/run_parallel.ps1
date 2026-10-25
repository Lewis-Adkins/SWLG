# Creates any missing bootstrap datasets once, then opens one PowerShell window per dataset group,
# each training only its datasets (main.py train --datasets <group>). Each window logs to overnight_<group>.log.
# When every window has finished:  py -3.14 main.py analyze;  py -3.14 main.py plot
# Edit $groups to change how the work is split.
# Usage (from the project folder):  .\run_parallel.ps1

$groups = @("0-4", "5-9")

# Setup first, in this window, so the training windows never race to create the same dataset files
py -3.14 -u main.py setup

foreach ($g in $groups) {
    $log = "overnight_$g.log"
    $cmd = "Set-Location '$PSScriptRoot'; py -3.14 -u main.py train --datasets $g 2>&1 | Tee-Object $log"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd
}
