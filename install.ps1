# Installs doekit for you, or updates it to the latest version. No administrator rights
# are needed. On Windows, paste this into Windows PowerShell:
#
#   irm https://raw.githubusercontent.com/hagneeptatt/doe/main/install.ps1 | iex
#
# Everything goes in one folder, doekit in your user folder: a private copy of conda
# (Miniforge), the packages doekit needs, and doekit itself. Nothing else on your
# computer is changed. Running it again updates doekit and keeps your saved studies.
# To remove doekit, delete that folder and the doekit shortcuts.
#
# Settings, mainly for testing: DOEKIT_HOME (where to install), DOEKIT_ARCHIVE (a .zip of
# the code to use instead of the latest release), DOEKIT_ICON_FOLDER (where to put the
# shortcut) and DOEKIT_NO_START (don't open doekit at the end).

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

function Install-Doekit {
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"  # downloads are far slower with a progress bar
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

    $repo = "hagneeptatt/doe"
    if ($env:DOEKIT_HOME) {
        $installDir = $env:DOEKIT_HOME
    } else {
        $installDir = Join-Path $env:USERPROFILE "doekit"
        if ($installDir.Contains(" ")) {
            # conda doesn't cope well with spaces in folder names
            $installDir = Join-Path $env:SystemDrive "doekit"
        }
    }
    $code = Join-Path $installDir "doe"
    $conda = Join-Path $installDir "miniforge"
    $envDir = Join-Path $installDir "env"
    $condaExe = Join-Path $conda "Scripts\conda.exe"
    $python = Join-Path $envDir "python.exe"

    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    $work = Join-Path ([IO.Path]::GetTempPath()) ("doekit-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $work | Out-Null
    Write-Step "Installing doekit in $installDir"

    try {
        # 1. A private copy of conda, so nothing else on this computer is affected.
        if (-not (Test-Path $condaExe)) {
            Write-Step "Downloading Miniforge (about 100 MB)"
            $installer = Join-Path $work "miniforge.exe"
            Invoke-WebRequest -UseBasicParsing -OutFile $installer `
                -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
            Write-Step "Installing Miniforge"
            $options = "/InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /NoShortcuts=1 /S /D=$conda"
            Start-Process -FilePath $installer -ArgumentList $options -Wait | Out-Null
            if (-not (Test-Path $condaExe)) { throw "Miniforge didn't install." }
        }

        # 2. The latest released version of doekit.
        Write-Step "Downloading doekit"
        $archive = $env:DOEKIT_ARCHIVE
        if (-not $archive) {
            $tag = (Invoke-RestMethod -UseBasicParsing -Uri "https://api.github.com/repos/$repo/releases/latest").tag_name
            if (-not $tag) { throw "Couldn't find the latest version of doekit." }
            $archive = "https://github.com/$repo/archive/refs/tags/$tag.zip"
        }
        $zip = Join-Path $work "doekit.zip"
        if (Test-Path $archive) {
            Copy-Item $archive $zip
        } else {
            Invoke-WebRequest -UseBasicParsing -Uri $archive -OutFile $zip
        }
        Expand-Archive -Path $zip -DestinationPath (Join-Path $work "unzipped")
        $new = (Get-ChildItem (Join-Path $work "unzipped") -Directory | Select-Object -First 1).FullName
        if (-not (Test-Path (Join-Path $new "app\streamlit_app.py"))) { throw "The download didn't contain doekit." }

        # 3. Close doekit if it's running, then swap in the new version. Saved studies and
        #    results come along, and the previous version is kept, just in case.
        if (Test-Path $code) {
            Write-Step "Updating your copy of doekit. Your saved studies are kept"
            if (Test-Path $python) {
                Push-Location $code
                & $python -m doekit.launcher --stop | Out-Null
                Pop-Location
            }
            foreach ($keep in "data", "results") {
                if (Test-Path (Join-Path $code $keep)) {
                    Remove-Item -Recurse -Force (Join-Path $new $keep) -ErrorAction SilentlyContinue
                    Copy-Item -Recurse (Join-Path $code $keep) (Join-Path $new $keep)
                }
            }
            $previous = Join-Path $installDir "doe-previous"
            if (Test-Path $previous) { Remove-Item -Recurse -Force $previous }
            Move-Item $code $previous
        }
        Move-Item $new $code

        # 4. The packages doekit needs, in an environment of their own.
        Write-Step "Installing the packages doekit needs. The first time, this takes about 5 to 10 minutes"
        Push-Location $code
        try {
            if (Test-Path $python) {
                & $condaExe env update -p $envDir -f environment.yml --prune
            } else {
                & $condaExe env create -p $envDir -f environment.yml
            }
            if ($LASTEXITCODE -ne 0) { throw "Couldn't install the packages doekit needs. Check your internet connection, then try again." }

            # 5. The shortcut, then start doekit.
            Write-Step "Making the doekit shortcut"
            if ($env:DOEKIT_ICON_FOLDER) {
                & $python scripts\make_shortcut.py --folder $env:DOEKIT_ICON_FOLDER
            } else {
                & $python scripts\make_shortcut.py
            }
            if ($LASTEXITCODE -ne 0) { throw "Couldn't make the shortcut. You can still start doekit with: $python -m doekit.launcher --root `"$code`"" }
        } finally {
            Pop-Location
        }

        Write-Step "doekit is installed"
        Write-Host "From now on, double-click the doekit shortcut on your Desktop, or find doekit in the Start menu."
        Write-Host "To update doekit later, run the same command again. Your saved studies are kept."
        if (-not $env:DOEKIT_NO_START) {
            Start-Process -FilePath (Join-Path $envDir "pythonw.exe") -WorkingDirectory $code `
                -ArgumentList @("-m", "doekit.launcher", "--root", "`"$code`"")
            Write-Host "doekit is opening in your browser now. You can close this window."
        }
    } catch {
        Write-Host ""
        Write-Host "Problem: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Nothing you'd saved in doekit has been lost. Try again, or ask for help with the message above."
        $global:DoekitInstallFailed = $true
    } finally {
        Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    }
}

# Everything is inside Install-Doekit, so nothing runs until the whole script has downloaded.
$global:DoekitInstallFailed = $false
Install-Doekit
