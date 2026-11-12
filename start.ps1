# 一键运行：本地 Elasticsearch(免 Docker) + 闪购小智 python-langgraph
#
#   用法（在项目根目录执行）：
#     .\start.ps1            # 一键启动：ES(后台) → 等就绪 → 应用(后台)，全部后台化
#     .\start.ps1 es-only    # 只启动 ES
#     .\start.ps1 stop       # 停止：应用 + ES（数据保留）
#     .\start.ps1 status     # 查看 ES / 应用运行状态与日志路径
#
# 需要启动的本地服务只有 ES 与应用本身：
#   - Milvus 用 Lite 嵌入式库、Redis 用 memory 模式、重排/LLM 走远程 API，均无需单独启动
#
# 前置需求：
#   - 已 `pip install -r requirements.txt`
#   - ES 分发包自带捆绑 JDK，脚本会自动下载/解压（仅首次）

[CmdletBinding()]
param(
    [ValidateSet("up", "stop", "status", "es-only", "logs")]
    [string]$Action = "up"
)

$ErrorActionPreference = "Stop"
$Root      = $PSScriptRoot
$EsRoot    = Join-Path $env:LOCALAPPDATA "elasticsearch-8.11.3"
$EsHome    = Join-Path $EsRoot "elasticsearch-8.11.3"
$EsUrl     = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.3-windows-x86_64.zip"
$RunDir    = Join-Path $Root "logs"          # 统一的 PID / 日志目录（放项目目录下）
$EsPidFile = Join-Path $RunDir "es.pid"
$AppPidFile= Join-Path $RunDir "app.pid"
$AppLog    = Join-Path $RunDir "app.log"
$AppErrLog = Join-Path $RunDir "app.err.log"

function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }

# ---------- 安装（仅首次） ----------
function Install-Es {
    if (Test-Path (Join-Path $EsHome "bin\elasticsearch.bat")) { return }
    Write-Step "首次运行：下载并解压 Elasticsearch 8.11.3（约600MB，请稍候）"
    New-Item -ItemType Directory -Force -Path $EsRoot | Out-Null
    $zip = Join-Path $EsRoot "elasticsearch.zip"
    if (-not (Test-Path $zip) -or (Get-Item $zip).Length -lt 1MB) {
        curl.exe -L --fail -o $zip $EsUrl
    }
    Write-Step "解压中..."
    Expand-Archive -Path $zip -DestinationPath $EsRoot -Force
    $cfg = Join-Path $EsHome "config\elasticsearch.yml"
    Add-Content -Path $cfg -Value @"

# ---- local dev (added by start.ps1) ----
discovery.type: single-node
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
network.host: 127.0.0.1
http.port: 9200
"@ -Encoding UTF8
    New-Item -ItemType Directory -Force -Path (Join-Path $EsHome "config\jvm.options.d") | Out-Null
    # 注意：jvm.options.d 不能带 UTF-8 BOM，否则 ES 解析崩溃，故用 ASCII 无 BOM 写入
    [System.IO.File]::WriteAllText((Join-Path $EsHome "config\jvm.options.d\heap-size.options"), "-Xms512m`r`n-Xmx512m`r`n", (New-Object System.Text.ASCIIEncoding))
    Write-Step "ES 安装完成: $EsHome"
}

# ---------- ES ----------
function Start-Es {
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    if (Test-Path $EsPidFile) {
        $oldPid = Get-Content $EsPidFile
        if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
            Write-Host "[ok] ES 已在运行 (PID $oldPid)" -ForegroundColor Green; return
        }
    }
    # 清除系统 JAVA_HOME/CLASSPATH（CLASSPATH 常被 JDK 安装写入指向已移除的 dt.jar，会令 ES JarHell 崩溃）
    Remove-Item Env:JAVA_HOME,Env:JAVA_HOME_LOCAL,Env:JDK_HOME,Env:CLASSPATH -ErrorAction SilentlyContinue
    [Environment]::SetEnvironmentVariable("ES_JAVA_OPTS", "-Xms512m -Xmx512m", "Process")
    $esBat = Join-Path $EsHome "bin\elasticsearch.bat"
    $p = Start-Process -FilePath $esBat -WorkingDirectory $EsHome -WindowStyle Hidden -PassThru
    Set-Content -Path $EsPidFile -Value $p.Id -Encoding UTF8
    Write-Host "[ok] ES 启动中 (PID $($p.Id))" -ForegroundColor Green
}

function Wait-EsReady {
    Write-Host "等待 Elasticsearch 就绪 (http://localhost:9200)..." -ForegroundColor Cyan
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $resp = Invoke-RestMethod -Uri "http://localhost:9200/" -TimeoutSec 3
            if ($resp.version.number) { Write-Host "[ok] ES 就绪 (version $($resp.version.number))" -ForegroundColor Green; return }
        } catch { }
        Start-Sleep -Seconds 2
    }
    Write-Host "[!] ES 120s 未就绪，请查看 $EsHome\logs 排查。" -ForegroundColor Red
    exit 1
}

# ---------- 残留清理 ----------
function Cleanup-ResidualApp {
    # 启动前清理非当前脚本管理的、仍在运行的 run.py 进程。
    # Milvus Lite 的 data/milvus_lite.db 同一时刻只能被一个 python 进程独占，
    # 若上一次未干净退出（如非脚本方式启动/手动杀掉主进程），残留进程仍持有该文件锁，
    # 会导致本次启动时 RAG 报 DataDirLockedError。这里先清理掉这些壳子进程。
    $managed = @()
    foreach ($f in @($AppPidFile, $EsPidFile)) {
        if (Test-Path $f) { $managed += @(Get-Content $f -ErrorAction SilentlyContinue) }
    }
    $ghosts = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'run\.py' -and $managed -notcontains [string]$_.ProcessId }
    foreach ($g in $ghosts) {
        Stop-Process -Id $g.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "[ok] 已清理残留应用进程 (PID $($g.ProcessId))" -ForegroundColor Yellow
    }
}

# ---------- 应用 ----------
function Start-App {
    if (Test-Path $AppPidFile) {
        $oldPid = Get-Content $AppPidFile
        if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
            Write-Host "[ok] 应用已在运行 (PID $oldPid)" -ForegroundColor Green; return
        }
    }
    Write-Host "[i] 启动应用后台进程..." -ForegroundColor Cyan
    $p = Start-Process -FilePath "python" -ArgumentList "run.py" -WorkingDirectory $Root `
        -WindowStyle Hidden -RedirectStandardOutput $AppLog -RedirectStandardError $AppErrLog -PassThru
    Set-Content -Path $AppPidFile -Value $p.Id -Encoding UTF8
    Write-Host "[ok] 应用启动中 (PID $($p.Id))" -ForegroundColor Green
}

function Wait-AppReady {
    Write-Host "等待应用就绪 (http://localhost:8000/xiaozhi/health)..." -ForegroundColor Cyan
    for ($i = 0; $i -lt 45; $i++) {
        try {
            $h = Invoke-RestMethod -Uri "http://localhost:8000/xiaozhi/health" -TimeoutSec 4
            if ($h.status -eq "ok") { Write-Host "[ok] 应用就绪 (skill=$($h.skills -join ','))" -ForegroundColor Green; return }
        } catch { }
        Start-Sleep -Seconds 2
    }
    Write-Host "[!] 应用 90s 未就绪，请查看日志：$AppLog / $AppErrLog" -ForegroundColor Red
    exit 1
}

# ---------- 停止 ----------
function Stop-ProcessByPidFile($pidFile, $label, $port) {
    if (Test-Path $pidFile) {
        $procId = Get-Content $pidFile
        if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            Write-Host "[ok] 已停止 $label (PID $procId)" -ForegroundColor Green
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    # 兜底：按端口杀
    $onPort = (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
    if ($onPort) {
        Stop-Process -Id $onPort -Force -ErrorAction SilentlyContinue
        Write-Host "[ok] 已停止 $label (按端口 $port, PID $onPort)" -ForegroundColor Green
    }
}

function Stop-All {
    Stop-ProcessByPidFile $AppPidFile "应用" 8000
    Stop-ProcessByPidFile $EsPidFile "ES" 9200
}

# ---------- 状态 ----------
function Show-Status {
    Write-Host "`n=== 运行状态 ===" -ForegroundColor Cyan
    foreach ($item in @(
        @{Name="ES(9200)";   Port=9200; PidFile=$EsPidFile},
        @{Name="应用(8000)"; Port=8000; PidFile=$AppPidFile}
    )) {
        $procId = (Get-NetTCPConnection -LocalPort $item.Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess)
        if ($procId) { Write-Host "  $($item.Name): 运行中 (PID $procId)" -ForegroundColor Green }
        else      { Write-Host "  $($item.Name): 未运行" -ForegroundColor Gray }
    }
    Write-Host "`n日志:"
    Write-Host "  ES:    $EsHome\logs"
    Write-Host "  应用:  $AppLog"
}

# ---------- 执行 ----------
Set-Location $Root
Install-Es

switch ($Action) {
    "up" {
        Cleanup-ResidualApp
        Start-Es
        Wait-EsReady
        Start-App
        Wait-AppReady
        Write-Host "`n✅ 全部启动完成" -ForegroundColor Green
        Write-Host "   聊天页:  http://localhost:8000"
        Write-Host "   健康检查: http://localhost:8000/xiaozhi/health"
        Write-Host "   停止:     .\start.ps1 stop"
    }
    "es-only" {
        Start-Es
        Wait-EsReady
        Write-Host "[ok] ES 运行于 http://localhost:9200 ；跑应用请执行 .\start.ps1" -ForegroundColor Green
    }
    "stop" {
        Stop-All
    }
    "status" {
        Show-Status
    }
    "logs" {
        if (-not (Test-Path $AppLog)) {
            Write-Host "[!] 应用日志不存在（应用可能未启动）：$AppLog" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "实时跟踪应用日志（Ctrl+C 退出）：$($AppLog)" -ForegroundColor Green
        Write-Host "查看 ES 日志目录：$EsHome\logs`n" -ForegroundColor Green
        Get-Content -Path $AppLog -Tail 50 -Wait
    }
}