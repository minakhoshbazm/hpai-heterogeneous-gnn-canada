# Remote Run Workflow: Chat → Compute Canada → Local Results

## Goal

The intended workflow is:

```text
You describe the run in this chat
        ↓
Codex prepares code, a Slurm job, and a run record locally
        ↓
The project is staged to Fir
        ↓
Slurm runs it on a compute node
        ↓
Outputs are retrieved to this Paper 5 folder
        ↓
Codex validates, post-processes, plots, and reports the result here
```

This keeps source code, scripts, logs, results, and analysis organized rather than creating ad-hoc folders.

## Directory convention

Local project root:

```text
G:\Other computers\My Computer\GoogleDrive\paper 4\paper 5
```

Remote project root:

```text
~/projects/paper5
```

Recommended structure on both sides:

```text
paper5/
  src/                 # model and analysis code
  jobs/                # Slurm submission scripts
  runs/<run-id>/       # immutable run inputs, Slurm log, and raw outputs
  results/<run-id>/    # retrieved and post-processed outputs
  operations/          # setup record and this workflow
```

Existing Paper 5 files are not moved automatically. Each new remote run will use this layout going forward.

## Mandatory session logging rule

After every material action—local code change, package build, upload, remote
environment check, Slurm submission, status check, download, verification, or
post-processing—the active session must update:

```text
operations/SESSION_STATE.md
```

That file is the compact handoff state for the next AI session. The detailed
history remains in the setup log and the run-specific record. A new session
should begin by reading `SESSION_STATE.md`, then the linked record relevant to
the requested task.

## What to tell Codex in chat

Use a request like:

```text
Run the Paper 5 controller-sensitivity experiment on Compute Canada:
use seeds 1–100, 8 CPUs, 8 GB per CPU, 4-hour limit.
Save the raw output as run-2026-08-09-controller-sensitivity,
then download results and make the figures and a summary table locally.
```

I will then:

1. check and update the local code as needed;
2. create one named run directory and a `RUN_RECORD.md` describing parameters, code version, and expected outputs;
3. create a Slurm `sbatch` script with the requested CPU, memory, and time limits;
4. stage only the needed files to the remote project;
5. submit and monitor the job;
6. retrieve the run outputs to the matching local `results/<run-id>/` directory;
7. verify expected files and post-process them locally.

I will report the Slurm job ID, status, remote path, local result path, and any failure clearly.

## Authentication findings

Fir accepts the registered public key but requires Duo through
`keyboard-interactive` authentication. The Windows SSH agent is enabled and the
encrypted private key has been loaded with `ssh-add`. A batch-only test therefore
returns:

```text
Permission denied (keyboard-interactive).
```

Codex does not bypass or silently complete Duo. Instead, the working method
keeps the user in the authentication loop while automating everything after it.

## Working graphical Duo automation

Git for Windows provides the graphical helper:

```text
C:\Program Files\Git\mingw64\bin\git-askpass.exe
```

For an authorized SSH or SCP action, Codex sets these variables only for the
current process:

```powershell
$env:SSH_ASKPASS='C:\Program Files\Git\mingw64\bin\git-askpass.exe'
$env:SSH_ASKPASS_REQUIRE='force'
$env:DISPLAY='codex-duo'
```

Codex then invokes Windows OpenSSH. A small graphical challenge appears on the
user's desktop. The user enters Duo option `1` and approves the phone push; the
command then continues automatically. This was validated with a read-only
`hostname` call that returned `login2`, and subsequently used for SCP transfers,
environment checks, module discovery, and Slurm submission.

Operational rules:

- The user handles only the graphical Duo challenge and phone approval.
- Never enter an Alliance password, SSH-key passphrase, or reusable credential
  into chat or an automation file.
- Each new SSH or SCP process may require a new Duo approval.
- Codex reports the intended action before displaying the authentication prompt.
- This is user-approved MFA, not unattended authentication or an MFA bypass.

## Useful Windows commands

Run these in Windows PowerShell, not in the Antigravity `>` prompt.

```powershell
# Open a remote shell
ssh fir

# Upload the Paper 5 project to its clean remote location
scp -r ".\paper 5" fir:"~/projects/"

# Download one completed result directory
scp -r fir:"~/projects/paper5/results/<run-id>" ".\paper 5\results\"

# Test whether automation can log in without a password/Duo prompt
ssh -o BatchMode=yes -o PreferredAuthentications=publickey fir "hostname"
```

Replace `<run-id>` with the actual run name. For large or repeated syncs, `rsync` is preferable if installed, but `scp` is sufficient to start.

## Useful commands on Fir

Run these at the normal remote shell prompt, where it looks like `[taba9481@login3 ~]$`.

```bash
# Make the remote project root once
mkdir -p ~/projects/paper5

# Go to it
cd ~/projects/paper5

# Submit a prepared Slurm job
sbatch jobs/<job-file>.sbatch

# Show your queued/running jobs
squeue -u "$USER"

# Inspect an accounting record after the job finishes
sacct -j <job-id> --format=JobID,JobName,State,Elapsed,AllocCPUS,MaxRSS

# Cancel a mistaken job
scancel <job-id>
```

## Slurm job-script baseline

Every actual computation should be submitted by a file in `jobs/`, similar to:

```bash
#!/bin/bash
#SBATCH --job-name=paper5-example
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=../runs/run-id/slurm-%j.out
#SBATCH --error=../runs/run-id/slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load the required modules/environment here.
# Run the project command here.
```

The resource values and environment will be tailored to each run; do not blindly submit this template as-is.

## Keep Antigravity available

At the remote shell:

```bash
tmux new -s agy
agy
```

Detach with keyboard sequence `Ctrl+b`, then `d`. Reattach later with:

```bash
tmux attach -t agy
```

Antigravity is helpful for interacting with the remote project, while Codex in this chat remains responsible for the local project changes, results organization, and post-processing.

## Confirmed remote sessions

Two named sessions have been created and verified:

```text
agy     # Antigravity CLI
paper5  # normal shell in ~/projects/paper5
```

Use `tmux attach -t agy` or `tmux attach -t paper5` after an approved SSH login.

## First authorized Compute Canada run

The first remote job is Diagnostic 1A, `diag_plant_response`. Its fresh
upload archive is:

```text
bundles/paper5_diagnostic1a_20260809.tar.gz
```

The exact upload, environment-check, submission, monitoring, and retrieval
commands are in `operations/compute_canada/diagnostic1a/server/README.md` and
inside the archive at `server/README.md`. The job uses the confirmed CPU
allocation `def-aliabadi_cpu`; it does not launch production or other
diagnostic blocks.

The experimental helper `operations/run_diagnostic1a_remote.ps1` is retained as
a record but is not the active method. Direct execution was blocked by the
Windows PowerShell execution policy, and a helper launched from the background
could not expose its prompt on the visible desktop. The working method is the
graphical askpass flow documented above.

### Windows connection-sharing result

On 2026-08-09, Windows OpenSSH 9.5 returned `getsockname failed: Not a
socket` when a `ControlMaster`/`ControlPath` session was attempted. No usable
shared control socket was created. Do not rely on that method on this Windows
installation. The graphical askpass flow is the current local automation
method.

### Diagnostic 1A automation status

- Environment check: passed
- Runtime: `StdEnv/2023`, `python/3.10.13`, `scipy-stack/2023b`
- Initial main/collector: `53884149` / `53884153` (initial dependency failure)
- Corrected main/collector: `53906482` / `53906483`
- Corrected states: both `COMPLETED`, exit code 0
- Delivery directory:
  `~/projects/paper5/results/diag-plant-response-20260809/`

The collector automatically places simulation outputs, Slurm logs, accounting
data, a checksum, and `READY.txt` in the delivery directory. Once ready, Codex
will use the same graphical Duo method to download and post-process the package
locally.

### Phone-only SSH option

If the user wants both the SSH challenge and Duo approval on the phone, install
an SSH client such as Termius, create a host with hostname
`fir.alliancecan.ca`, username `taba9481`, port `22`, and import the existing
private key only through the app's encrypted key store. Connect, enter Duo
option `1` in the phone terminal, and approve the Duo push on that same phone.
Never send the private key or its passphrase through chat. This phone session
is independent of Codex's automated Windows session.

For a true no-keystroke Duo flow, ask Alliance support whether automatic Duo
push can be enabled or made the default for the account. The SSH client still
needs to receive the challenge response unless the account-side Duo policy
removes the method-selection prompt; Codex cannot change that account policy.
