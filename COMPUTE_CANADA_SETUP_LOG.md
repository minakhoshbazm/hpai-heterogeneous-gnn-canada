# Compute Canada and Antigravity Setup Log

This is the record of the remote-computing setup completed on 2026-08-09.

The compact resume file is `operations/SESSION_STATE.md`. It must be updated
after every material local or remote action so a later AI session can resume
without reconstructing the conversation.

## Accounts and connection details

- Alliance username: `taba9481`
- Login host: `fir.alliancecan.ca`
- SSH port: `22`
- Slurm CPU allocation: `def-aliabadi_cpu` (confirmed with `sacctmgr`; a GPU allocation also exists but is not used for Diagnostic 1A)
- Local SSH key pair:
  - private key: `C:\Users\Moji\.ssh\id_ed25519_alliance` (**private; never share or upload it**)
  - public key: `C:\Users\Moji\.ssh\id_ed25519_alliance.pub`
- The public-key line was registered in the Alliance CCDB security portal.
- Login was confirmed with password and Duo authentication.

## Steps completed

1. Generated an ED25519 SSH key on the Windows computer:

   ```powershell
   ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519_alliance"
   ```

2. Displayed the public key, then copied its complete one-line value:

   ```powershell
   Get-Content "$env:USERPROFILE\.ssh\id_ed25519_alliance.pub"
   ```

3. Added that public key in the Alliance account portal under SSH keys. The private-key file was not shared.

4. Confirmed the allocated login host and username from the WinSCP connection details:

   - host: `fir.alliancecan.ca`
   - user: `taba9481`

5. Connected successfully from Windows using SSH and completed password + Duo authentication:

   ```powershell
   ssh fir
   ```

   If the short name `fir` ever stops working, use the explicit form:

   ```powershell
   ssh -i "$env:USERPROFILE\.ssh\id_ed25519_alliance" taba9481@fir.alliancecan.ca
   ```

6. Installed Antigravity on the remote Linux account (not on Windows):

   ```bash
   curl -fsSL https://antigravity.google/cli/install.sh | bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

   The installer reported that `agy` was installed at:

   ```text
   /home/taba9481/.local/bin/agy
   ```

   The `logging before google.Init` lines appeared during installation but the final installation status was successful.

7. Started Antigravity remotely:

   ```bash
   agy
   ```

   It authenticated to the Google account and is available inside the SSH terminal.

8. Diagnosed key-only SSH. Fir confirmed it accepts the public key with fingerprint `SHA256:bVrV6H2quS3Zd8ld3WAIYuQ23btEwvCZAHWzesaOciE`. The initial non-interactive test failed because the Windows SSH agent was not running and `BatchMode=yes` prevented a prompt for the encrypted private key's passphrase. The corrective action is to start `ssh-agent` and add the key with `ssh-add`.

9. Started the Windows SSH agent and loaded the key. `ssh-add -l` confirmed the expected ED25519 fingerprint. A second key-only test then failed with `Permission denied (keyboard-interactive)`, which confirms that Fir accepts the key but requires an interactive second factor (Duo) for new connections.

10. Created and verified two persistent tmux sessions on Fir:

    ```text
    agy:    Antigravity CLI session
    paper5: normal Paper 5 project shell
    ```

    The `paper5` session created the clean remote directory layout:

    ```text
    ~/projects/paper5/{src,jobs,runs,results}
    ```

11. Prepared the first fresh Compute Canada archive for the current diagnostic code:

    ```text
    bundles/paper5_diagnostic1a_20260809.tar.gz
    ```

    It contains only the current code, runtime resources, vendored dependencies,
    diagnostics, and `server/` submission files. Existing `HVAC CODE/Results`
    folders are deliberately excluded. The job uses `def-aliabadi_cpu` and runs
    only `diag_plant_response` (Diagnostic 1A, nine deterministic cells).

12. Fir's current module catalogue was checked after the first environment
    probe. The package defaults were updated from unavailable `python/3.10` to
    the compatible set `StdEnv/2023`, `python/3.10.13`, and
    `scipy-stack/2023b`.

13. Diagnostic 1A passed its remote environment/dry-plan check and was
    submitted as Slurm job `53884149` under `def-aliabadi_cpu`. The verified
    matrix contains exactly nine pending deterministic plant-response cells.

14. Tested Windows OpenSSH connection sharing. `ControlMaster`/`ControlPath`
    failed with `getsockname failed: Not a socket`, so this Windows installation
    cannot use that approach. A visible PowerShell helper was also unsuitable:
    the background process could not expose its terminal prompt, and direct
    script execution was blocked by the local PowerShell execution policy.

15. Established the working user-in-the-loop automation method with Git's
    graphical askpass helper:

    ```text
    C:\Program Files\Git\mingw64\bin\git-askpass.exe
    ```

    Codex launches Windows OpenSSH with `SSH_ASKPASS_REQUIRE=force` and a dummy
    `DISPLAY` value. A small graphical challenge appears on the user's desktop;
    the user enters option `1` and approves the Duo push on the phone. The
    authenticated command then continues under Codex control. A read-only test
    succeeded and returned hostname `login2`.

16. Used the graphical Duo method to verify and manage the remote package.
    Both remote archive copies matched the local size exactly:

    ```text
    59,030,172 bytes
    ```

    The authoritative copy is
    `~/projects/paper5/src/paper5_diagnostic1a_20260809.tar.gz`. It was unpacked
    under `~/projects/paper5/runs/diag-plant-response-20260809/`.

17. Submitted the main Diagnostic 1A job as `53884149`. Its first observed
    queue state was `PENDING (Priority)`, which is a normal scheduling state.
    Submitted collector job `53884153` with dependency
    `afterany:53884149`. It will package results, Slurm logs, accounting data,
    a SHA-256 checksum, and `READY.txt` under
    `~/projects/paper5/results/diag-plant-response-20260809/`.

## Important distinctions

- `agy` is currently installed on **Fir**, so `agy` in Windows PowerShell is expected to say it is not recognized.
- `login3` is a login node. It is appropriate for editing, file transfer, preparing jobs, and submitting jobs; long or resource-heavy calculations must run through Slurm on compute nodes.
- A connection-reset message after starting a terminal application means the SSH session ended; it does not mean Antigravity was uninstalled. Using `tmux` prevents work from disappearing when a connection drops.
- `tmux` sessions are local to a specific login node. A session created on
  `login1` is not visible after a later connection lands on `login2`, although
  the home/project files are shared.
- Automated SSH/SCP now works through graphical Duo approval. Each new
  connection can create a small prompt; select option `1` and approve the phone
  push. No password, passphrase, or Duo code is sent to Codex.

## Persistent Antigravity session

On Fir:

```bash
tmux new -s agy
agy
```

To detach while leaving it running, press `Ctrl+b`, release, then press `d`.

To return later:

```powershell
ssh fir
```

```bash
tmux attach -t agy
```

## Persistent Paper 5 project session

The dedicated project shell is separate from Antigravity:

```bash
tmux attach -t paper5
cd ~/projects/paper5
```

Check both sessions with:

```bash
tmux list-sessions
```

Both sessions survive an SSH disconnect, but a new SSH connection still needs
the normal Duo approval.

## Security reminders

- Never paste, email, upload, or commit `id_ed25519_alliance`.
- Never put a password, Duo code, or Google authorization code into a project file or Slurm script.
- Use the public key (`.pub`) only when a service explicitly asks for an SSH public key.
