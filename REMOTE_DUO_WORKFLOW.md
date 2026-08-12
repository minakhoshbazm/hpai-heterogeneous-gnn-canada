# Remote Duo Workflow

Date: August 10, 2026

This project runs on the Digital Research Alliance of Canada Fir cluster at:

`/home/mkhoshba/projects/def-drozita/mkhoshba/hpai-heterogeneous-gnn-canada`

## Goal

Reduce repeated Duo prompts while working on Fir from Windows.

## What works on this setup

### 1. Keep the SSH key loaded in the Windows agent

In PowerShell:

```powershell
ssh-add "$env:USERPROFILE\.ssh\id_ed25519"
ssh-add -l
```

Expected result:

- the `id_ed25519` key is listed
- GitHub/Fir login uses the key first
- you should only need the Duo step instead of typing the full password repeatedly

### 2. Use public key + Duo, not password auth

Local SSH config file:

`C:\Users\mina_\.ssh\config`

Recommended contents:

```sshconfig
Host fir.alliancecan.ca
    User mkhoshba
    PreferredAuthentications publickey,keyboard-interactive
    PasswordAuthentication no
    PubkeyAuthentication yes
    IdentityAgent SSH_AUTH_SOCK
    ServerAliveInterval 60
    ServerAliveCountMax 120
    TCPKeepAlive yes
```

This helps keep the flow to:

- key authentication
- Duo approval

instead of:

- username/password
- then Duo

### 3. Keep one remote shell open in `tmux`

Once connected to Fir, start or reuse a persistent session:

```bash
tmux attach -t hpai || tmux new -s hpai
```

Inside that session, long-running work can continue even if the local terminal disconnects.

Detach without stopping the session:

```text
Ctrl+b, then d
```

This does not remove all Duo prompts for new SSH connections, but it prevents losing the active remote environment and reduces the need to reconnect manually.

## What did not work well here

### SSH multiplexing / persistent control socket

I tested a Windows OpenSSH `ControlMaster` / `ControlPersist` setup to try to reuse one authenticated connection for later `ssh` and `scp` calls.

That failed in this environment with:

```text
getsockname failed: Not a socket
```

Because of that, the multiplexing settings were removed from `C:\Users\mina_\.ssh\config`.

Conclusion:

- on this machine, ordinary `ssh`/`scp` works
- control-socket multiplexing is not reliable in the current Windows OpenSSH setup

## Practical recommendation for this project

Use this workflow:

1. Load the SSH key once with `ssh-add`
2. Keep the Fir shell alive in `tmux`
3. Let Codex use key + Duo for remote commands
4. Avoid closing the active Fir session unless necessary

## Important limitation

For Codex-triggered fresh `ssh` or `scp` calls from Windows, Duo approval may still be required because each new remote connection is authenticated separately when multiplexing is unavailable.

So the current best stable setup is:

- no repeated full password entry
- still occasional Duo approval for new remote connections
- persistent remote work kept alive in `tmux`
