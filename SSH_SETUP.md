# SSH Setup for Talon Cluster Access

## Problem
When running `python scripts/cluster_runner.py` from PowerShell, SSH password prompts don't work properly with subprocess automation.

## Solution: SSH Key-Based Authentication

### Step 1: Generate SSH Key Pair (if you don't have one)

```powershell
# Generate key pair (press Enter for all prompts to use defaults)
ssh-keygen -t rsa -b 4096 -f $env:USERPROFILE\.ssh\id_rsa -N ""
```

This creates:
- `~\.ssh\id_rsa` (private key - keep secret!)
- `~\.ssh\id_rsa.pub` (public key)

### Step 2: Copy Public Key to Talon

```powershell
# One-time setup: Copy your public key to Talon
cat $env:USERPROFILE\.ssh\id_rsa.pub | ssh jayapreethi.mohan@talon.und.edu "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

You'll be prompted for password once - this is normal and expected.

### Step 3: Test Password-Less SSH

```powershell
# This should NOT ask for password
ssh jayapreethi.mohan@talon.und.edu "echo OK"
```

If it works without prompting, your SSH key is properly configured!

### Step 4: Use cluster_runner.py

```powershell
python scripts/cluster_runner.py
```

This will now work without password prompts.

---

## Alternative: Manual SSH Commands

If you prefer not to set up key-based auth, you can run SSH commands directly:

```powershell
# Test connection
ssh jayapreethi.mohan@talon.und.edu "echo OK"

# Run benchmarks on cluster
ssh jayapreethi.mohan@talon.und.edu `
    "cd reaction_diffusion_scaling && python3 scripts/run_benchmark_suite.py --local-only --config config/benchmark_config.yaml"

# Copy results back
scp -r jayapreethi.mohan@talon.und.edu:reaction_diffusion_scaling/outputs/benchmark_*/ outputs/
```

---

## Troubleshooting

### "Permission denied (publickey)"
- SSH key not copied to Talon
- Run Step 2 again

### "Connection refused"
- Network issue or Talon cluster is down
- Try: `ping talon.und.edu`

### "ssh: command not found"
- OpenSSH not installed
- Install from Microsoft Store or use WSL

---

## What This Enables

Once key-based auth is set up:
- `.\benchmark.ps1 benchmark-cpu` works automatically
- No password prompts required
- Can schedule automated benchmarks
- Compatible with cron/task scheduler

