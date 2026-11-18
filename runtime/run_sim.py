import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="CoralNPU V2 Custom Simulator Runtime Launcher")
    parser.add_argument("elf", help="Path to input target ELF file")
    parser.add_argument("--dtcm-addr", type=lambda x: int(x, 16), default=0x10000, 
                        help="DTCM base address for output extraction (default: 0x10000)")
    parser.add_argument("--size", type=int, default=32, 
                        help="Number of bytes to read from DTCM output (default: 32)")
    parser.add_argument("--out", type=str, default="dtcm_output.bin", 
                        help="Path to save raw DTCM binary output")
    parser.add_argument("--log-out", type=str, default="simulation.log", 
                        help="Path to save simulator stdout log file")
    parser.add_argument("--timeout", type=float, default=2.0, 
                        help="Maximum execution timeout in seconds (default: 2.0)")

    args = parser.parse_args()

    elf_path = os.path.abspath(args.elf)
    if not os.path.exists(elf_path):
        print(f"[Error] ELF file not found at: {elf_path}", file=sys.stderr)
        sys.exit(1)

    sim_binary = os.path.abspath("../third_party/coral-mpact/bazel-bin/sim/coral_runtime")
    if not os.path.exists(sim_binary):
        print(f"[Error] Simulator binary not found at {sim_binary}.", file=sys.stderr)
        print("Build it using: cd ~/compilers-btp/third_party/coral-mpact && bazel build //sim:coral_runtime", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sim_binary,
        f"--elf={elf_path}",
        f"--dtcm_addr={args.dtcm_addr:#x}",
        f"--size={args.size}",
        f"--out={os.path.abspath(args.out)}",
        "--print_stdout=true"
    ]

    print(f"[Runtime] Executing simulator (timeout: {args.timeout}s)...")
    
    try:
        # Run process with timeout, capturing stdout & stderr to log file
        with open(args.log_out, "w") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                text=True
            )
        print(f"[Runtime] Simulation finished. Logs written to {args.log_out}")
        print(f"[Runtime] DTCM data dumped to binary file {args.out}")

    except subprocess.TimeoutExpired:
        print(f"[Warning] Simulation exceeded timeout of {args.timeout} seconds and was terminated.", file=sys.stderr)
        print(f"[Runtime] Partial logs captured in {args.log_out}", file=sys.stderr)
        sys.exit(124)

    except Exception as e:
        print(f"[Error] Failed to execute simulator: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()