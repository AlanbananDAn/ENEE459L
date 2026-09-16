# Lab 02 — Is this the toolchain the course locked, and how do you know?

Lab 01 established what the machine is. This lab establishes what is installed on
it, which is a harder question than it sounds, because the software stack has a
failure mode the hardware does not: **it can be wrong and still work.**

A board with the wrong PyTorch wheel runs your code. It imports cleanly. It
prints a version number that looks right. It runs your model on the CPU at a
thirtieth of the speed and never says a word about it, and the first thing you
will notice is that your week-seven numbers are strange in a way you cannot
explain.

## The lab question

**Is every component on this unit the version the course pinned — and for the
ones you cannot confirm, do you know that you cannot confirm them?**

## The two machines this lab is built around

Both of these pass a casual inspection. Both have `pip list` output that looks
correct. They fail in different weeks, for different reasons, and telling them
apart is the entire assignment.

**The stock wheel.** `pip install torch` on an aarch64 machine succeeds. It
fetches a wheel from PyPI that was never built against CUDA. `import torch`
works, `torch.__version__` prints `2.5.0`, and `torch.cuda.is_available()`
returns `False`. The only trace in `pip list` is what is *missing*: an NVIDIA
build carries a local version tag like `2.5.0a0+872d972e41.nv24.08`, and a stock
one does not. Five characters at the end of a version string, which everybody
skims past.

**The sealed virtual environment.** TensorRT on Jetson is an apt package. There
is no pip wheel for it; it installs into the system `dist-packages`. A virtual
environment created without `--system-site-packages` therefore cannot see it —
and nothing complains, because nothing imports TensorRT until week nine, by
which point nobody remembers how the venv was made.

## What You Will Build

Five standalone probes reading sysfs, /proc, and CLI tools:

- **`probe_torch`**: The CPU module name from `/proc/device-tree/model`.
- **`probe_cuda`**: Total RAM in kB (always less than 8 GB on Orin Nano).
- **`probe_opencv`**: Which device the root filesystem boots from (NVMe, SD, or other).
- **`probe_tensorrt`**: Whether an NVMe drive exists in `/sys/block/nvme0n1` (fitted ≠ booted-from).
- **`probe_l4t`**: Two PCIe numbers—what the link *can* do (capability) and what it *did* do (negotiated).

All reads use a `root` parameter so tests can inject fake filesystems. When you cannot read something, you return `unknown(source, why)`, not a plausible default.

## How to Write code

The instructions for writing the code are provided in slides in `Module 1` on ELMS. The slide numbers for each of the function are mentioned below -

1. **probe_torch**: slide 6
2. **probe_cuda**: slide 8
3. **probe_opencv**: slide 10.
4. **probe_tensorrt**: slide 11.
5. **probe_l4t**: slide 13.

## How to Run

```bash
cd lab02/

# Generate the report on the board
sudo python probes.py
```

After completing the code, please validate the resulting JSON output file against `sample_system_report.json` to ensure it conforms to the expected format before submission.

## How to Debug your code

There are two ways to debug code - 
1. One way is to use breakpoints. For that, we use `pdb` the package and `pdb.set_trace()` to add a breakpoint at any line of code. 
2. Another way is to just the output by using command `print(out)` where out is output of any function. 

## How to save your work

For saving your work, you create a new branch named `solution` by running the following command.
```bash
git switch -c solution2
```

After this, push your changes with following set of commands - 
```bash
git add .
git commit -m "Adding things"
git push -u origin solution2
```

## Analysis

1. **probe_torch**: The CPU module name is readable.
2. **probe_cuda**: Total memory ≥ 6 GB (Orin Nano should report ~7.6 GB).
3. **probe_opencv**: Root filesystem is on `/dev/nvme0n1`, not `/dev/mmcblk0p1`.
4. **probe_tensorrt**: An NVMe drive exists and reports a model.
5. **probe_l4t**: Both PCIe capability and negotiated speed are known.

