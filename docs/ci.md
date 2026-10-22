# Continuous integration

Two workflows in `.github/workflows/`.

## `tests.yml`: every test on the CPU

On every push, pull request and by hand. The whole suite (`pytest tests`) on the CPU of a GitHub hosted
`ubuntu-latest` runner, one job each on Python 3.10 and 3.13; a job stops after 30 minutes. Nothing is left out: the tests
that need CuPy and a GPU (marked `gpu`) skip themselves on these runners, as do the tests that need PyTorch (the
`[ml]` extra is not installed). Each job lists the skipped tests
with their reasons (`-rs`) and prints the number of tests, skipped tests, failures and errors from its JUnit report.

Measured on two CPU cores of the development machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`, `taskset -c 0,1`,
Python 3.12): 253 passed and 37 skipped in 5 minutes 46 seconds, so the suite is not split; the slowest tests are the
tiny scenario's end to end runs (`test_ci_end_to_end.py`, 50 s and 18 s). The 37 skipped: the GPU tests and the
PyTorch tests.

## `gpu-tests.yml`: the GPU tests

By hand (`workflow_dispatch`) and every Monday at 04:17 UTC. The tests marked `gpu` (the CUDA kernels of the FDTD and
of the chain, and the tests that compare the GPU with the CPU result; `tests/harbor/_gpu.py` defines the marker) on a
self hosted runner with the labels `self-hosted` and `gpu`: `pytest tests -m gpu`. The job stops after 10 minutes of
running. A job waits in the queue until a runner with those labels is online; GitHub keeps a queued job for up to
24 hours and then fails it, so with no runner registered the workflow fails a day later rather than at once (the job
timeout counts only once a runner has taken the job).

### Registering a Linux machine with an NVIDIA card as the runner

The steps of the GitHub documentation ("Adding self-hosted runners", "Using labels with self-hosted runners",
"Configuring the self-hosted runner application as a service"), for this workflow:

1. On the machine: an NVIDIA driver and `nvidia-smi` working, CUDA 12 compatible (the `[cuda]` extra installs
   `cupy-cuda12x`), Python 3.10 or newer with `venv`, and `git`.
2. On GitHub, in the repository: **Settings**, **Actions**, **Runners**, **New self-hosted runner**, choose **Linux**
   and the machine's architecture. The page shows the commands with a registration token valid for one hour.
3. On the machine, as an ordinary user (not root), in an empty directory: run the download and extract commands the
   page shows (`curl -o actions-runner-linux-x64-<version>.tar.gz -L <url>` and `tar xzf ...`).
4. Configure it with the label the workflow asks for:
   `./config.sh --url https://github.com/<owner>/<repo> --token <token> --labels gpu`
   (a self hosted runner gets the labels `self-hosted`, `Linux` and `X64` by itself; `--labels gpu` adds `gpu`).
5. Run it: `./run.sh` in a terminal, or as a service that starts with the machine:
   `sudo ./svc.sh install` and `sudo ./svc.sh start` (`sudo ./svc.sh status` shows it).
6. The runner appears under **Settings**, **Actions**, **Runners** as idle with the labels `self-hosted`, `Linux`,
   `X64`, `gpu`. Start **GPU tests** from the **Actions** tab (**Run workflow**) to check it.

A self hosted runner runs the code of every workflow run that reaches it; GitHub advises against self hosted runners on
public repositories, where a pull request from a fork could run code on the machine.
