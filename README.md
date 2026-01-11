# DD2360 GPU Programming — HW1 (CUDA Basics)

This repository contains my solution for **Homework 1** of **DD2360 GPU Programming** (KTH).  
The homework covers **GPU architecture fundamentals** and **CUDA programming basics**, including performance measurements and visualization.

---

## 📦 Repository Contents

```

DD2360HT25_HW1_Group18/
├── Q[1]/                           # Written/theory answers
├── Q[2]/                           # CUDA vector addition + benchmarks + plots
│   ├── vecAdd.cu
│   ├── timings.csv
│   ├── plot_timings.py
│   ├── timings_stacked.png
│   └── run_timings.bat            # (Windows) batch script for timing runs
├── Q[3]/                           # CUDA matrix multiplication + benchmarks + plots
│   ├── matmul.cu
│   ├── mm_float_10_big.csv
│   ├── mm_double_10_big.csv
│   ├── plot_mm_cpu_10_big.py
│   ├── matmul_stacked_float_10_big.png
│   ├── matmul_stacked_double_10_big.png
│   └── run_mm10_bigger.bat        # (Windows) batch script for benchmark runs
├── Q[4]/                           # Written answers / discussion
└── Assignment I GPU architecture and CUDA Basics.pdf  # (Optional) assignment statement

````

> Note: The assignment PDF may be omitted from the public repo depending on course policy.

---

## ✅ Requirements

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** (nvcc)
- For plotting (optional):
  - **Python 3**
  - `matplotlib` (and typical scientific Python dependencies)

---

## 🚀 Build & Run

### Q[2] — Vector Addition (CUDA)

Compile:
```bash
cd "Q[2]"
nvcc -O2 -lineinfo -std=c++17 vecAdd.cu -o vecAdd
````

Run (example):

```bash
./vecAdd 263149
```

Generate plot from `timings.csv`:

```bash
python plot_timings.py
```

**Windows batch script note**
If `run_timings.bat` contains an absolute local path like:

```
cd /d "C:\Users\..."
```

replace it with:

```
cd /d "%~dp0"
```

so the script becomes portable on other machines.

---

### Q[3] — Matrix Multiplication (CUDA)

Compile (float, default):

```bash
cd "Q[3]"
nvcc -O2 -lineinfo -DFAST_CHECK=1 matmul.cu -o matmul_float
```

Run (example):

```bash
./matmul_float 128 256 32 16 16
```

Compile (double):

```bash
nvcc -O2 -lineinfo -DUSE_DOUBLE=1 -DFAST_CHECK=1 matmul.cu -o matmul_double
```

Run (example):

```bash
./matmul_double 128 256 32 16 16
```

Generate plots (from existing CSVs):

```bash
python plot_mm_cpu_10_big.py
```

**Windows batch script note**
Similarly, for `run_mm10_bigger.bat`, replace any absolute `cd` path with:

```
cd /d "%~dp0"
```

---

## 📊 Results & Plots

* Vector addition timing breakdown:

  * `Q[2]/timings_stacked.png`
* Matrix multiplication timing breakdown (float/double):

  * `Q[3]/matmul_stacked_float_10_big.png`
  * `Q[3]/matmul_stacked_double_10_big.png`

CSV benchmark data:

* `Q[2]/timings.csv`
* `Q[3]/mm_float_10_big.csv`
* `Q[3]/mm_double_10_big.csv`

---

## 🧠 Notes on Academic Integrity

This repository is published for **portfolio/learning** purposes.
If you are currently taking this course, please follow your institution’s **academic integrity policy**.

---

## 🏷️ Suggested Topics (GitHub)

Add these in **About → Topics** to make the project easier to search:
`cuda`, `gpu-programming`, `dd2360`, `kth`, `benchmark`, `vector-addition`, `matrix-multiplication`, `performance`

---

## 📄 License

MIT License.
