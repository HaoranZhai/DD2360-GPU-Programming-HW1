import csv
import matplotlib.pyplot as plt

def read_cpu_rows(path):
    recs = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for row in csv.reader(f):
            if not row: 
                continue
            parts = row if len(row) > 1 else row[0].split(',')
            if parts[0] != 'CPUCSV':
                continue
            dtype = parts[1]
            m, k, n = int(parts[2]), int(parts[3]), int(parts[4])
            h2d, ker, d2h, tot = map(float, parts[5:9])
            recs.append((dtype, m, k, n, h2d, ker, d2h, tot))
    if not recs:
        raise RuntimeError(f"No CPUCSV rows found in {path}")
    recs.sort(key=lambda x: (x[1], x[3]))  # sort by m,n
    return recs

def plot_stacked_many(recs, title, outfile, ylog=False):
    labels = [f"A({r[1]}×{r[2]}) × B({r[2]}×{r[3]})" for r in recs]
    H2D  = [r[4] for r in recs]
    KER  = [r[5] for r in recs]
    D2H  = [r[6] for r in recs]
    x = list(range(len(recs)))
    plt.figure(figsize=(14,5))
    plt.bar(x, H2D, label='H2D (ms)')
    plt.bar(x, KER, bottom=H2D, label='Kernel (ms)')
    plt.bar(x, D2H, bottom=[h+k for h,k in zip(H2D, KER)], label='D2H (ms)')
    plt.xticks(x, labels, rotation=25, ha='right')
    plt.ylabel('Time (ms)')
    plt.title(title)
    if ylog:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=220)
    print(f"Saved: {outfile}")

def main():
    rec_f = read_cpu_rows('mm_float_10_big.csv')
    rec_d = read_cpu_rows('mm_double_10_big.csv')
    plot_stacked_many(rec_f, 'MatMul CPU-side Time Breakdown (float, 10 larger cases)',
                      'matmul_stacked_float_10_big.png', ylog=False)
    plot_stacked_many(rec_d, 'MatMul CPU-side Time Breakdown (double, 10 larger cases)',
                      'matmul_stacked_double_10_big.png', ylog=False)

if __name__ == '__main__':
    main()
