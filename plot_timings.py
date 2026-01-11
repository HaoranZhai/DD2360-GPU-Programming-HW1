import os, csv
import matplotlib.pyplot as plt

# 1) 选择 CSV 文件
csv_path = None
if os.path.exists('timings.csv'):
    csv_path = 'timings.csv'
else:
    for fn in os.listdir('.'):
        low = fn.lower()
        if low.startswith('timings') and (low.endswith('.csv') or low.endswith('.txt')):
            csv_path = fn
            break

if not csv_path:
    raise FileNotFoundError("No CSV found. Expect 'timings.csv' with lines like: CSV,N,H2D,Kernel,D2H,Total")

print("Using file:", csv_path)

# 2) 读取数据（兼容一整行在一个单元格、分号分隔等情况）
N_list, H2D, KERNEL, D2H = [], [], [], []
with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        # 有些编辑器会用分号或把整行包成一个单元格
        if not line.startswith('CSV'):
            continue
        if ';' in line and ',' not in line:
            parts = line.split(';')
        else:
            parts = line.split(',')
        if parts[0] != 'CSV' or len(parts) < 5:
            continue
        try:
            N_list.append(int(parts[1]))
            H2D.append(float(parts[2]))
            KERNEL.append(float(parts[3]))
            D2H.append(float(parts[4]))
        except Exception as e:
            print("Skip row:", parts, e)

if not N_list:
    raise RuntimeError("Parsed 0 rows. Ensure lines look like: CSV,512,0.10,0.27,0.11,0.49")

# 3) 作图（堆叠柱）
x = range(len(N_list))
plt.figure()
plt.bar(x, H2D, label='H2D (ms)')
plt.bar(x, KERNEL, bottom=H2D, label='Kernel (ms)')
plt.bar(x, D2H, bottom=[h+k for h,k in zip(H2D, KERNEL)], label='D2H (ms)')
plt.xticks(list(x), [str(n) for n in N_list], rotation=45, ha='right')
plt.ylabel('Time (ms)')
plt.title('Vector Add – Time Breakdown (Stacked)')
plt.legend()
plt.tight_layout()

# 4) 一定保存成 PNG（即使不弹窗）
out_png = 'timings_stacked.png'
plt.savefig(out_png, dpi=200)
print("Saved figure to", out_png)

# 5) 尝试弹窗（环境不支持也没关系）
try:
    plt.show()
except Exception as e:
    print("GUI show failed, PNG is saved:", e)
