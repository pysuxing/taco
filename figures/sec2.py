import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path as path
from timing import loadb, statistics

lb = 3072
ub = 9216
x = np.arange(lb, ub+1, 128)
xindices = (x >> 7) - 1

def select(data, ntc, nc, mem):
  assert ntc * nc >= 4
  ntcs = [4, 2, 1]
  ncs = [16, 8, 4, 2, 1]
  mems = ['0', '0-1', '0-3', '0-7']

  i0 = ntcs.index(ntc)
  i1 = ncs.index(nc)
  i2 = mems.index(mem)
  y = data[i0, i1, i2, xindices]
  indices = y != 0.0
  return x[indices], y[indices]

if __name__ == '__main__':
  con = np.load('v3.npy')
  scp = np.load('v3seg.npy')
  ntcs = [4, 2]
  ncs = [16, 8, 4, 2, 1]
  mems = ['0-7']
  nts = [4, 8, 16, 32, 64]
  markers = ['o', 'x', '*', 's']
  figs = [plt.subplots() for nt in nts]
  for i0, ntc in enumerate(ntcs):
    for i1, nc in enumerate(ncs):
      nt = ntc * nc
      if nt not in nts:
        continue
      idx = nts.index(nt)
      fig, ax = figs[idx]
      for mem in mems:
        xc, yc = select(con, ntc, nc, mem)
        xs, ys = select(scp, ntc, nc, mem)
        if nt == 64:
          out = (xs % 256 == 0) & (xs <= 6144)
          xc = xc[out]
          yc = yc[out]
          xs = xs[out]
          ys = ys[out]
        label = '$NT_C='+str(ntc)+'$' + ' ' + '$NC='+str(nc)+'$'
        yc *= 100
        ys *= 100
        if ntc == 2:
          ax.plot(xc, yc, markersize=10, marker='x', label=label+ ' $Base$')
          ax.plot(xs, ys, markersize=10, marker='o', label=label+ ' $SCP$')
        else:
          ax.plot(xc, yc, markersize=10, marker='+', label=label+ ' $Base$')
          ax.plot(xs, ys, markersize=10, marker='s', label=label+ ' $SCP$')
  for i, (fig, ax) in enumerate(figs):
    ax.set_position([0.1, 0.2, 0.8, 0.75])
    left, right = ax.get_xlim()
    if right >= 9216:
      xticks = x[::4]
    elif right >= 6144:
      xticks = x[::2]
    else:
      xticks = x
    rs = xticks <= right
    ax.set_xticks(xticks[rs])
    ax.set_xticklabels(xticks[rs], rotation=45, fontsize='x-large')
    ax.set_ylim(50, 95)
    yticklabels=np.arange(50, 100, 5)
    ax.set_yticklabels(yticklabels, fontsize='x-large')
    ax.legend(loc='lower left', fontsize='xx-large')
    ax.set_xlabel('Matrix Size ($M=N=K$)', fontsize='xx-large')
    ax.set_ylabel('Average Thread Efficiency ($\%$)', fontsize='xx-large')
    fig.savefig('sec2-' + str(nts[i]) + '.pdf')
