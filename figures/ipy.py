import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path as path
from timing import loadb, statistics

ntcs = [4, 2, 1]
ncs = [16, 8, 4, 2, 1]
mems = ['0', '0-1', '0-3', '0-7']
lb = 3072
ub = 9216
x = np.arange(lb, ub+1, 128)
xindices = (x >> 7) - 1
# xticks = x[::2]
# xticklabels = np.array([str(t>>10)+'K' for t in xticks])

def summarize(prog):
  shape = (len(ntcs), len(ncs), len(mems), 9216/128)
  results = np.zeros(shape)

  for i0, ntc in enumerate(ntcs):
    for i1, nc in enumerate(ncs):
      nt = ntc * nc
      if nt < 4:
        continue
      lb = 3072
      ub = nt * 1024
      if nt <= 32 and ub > 6144:
        ub = 6144
      if ub > 9216:
        ub = 9216
      for i2, mem in enumerate(mems):
        step = 128
        size = lb
        while size <= ub:
          if size >= 6144:
            step = 512
          else:
            step = 128
          filename = prog + '_' + str(ntc) + '_' + str(nc) + '_' + mem + '_' + str(size) + '.tout'
          if not path.exists(filename):
            print('MISSING', filename)
            continue
          data = loadb(filename)
          gflops = statistics(data, peak=8.8, verbose=False)
          print(filename, gflops)
          results[i0, i1, i2, (size>>7)-1] = gflops
          size += step
  np.save(prog + '.npy', results)
  return results

def select(data, ntc, nc, mem):
  assert ntc * nc >= 4

  i0 = ntcs.index(ntc)
  i1 = ncs.index(nc)
  i2 = mems.index(mem)
  y = data[i0, i1, i2, xindices]
  indices = y != 0.0
  return x[indices], y[indices]

def plot(con, scp):
  nts = { x * y for x in ntcs for y in ncs if x * y >= 4 }
  nts = list(nts)
  nts.sort()
  figprops = dict(figsize=(10*len(nts), 10*len(mems)))
  fig, axs = plt.subplots(len(mems), len(nts), **figprops)
  
  for ntc in ntcs:
    for nc in ncs:
      nt = ntc * nc
      if nt < 4:
        continue
      c = nts.index(nt)
      for r, mem in enumerate(mems):
        label = str(ntc) + 'x' + str(nc) + ' ' + str(mem)
        ax = axs[r, c]
        xc, yc = select(con, ntc, nc, mem)
        xs, ys = select(scp, ntc, nc, mem)
        lc, = ax.plot(xc, yc, marker='o', label='con '+label)
        ls, = ax.plot(xs, ys, marker='x', label='scp '+label)
  for ax in axs.flat:
    ax.legend()
  fig.savefig('drawback.pdf')

def average(data):
  ntcs = [1,2,4]
  nts = [4,8,16,32,64]
  mem = '0-7'

  shape = (len(ntcs), len(nts))
  results = np.zeros(shape)
  for r, ntc in enumerate(ntcs):
    for c, nt in enumerate(nts):
      nc = nt//ntc
      if nc > 16:
        continue
      xs, ys = select(data, ntc, nc, mem)
      if nt == 64:
        out = (xs % 256 == 0) & (xs <= 6144)
        xs = xs[out]
        ys = ys[out]
      results[r, c] = np.mean(ys)
  return results

def gap(con, scp):
  diff = scp - con
  nts = { x * y for x in ntcs for y in ncs if x * y >= 4 }
  nts = list(nts)
  nts.sort()
  
  shape = (len(ntcs), len(ncs), len(mems))
  results = np.zeros(shape)
  for i0, ntc in enumerate(ntcs):
    for i1, nc in enumerate(ncs):
      nt = ntc * nc
      if nt < 4:
        continue
      for i2, mem in enumerate(mems):
        x, y = select(diff, ntc, nc, mem)
        results[i0, i1, i2] = np.average(y)
  print(results.shape)
  return results

if __name__ == '__main__':
  if not path.exists('v3.npy'):
    con = summarize('v3')
  else:
    con = np.load('v3.npy')
  if not path.exists('v3seg.npy'):
    scp = summarize('v3seg')
  else:
    scp = np.load('v3seg.npy')
  

  # a = average(con)
  # print(a*100)
  # diff = gap(con, scp)
  # print(diff*100)
  a = average(con)
  print(a*100)
  a = average(scp)
  print(a*100)
  a = average(scp-con)
  print(a*100)

