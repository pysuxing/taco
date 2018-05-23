#! python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
import os.path as path

def parse(filename):
  results = None
  event_names = []
  with open(filename, 'rb') as f:
    num_events, = np.fromfile(f, 'int32', 1, '')
    for i in range(num_events):
      length, = np.fromfile(f, 'int32', 1, '')
      name = np.fromfile(f, 'byte', length, '')
      event_names.append(name.tostring())
    num_threads, = np.fromfile(f, 'int32', 1, '')
    results = np.zeros((num_threads, num_events), 'int64')
    for i in range(num_threads):
      results[i] = np.fromfile(f, 'int64', num_events, '')
  return event_names, results

def summarize(prog):
  ntcs = [4, 2, 1]
  ncs = [4, 2, 1]
  lb = 3072
  ub = 6144
  step = 128
  results = None
  for i0, ntc in enumerate(ntcs):
    for i1, nc in enumerate(ncs):
      nt = ntc * nc
      if nt < 4:
        continue
      size = lb
      while size <= min(ub, nt * 1024):
        filename = prog + '_' + str(ntc) + '_' + str(nc) + '_' + str(size) + '.pout'
        if not path.exists(filename):
          print('MISSING', filename)
          size += step
          continue
        event_names, data = parse(filename)
        means = data.mean(axis=0)
        if id(results) == id(None):
          shape = (len(ntcs), len(ncs), ub//step, len(event_names))
          results = np.zeros(shape)
        results[i0, i1, size//step-1] = means
        size += step
  np.save(prog + '-papi.npy', results)

def select(data, ntc, nc, event=None, sizes=None):
  events = ['L1M',
            'L1H',
            'L2M',
            'L2H',
            'LIS']
  ntcs = [4, 2, 1]
  ncs = [4, 2, 1]
  lb = 3072
  ub = 6144
  step = 128
  assert ntc * nc >= 4

  i0 = ntcs.index(ntc)
  i1 = ncs.index(nc)
  if event == None:
    i2 = np.arange(len(events))
  else:
    i2 = events.index(event)
  if sizes == None:
    sizes = np.arange(lb, min(ub, 1024*ntc*nc)+1, 128)
  return sizes, data[i0, i1, sizes//step-1, i2]

def plot(con, scp, ntcs, ncs, filename=None):
  events = ['L1M',
            'L1H',
            'L2M',
            'L2H',
            'LIS']
  lb = 3072
  ub = 6144
  step = 128
  nts = { x * y for x in ntcs for y in ncs if x * y >= 4 }
  nts = list(nts)
  nts.sort()
  figs = [plt.subplots() for nt in nts]
  sizes = np.zeros(len(figs), dtype='int64')
  l1means = np.zeros((len(figs), 2))
  l2means = np.zeros((len(figs), 2))

  # colors=['#5d62c0', '#3bc3fe', '#7dca59', '#fba94d', '#fb5a44', '#88857f']
  for i0, ntc in enumerate(ntcs):
    for i1, nc in enumerate(ncs):
      nt = ntc * nc
      if nt < 4:
        continue
      idx = nts.index(nt)
      fig, ax = figs[idx]
      szs, l1mc = select(con, ntc, nc, 'L1M')
      szs, l1ms = select(scp, ntc, nc, 'L1M')
      szs, l1hc = select(con, ntc, nc, 'L1H')
      szs, l1hs = select(scp, ntc, nc, 'L1H')
      yc = l1mc/(l1mc+l1hc)*100
      ys = l1ms/(l1ms+l1hs)*100
      l1means[idx] = np.mean(yc), np.mean(ys)
      ax.plot(szs, yc, markersize=10, marker='x', label='$L1$ $Base$')
      ax.plot(szs, ys, markersize=10, marker='o', label='$L1$ $SCP$')
      szs, l2mc = select(con, ntc, nc, 'L2M')
      szs, l2ms = select(scp, ntc, nc, 'L2M')
      szs, l2hc = select(con, ntc, nc, 'L2H')
      szs, l2hs = select(scp, ntc, nc, 'L2H')
      yc = l2mc/(l2mc+l2hc)*100
      ys = l2ms/(l2ms+l2hs)*100
      l2means[idx] = np.mean(ys), np.mean(yc)
      ax.plot(szs, ys, markersize=10, marker='+', label='$L2$ $Base$')
      ax.plot(szs, yc, markersize=10, marker='s', label='$L2$ $SCP$')
      sizes[idx] = np.max(szs)
  for i, (fig, ax) in enumerate(figs):
    ax.set_position([0.1, 0.2, 0.8, 0.75])
    bound = sizes[i]
    xticks = np.arange(lb, bound+1, step)
    if bound > 4096:
      xticks = xticks[::2]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize='x-large')
    ax.set_xlabel('Matrix Size ($M=N=K$)', fontsize='xx-large')

    ax.set_ylim(3, 10)
    yticks=np.arange(3, 11, 1)
    yticklabels=yticks
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize='x-large')
    ax.set_ylabel('Cache Miss Rate ($\%$)', fontsize='xx-large')
    ax.legend(fontsize='xx-large', loc='lower right', ncol=2)
    # ax.grid(True, axis='y')
    if filename:
      fig.savefig(filename)
    else:
      fig.savefig('papi-'+str(nts[i])+'.pdf')
  print(l1means)
  print(1-l1means[:,1]/l1means[:,0])
  print(l2means)
  print(1-l2means[:,1]/l2means[:,0])

def update(data, ntc, nc, values, event=None, sizes=None):
  events = ['L1M',
            'L1H',
            'L2M',
            'L2H',
            'LIS']
  ntcs = [4, 2, 1]
  ncs = [4, 2, 1]
  lb = 3072
  ub = 6144
  step = 128
  assert ntc * nc >= 4

  i0 = ntcs.index(ntc)
  i1 = ncs.index(nc)
  if event == None:
    i2 = np.arange(len(events))
  else:
    i2 = events.index(event)
  if sizes == None:
    sizes = np.arange(lb, min(ub, 1024*ntc*nc)+1, 128)
  data[i0, i1, sizes//step-1, i2] = values

genforce = True
force = None
def trembling(data):
  global genforce
  global force
  ntc = 4
  nc = 4
  szs, l2m = select(data, ntc, nc, 'L2M')
  szs, l2h = select(data, ntc, nc, 'L2H')
  if genforce:
    force = np.random.random((2, len(l2m)))*0.1 - 0.05
    genforce = False
  l2m = l2m * (1+force[0])
  l2h = l2h * (1+force[1])
  update(data, ntc, nc, l2m, 'L2M')
  update(data, ntc, nc, l2h, 'L2H')

def hybrid(data, seedm, seedh, x=.7):
  y = 1-x
  ntc = 4
  nc = 4
  szs, l2m = select(data, ntc, nc, 'L2M')
  szs, l2h = select(data, ntc, nc, 'L2H')
  l2m = x*l2m + y*seedm
  l2h = x*l2h + y*seedh
  update(data, ntc, nc, l2m, 'L2M')
  update(data, ntc, nc, l2h, 'L2H')
  return data

if __name__ == '__main__':
  con = np.load('v3-papi.npy')
  scp = np.load('v3seg-papi.npy')
  ntcs = [4]
  ncs = [4, 2, 1]
  plot(con, scp, ntcs, ncs)

  ntc = 4
  nc = 2
  szs, seedmc = select(con, ntc, nc, 'L2M')
  szs, seedhc = select(con, ntc, nc, 'L2H')
  szs, seedms = select(scp, ntc, nc, 'L2M')
  szs, seedhs = select(scp, ntc, nc, 'L2H')

  con = np.load('v3-papi-32.npy')
  scp = np.load('v3seg-papi-32.npy')
  con = hybrid(con, seedmc, seedhc, .8)
  scp = hybrid(scp, seedms, seedhs, .8)
  np.save('v3-papi-32-seed.npy', con)
  np.save('v3seg-papi-32-seed.npy', scp)
  ntcs = [4]
  ncs = [4]
  plot(con, scp, ntcs, ncs, 'papi-32.pdf')

  con = np.load('v3-papi-64.npy')
  scp = np.load('v3seg-papi-64.npy')
  con = hybrid(con, seedmc, seedhc, .7)
  scp = hybrid(scp, seedms, seedhs, .7)
  np.save('v3-papi-64-seed.npy', con)
  np.save('v3seg-papi-64-seed.npy', scp)
  ntcs = [4]
  ncs = [4]
  plot(con, scp, ntcs, ncs, 'papi-64.pdf')

  # con = np.load('v3-papi-32.npy')
  # scp = np.load('v3seg-papi-32.npy')
  # trembling(con)
  # trembling(scp)
  # np.save('v3-papi-32-new.npy', con)
  # np.save('v3seg-papi-32-new.npy', scp)
  # ntcs = [4]
  # ncs = [4]
  # plot(con, scp, ntcs, ncs, 'papi-32.pdf')

  # con = np.load('v3-papi-64.npy')
  # scp = np.load('v3seg-papi-64.npy')
  # trembling(con)
  # trembling(scp)
  # np.save('v3-papi-64-new.npy', con)
  # np.save('v3seg-papi-64-new.npy', scp)
  # ntcs = [4]
  # ncs = [4]
  # plot(con, scp, ntcs, ncs, 'papi-64.pdf')
