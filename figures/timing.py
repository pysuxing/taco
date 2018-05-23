#! python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
import os.path as path

def styles(num_threads, colors, markers):
  clen = len(colors)
  mlen = len(markers)
  if (num_threads <= min(clen, mlen)):
    return list(zip(colors, markers))
  return [(c, m) for m in markers for c in colors]

class thread_data():
  def __init__(self):
    self.kernels = None
    self.pa = None
    self.pb = None
    self.s0 = None
    self.s1 = None
    self.s2 = None
    self.init = None
    self.performance = None
    self.owners = None

def loadb(filename):
# typedef struct {
#   int32_t kind;
#   int32_t owner;
#   double start;
#   double stop;
#   double amount;
# } time_point_t;
  dt = { 'names': ['kind', 'owner', 'start', 'stop', 'amounts'],
         'formats': ['int32', 'int32', 'float64', 'float64', 'float64'] }
  tds = []
  with open(filename, 'rb') as f:
    num_threads = np.fromfile(f, 'int32', 1, '')
    num_points = np.fromfile(f, 'int32', num_threads, '')
    for i in range(num_threads):
      rtd = np.fromfile(f, dt, num_points[i], '')
      td = thread_data()
      kindices = rtd['kind'] == 0
      aindices = rtd['kind'] == 1
      bindices = rtd['kind'] == 2
      sindices0 = rtd['kind'] == 3
      sindices1 = rtd['kind'] == 4
      sindices2 = rtd['kind'] == 5
      iindices = rtd['kind'] == 6
      td.kernels = rtd[kindices]
      td.pa = rtd[aindices]
      td.pb = rtd[bindices]
      td.s0 = rtd[sindices0]
      td.s1 = rtd[sindices1]
      td.s2 = rtd[sindices2]
      td.init = rtd[iindices]
      tds.append(td)
  return tds

def statistics(data, peak, verbose=False):
  num_threads = len(data)
  jobs = np.zeros(num_threads, int)
  whole = 0.0
  time = 0.0
  for i, td in enumerate(data):
    kstarts, kstops, kamounts = td.kernels['start'], td.kernels['stop'], td.kernels['amounts']
    astarts, astops, aamounts = td.pa['start'], td.pa['stop'], td.pa['amounts']
    bstarts, bstops, bamounts = td.pb['start'], td.pb['stop'], td.pb['amounts']
    sstarts0, sstops0, samounts0 = td.s0['start'], td.s0['stop'], td.s0['amounts']
    sstarts1, sstops1, samounts1 = td.s1['start'], td.s1['stop'], td.s1['amounts']
    sstarts2, sstops2, samounts2 = td.s2['start'], td.s2['stop'], td.s2['amounts']
    istarts, istops, iamounts = td.init['start'], td.init['stop'], td.init['amounts']
    kintervals = kstops - kstarts
    aintervals = astops - astarts
    bintervals = bstops - bstarts
    s0intervals = sstops0 - sstarts0
    s1intervals = sstops1 - sstarts1
    s2intervals = sstops2 - sstarts2
    iintervals = istops - istarts
    ktotal = np.sum(kintervals)
    atotal = np.sum(aintervals)
    btotal = np.sum(bintervals)
    stotal0 = np.sum(s0intervals)
    stotal1 = np.sum(s1intervals)
    stotal2 = np.sum(s2intervals)
    itotal = np.sum(iintervals)
    total = ktotal + atotal + btotal + stotal0 + stotal1 + stotal2 + itotal
    td.performances = [itotal/total*100,
                       ktotal/total*100,
                       atotal/total*100,
                       btotal/total*100,
                       stotal0/total*100,
                       stotal1/total*100,
                       stotal2/total*100,
                       np.sum(kamounts)/ktotal,
                       np.sum(aamounts)/atotal,
                       np.sum(bamounts)/btotal]
    td.owners = np.zeros(num_threads, int)
    for entry in td.kernels:
      td.owners[int(entry[1])] += 1
    jobs += td.owners
    lb = min(np.min(istarts), np.min(kstarts), np.min(astarts), np.min(bstarts),
             np.min(sstarts0), np.min(sstarts1), np.min(sstarts2))
    ub = max(np.max(istops), np.max(kstops), np.max(astops), np.max(bstops),
             np.max(sstops0), np.max(sstops1), np.max(sstops2))
    whole += np.sum(kamounts)
    if ub - lb > time:
      time = ub - lb
  if verbose:
    for i, td in enumerate(data):
      print('{:2d} init:{:8.4f} kernel:{:8.4f} pa:{:8.4f} pb:{:8.4f} '
            's0:{:8.4f} s1:{:8.4f} s2:{:8.4f} '
            'GFLOPS:{:8.4f} ABW:{:8.4f} BBW:{:8.4f}'.format(i, *td.performances))
      fmt = '{:2d} {}|' + num_threads * ' {:d}'
    for i, td in enumerate(data):
      print(fmt.format(i, jobs[i], *td.owners))
  overall = whole/time/num_threads/peak
  return overall

def timeline(filename, data, figsize=(100, 20)):
  colors = ['black', 'blue', 'green', 'red', 'cyan', 'sandybrown', 'tan', 'purple']
  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(1,1,1)
  ax.set_ylim(0, len(data))
  for i, td in enumerate(data):
    yk = i+4/8
    ya = i+3/8
    yb = i+2/8
    ys0 = i+1/8
    ys1 = i+0.7/8
    ys2 = i+0.4/8
    ax.hlines(np.full(len(td.kernels), yk), td.kernels['start'], td.kernels['stop'], color=colors[0])
    ax.hlines(np.full(len(td.pa), ya), td.pa['start'], td.pa['stop'], color=colors[1])
    ax.hlines(np.full(len(td.pb), yb), td.pb['start'], td.pb['stop'], color=colors[2])
    ax.hlines(np.full(len(td.s0), ys0), td.s0['start'], td.s0['stop'], color=colors[3])
    ax.hlines(np.full(len(td.s1), ys1), td.s1['start'], td.s1['stop'], color=colors[4])
    ax.hlines(np.full(len(td.s2), ys2), td.s2['start'], td.s2['stop'], color=colors[5])
    # NOTE with dynamic load balancing, the sync0 time is very small 
    # ax.vlines(td.s0['start'], i+0.4/8, i+4/8, color='black', linestyle='dashed')
    # ax.vlines(td.s0['stop'], i+0.4/8, i+4/8, color='black', linestyle='dotted')
  fig.savefig(filename)

def performance(filename, data, figsize=(100, 100)):
  markers = ['o', '+', 'x', 'd', 'v', '^', '<', '>']
  colors = ['pink', 'blue', 'green', 'red', 'purple', 'tan', 'sandybrown', 'cyan']
  num_threads = len(data)
  fig = plt.figure(figsize=figsize)
  for i, td in enumerate(data):
    kax = fig.add_subplot(num_threads,1,num_threads-i)
    pax = kax.twinx()
    sax = kax.twinx()
    # kax.set_ylim(0, 9)
    # pax.set_ylim(0, 3)
    sax.set_ylim(0, 1)
    sax.get_yaxis().set_visible(False)
    
    kstarts, kstops, kamounts = td.kernels['start'], td.kernels['stop'], td.kernels['amounts']
    astarts, astops, aamounts = td.pa['start'], td.pa['stop'], td.pa['amounts']
    bstarts, bstops, bamounts = td.pb['start'], td.pb['stop'], td.pb['amounts']
    sstarts0, sstops0, samounts0 = td.s0['start'], td.s0['stop'], td.s0['amounts']
    sstarts1, sstops1, samounts1 = td.s1['start'], td.s1['stop'], td.s1['amounts']
    sstarts2, sstops2, samounts2 = td.s2['start'], td.s2['stop'], td.s2['amounts']
    istarts, istops, iamounts = td.init['start'], td.init['stop'], td.init['amounts']
    kintervals = kstops - kstarts
    aintervals = astops - astarts
    bintervals = bstops - bstarts
    gflops = kamounts/kintervals
    agbps = aamounts/aintervals
    bgbps = bamounts/bintervals
    kindices0 = td.kernels['owner'] == i
    kindices1 = td.kernels['owner'] != i
    kax.bar(kstarts[kindices0], gflops[kindices0], kintervals[kindices0], color=colors[0], linewidth=0)
    kax.bar(kstarts[kindices1], gflops[kindices1], kintervals[kindices1], color='black', linewidth=0)
    # kax.bar(kstarts, gflops, kintervals, color=colors[0], linewidth=0)
    pax.bar(astarts, agbps, aintervals, color=colors[1], linewidth=0)
    pax.bar(bstarts, bgbps, bintervals, color=colors[2], linewidth=0)
    sax.bar(sstarts0, np.ones(sstarts0.size),
            sstops0 - sstarts0, color=colors[3], linewidth=0)
    sax.bar(sstarts1, np.ones(sstarts1.size),
            sstops1 - sstarts1, color=colors[4], linewidth=0)
    sax.bar(sstarts2, np.ones(sstarts2.size),
            sstops2 - sstarts2, color=colors[5], linewidth=0)
    sax.bar(istarts, np.ones(istarts.size),
            istops - istarts, color=colors[6], linewidth=0)
    pax.plot(astarts, agbps, color=colors[1], marker='o', linewidth=0)
    pax.plot(bstarts, bgbps, color=colors[2], marker='*', linewidth=0)
    print(i)
  fig.savefig(filename)

if __name__ == '__main__':
  pass
