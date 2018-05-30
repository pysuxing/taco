#! python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
import os.path as path

ntc=4
nways=16
wc=128
seg=wc/ntc

height=0.8

colors = ['lightskyblue', 'tan', 'lightgrey', 'pink', 'white']
hatches = ['/', '..', '\\', 'x', None]

usehatch = True
usecolor = True

def wpart():
  fig, ax = plt.subplots()
  wpt = 3

  for t in range(ntc):
    ways = np.arange(wpt*t, wpt*(t+1))
    if usehatch and usecolor:
      ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3,
              color=colors[t], hatch=hatches[t], label='$T'+str(t)+'$')
    elif usehatch:
      ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3,
              color='None', hatch=hatches[t], label='$T'+str(t)+'$')
    else:
      ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3,
              color=colors[t], label='$T'+str(t)+'$')
  t = ntc
  ways = np.arange(wpt*ntc, nways)
  if usehatch and usecolor:
    ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3,
            color=colors[ntc], hatch=hatches[ntc])
  if usehatch:
    ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3,
            color='None', hatch=hatches[ntc])
  else:
    ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3, color=colors[ntc])

  yticks = np.arange(0,nways)
  ax.set_yticks(yticks+height/2)
  ax.set_yticklabels(yticks, fontsize='x-large')
  ax.set_ylabel('Cache ways', fontsize='xx-large')
  ax.set_xlim(0, wc+seg+4)
  xticks = np.linspace(0, wc, ntc+1).astype(int)
  ax.set_xticks(xticks+.5)
  ax.set_xticklabels([x*16 for x in xticks], fontsize='x-large')
  # ax.set_xticklabels([str(x)+'K' for x in xticks], fontsize='x-large')
  ax.set_xlabel('Cache sets', fontsize='xx-large')
  ax.legend(fontsize='x-large', loc='upper right')
  ax.tick_params(length=0)
  ax.set_frame_on(False)
  ax.set_aspect(.8 * wc/nways)
  fig.savefig('wpart.pdf')

def spart():
  fig, ax = plt.subplots()
  ways = np.arange(0, 12)
  for t in range(ntc):
    if usehatch and usecolor:
      ax.barh(ways, seg*np.ones(len(ways)), height=height, left=t*seg, linewidth=.3,
              color=colors[t], hatch=hatches[t], label='$T'+str(t)+'$')
    elif usehatch:
      ax.barh(ways, seg*np.ones(len(ways)), height=height, left=t*seg, linewidth=.3,
              hatch=hatches[t], label='$T'+str(t)+'$', color='None')
    else:
      ax.barh(ways, seg*np.ones(len(ways)), height=height, left=t*seg, linewidth=0,
              color=colors[t], label='$T'+str(t)+'$')
  t = ntc
  ways = np.arange(12, nways)
  if usehatch and usecolor:
    ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3,
            color=colors[ntc], hatch=hatches[ntc])
  if usehatch:
    ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3,
            hatch=hatches[ntc], color='None')
  else:
    ax.barh(ways, wc*np.ones(len(ways)), height=height, left=0, linewidth=.3, color=colors[ntc])

  yticks = np.arange(0,nways)
  ax.set_yticks(yticks+height/2)
  ax.set_yticklabels(yticks, fontsize='x-large')
  ax.set_ylabel('Cache ways', fontsize='xx-large')
  ax.set_xlim(0, wc+seg+4)
  xticks = np.linspace(0, wc, ntc+1).astype(int)
  ax.set_xticks(xticks+.5)
  ax.set_xticklabels([x*16 for x in xticks], fontsize='x-large')
  # ax.set_xticklabels([str(x)+'K' for x in xticks], fontsize='x-large')
  ax.set_xlabel('Cache sets', fontsize='xx-large')
  ax.legend(fontsize='x-large', loc='upper right')
  ax.tick_params(length=0)
  ax.set_frame_on(False)
  ax.set_aspect(.8 * wc/nways)
  fig.savefig('spart.pdf')

if __name__ == '__main__':
  wpart()
  spart()
  
