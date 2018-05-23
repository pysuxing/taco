#! python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
import os.path as path
from timing import loadb, statistics, performance

def papi():
  lb = 3072
  ub = 6144
  step = 128
  sizes = np.arange(lb, ub+1, step)
  scp = np.load('v3seg-4x4-1.npy')
  pri = np.load('v3pb-4x4-1.npy')

  l1ms, l1hs, l2ms, l2hs = scp
  l1mp, l1hp, l2mp, l2hp = pri

  fig, ax = plt.subplots()
  ys = l1ms/(l1ms+l1hs)*100
  yp = l1mp/(l1mp+l1hp)*100
  l1, = ax.plot(sizes, ys, markersize=10, marker='o', label='$L1$ $SCP$')
  l2, = ax.plot(sizes, yp, markersize=10, marker='x', label='$L1$ $SCP$-$P$')
  ys = l2ms/(l2ms+l2hs)*100
  yp = l2mp/(l2mp+l2hp)*100
  l3, = ax.plot(sizes, ys, markersize=10, marker='s', label='$L2$ $SCP$')
  l4, = ax.plot(sizes, yp, markersize=10, marker='+', label='$L2$ $SCP$-$P$')

  ax.set_position([0.1, 0.2, 0.8, 0.75])
  xticks = sizes
  if np.max(sizes) > 4096:
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
  fig.savefig('privb-papi.pdf')

def breakdown():
  scp = np.load('breakdown-scp.npy')
  pri = np.load('breakdown-pri.npy')
  lb = 3072
  ub = 6144
  step = 128
  sizes = np.arange(lb, ub+1, step)
  sampled = False
  if sampled:
    scp = scp[::2]
    pri = pri[::2]
    xticks = sizes[::2]
    sizes = sizes[::2]
    width = 90
  else:
    xticks = sizes
    width = 45
  
  x0 = sizes - width
  x1 = sizes

  yscp = np.zeros(scp.shape)
  yscp = np.cumsum(scp, axis=1)
  ypri = np.zeros(pri.shape)
  ypri = np.cumsum(pri, axis=1)
  # print(yscp)
  # print(ypri)
  fig, ax = plt.subplots()
  ax.set_position([0.1, 0.2, 0.8, 0.75])

  scp_o = ax.bar(x0, scp[:, 0], width=width, bottom=0,
                 color='yellow', label='$SCP$ $Sync$')
  scp_a = ax.bar(x0, scp[:, 1], width=width, bottom=yscp[:, 0],
                 color='skyblue', label='$SCP$ $PackA$')
  scp_b = ax.bar(x0, scp[:, 2], width=width, bottom=yscp[:, 1],
                 color='tan', label='$SCP$ $PackB$')
  pri_o = ax.bar(x1, pri[:, 0], width=width, bottom=0,
                 color='yellow', hatch='//', label='$SCP$-$P$ $Sync$')
  pri_a = ax.bar(x1, pri[:, 1], width=width, bottom=ypri[:, 0],
                 color='skyblue', hatch='//', label='$SCP$-$P$ $PackA$')
  pri_b = ax.bar(x1, pri[:, 2], width=width, bottom=ypri[:, 1],
                 color='tan', hatch='//', label='$SCP$-$P$ $PackB$')
  ax.set_xlim(lb-step, ub+step)
  ax.set_xticks(xticks)
  ax.set_xticklabels(xticks, rotation=45, fontsize='x-large')
  ax.set_xlabel('Matrix Size ($M=N=K$)', fontsize='xx-large')
  ax.set_ylim(0, 10)
  yticks=np.arange(0, 11, 1)
  yticklabels=yticks
  ax.set_yticks(yticks)
  ax.set_yticklabels(yticklabels, fontsize='x-large')
  ax.set_ylabel('Occupancy Ratio ($\%$)', fontsize='xx-large')
  ax.legend(fontsize='xx-large', loc='upper right', ncol=2)
  fig.set_figwidth(2*fig.get_figwidth())
  fig.savefig('privb-breakdown.pdf')
  print(np.mean(scp, axis=0))
  print(np.mean(pri, axis=0))

def ate():
  scp = np.load('ate-scp.npy')
  pri = np.load('ate-pri.npy')

  lb = 3072
  ub = 6144
  step = 128
  sizes = np.arange(lb, ub+1, step)
  
  fig, ax = plt.subplots()
  ax.set_position([0.1, 0.2, 0.8, 0.75])

  ax.plot(sizes, scp*100, markersize=10, marker='o', label='$SCP$')
  ax.plot(sizes, pri*100, markersize=10, marker='x', label='$SCP$-$P$')

  # ax.set_xlim(lb-step, ub+step)
  xticks = sizes[::2]
  ax.set_xticks(xticks)
  ax.set_xticklabels(xticks, rotation=45, fontsize='x-large')
  ax.set_xlabel('Matrix Size ($M=N=K$)', fontsize='xx-large')
  ax.set_ylim(50, 95)
  yticks=np.arange(50, 96, 5)
  yticklabels=yticks
  ax.set_yticks(yticks)
  ax.set_yticklabels(yticklabels, fontsize='x-large')
  ax.set_ylabel('Average Thread Efficiency ($\%$)', fontsize='xx-large')

  ax.legend(fontsize='xx-large', loc='lower left', ncol=1)
  fig.savefig('privb-ate.pdf')

def boxcolmajor(box, row, col):
  x, y = box.get_xy()
  w = box.get_width()
  h = box.get_height()

  vs = np.empty((col, row, 4))
  cunit = w / col
  runit = h / row
  for i in range(col):
    xi = x + i * cunit
    xii = xi + cunit
    xi = xi + cunit*.2
    xii = xii - cunit*.2
    for j in range(row):
      yj = y + h - j * runit
      yj = yj - runit*.5
      yjj = yj
      vs[i,j] = np.array([xi,yj,xii,yjj])
  vs = vs.reshape((row*col*2, 2))
  l = mpl.lines.Line2D(vs[:, 0], vs[:, 1])
  return l

def strategies():
  ntc=4
  nc=4
  nt=ntc * nc
  xs = np.arange(nt)
  ys = np.ones(nt)
  fontdict=dict(verticalalignment='center', horizontalalignment='center', rotation='vertical',
                fontsize='x-large')
  tid = 1
  # conventional
  fig, ax = plt.subplots()
  recs = ax.bar(xs, ys, width=1, color='None', linewidth=2)
  for i, rec in enumerate(recs):
    x, y = rec.get_xy()
    w = rec.get_width()
    h = rec.get_height()
    cx = x + w/2
    cy = y + h/2
    # t = ax.text(cx, cy, str(i), fontdict=fontdict)
    if i == tid:
      l = boxcolmajor(rec, 8, 2)
      l.set_color('blue')
      ax.add_artist(l)
      
  ax.set_aspect(2)
  # ax.set_axis_off()
  ax.tick_params(length=0)
  ax.set_frame_on(False)
  ax.set_xticks(xs+.5)
  ax.set_xticklabels(xs, fontsize='x-large')
  # ax.set_xlabel('Subtasks of packing $B_2$', fontsize='xx-large')
  ax.get_yaxis().set_visible(False);
  fig.savefig('strategy-conventional.pdf')
  # full
  fig, ax = plt.subplots()
  recs = ax.bar(xs, ys, width=1, color='None', linewidth=2)
  for i, rec in enumerate(recs):
    x, y = rec.get_xy()
    w = rec.get_width()
    h = rec.get_height()
    cx = x + w/2
    cy = y + h/2
    # t = ax.text(cx, cy, '0-15', fontdict=fontdict)
    l = boxcolmajor(rec, 8, 2)
    l.set_color('blue')
    ax.add_artist(l)
  ax.set_aspect(2)
  # ax.set_axis_off()
  ax.tick_params(length=0)
  ax.set_frame_on(False)
  ax.set_xticks(xs+.5)
  ax.set_xticklabels(xs, fontsize='x-large')
  # ax.set_xlabel('Subtasks of packing $B_2$', fontsize='xx-large')
  ax.get_yaxis().set_visible(False);
  fig.savefig('strategy-full.pdf')
  # partial
  fig, ax = plt.subplots()
  recs = ax.bar(xs, ys, width=1, color='None', linewidth=2)
  labels = ['0-3','4-7','8-11','12-15']
  for i, rec in enumerate(recs):
    x, y = rec.get_xy()
    w = rec.get_width()
    h = rec.get_height()
    cx = x + w/2
    cy = y + h/2
    # t = ax.text(cx, cy, labels[i//4], fontdict=fontdict)
    if i//4 == tid//4:
      l = boxcolmajor(rec, 8, 2)
      l.set_color('blue')
      ax.add_artist(l)
  ax.set_aspect(2)
  # ax.set_axis_off()
  ax.tick_params(length=0)
  ax.set_frame_on(False)
  ax.set_xticks(xs+.5)
  ax.set_xticklabels(xs, fontsize='x-large')
  # ax.set_xlabel('Subtasks of packing $B_2$', fontsize='xx-large')
  ax.get_yaxis().set_visible(False);
  fig.savefig('strategy-partial.pdf')

if __name__ == '__main__':
  papi()
  breakdown()
  ate()
  strategies()
