#! python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
import os.path as path

def boxlabel(ax, box, label, pos, distance=.2, fontdict=dict(), inside=False, arrows=True):
  assert pos in ['n','s','e','w']
  lbcolor = 'black'
  if 'color' in fontdict:
    lbcolor = fontdict['color']
  factor = 1
  if inside:
    factor = -1
  x, y = box.get_xy()
  w = box.get_width()
  h = box.get_height()
  if pos == 'n':
    tx = x + w/2
    ty = y + h + h*distance*factor
    a0x0, a0y0 = tx-distance*w, ty
    a0x1, a0y1 = x, ty
    a1x0, a1y0 = tx+distance*w, ty
    a1x1, a1y1 = x+w, ty
  elif pos == 's':
    tx = x + w/2
    ty = y - h*distance*factor
    a0x0, a0y0 = tx-distance*w, ty
    a0x1, a0y1 = x, ty
    a1x0, a1y0 = tx+distance*w, ty
    a1x1, a1y1 = x+w, ty
  elif pos == 'e':
    ty = y+h/2
    tx = x + w + w*distance*factor
    a0x0, a0y0 = tx, ty+distance*h
    a0x1, a0y1 = tx, y+h
    a1x0, a1y0 = tx, ty-distance*h
    a1x1, a1y1 = tx, y
  else:
    ty = y+h/2
    tx = x - w*distance*factor
    a0x0, a0y0 = tx, ty+distance*h
    a0x1, a0y1 = tx, y+h
    a1x0, a1y0 = tx, ty-distance*h
    a1x1, a1y1 = tx, y
  t = ax.text(tx, ty, label, fontdict=fontdict)
  if not arrows:
    return
  if pos == 'n':
    a0x0, a0y0 = tx-distance*w, ty
    a0x1, a0y1 = x+w/10, ty
    a1x0, a1y0 = tx+distance*w, ty
    a1x1, a1y1 = x+w-w/10, ty
    bar0x = [x, x]
    bar0y = [ty-h*distance/2, ty+h*distance/2]
    bar1x = [x+w, x+w]
    bar1y = bar0y
  elif pos == 's':
    a0x0, a0y0 = tx-distance*w, ty
    a0x1, a0y1 = x+w/10, ty
    a1x0, a1y0 = tx+distance*w, ty
    a1x1, a1y1 = x+w-w/10, ty
    bar0x = [x, x]
    bar0y = [ty-h*distance/2, ty+h*distance/2]
    bar1x = [x+w, x+w]
    bar1y = bar0y
  elif pos == 'e':
    a0x0, a0y0 = tx, ty+distance*h
    a0x1, a0y1 = tx, y+h-w/10
    a1x0, a1y0 = tx, ty-distance*h
    a1x1, a1y1 = tx, y+w/10
    bar0x = [tx-w*distance/2, tx+w*distance/2]
    bar0y = [y, y]
    bar1x = bar0x
    bar1y = [y+h, y+h]
  else:
    a0x0, a0y0 = tx, ty+distance*h
    a0x1, a0y1 = tx, y+h-distance*w/2
    a1x0, a1y0 = tx, ty-distance*h
    a1x1, a1y1 = tx, y+distance*w/2
    bar0x = [tx-w*distance/2, tx+w*distance/2]
    bar0y = [y, y]
    bar1x = bar0x
    bar1y = [y+h, y+h]
  a0 = ax.arrow(a0x0, a0y0, a0x1-a0x0, a0y1-a0y0, color=lbcolor, head_width=.05)
  a1 = ax.arrow(a1x0, a1y0, a1x1-a1x0, a1y1-a1y0, color=lbcolor, head_width=.05)
  ax.plot(bar0x, bar0y, color=lbcolor)
  ax.plot(bar1x, bar1y, color=lbcolor)

def boxrowmajor(box, row, col):
  x, y = box.get_xy()
  w = box.get_width()
  h = box.get_height()

  vs = np.empty((row, col, 4))
  cunit = w / col
  runit = h / row
  for i in range(row):
    yi = y + h - i * runit
    yii = yi - runit
    yi = yi - runit*.2
    yii = yii + runit*.2
    for j in range(col):
      xj = x + j * cunit
      xj = xj + cunit*.5
      xjj = xj
      vs[i,j] = np.array([xj,yi, xjj,yii])
  vs = vs.reshape((row*col*2, 2))
  l = mpl.lines.Line2D(vs[:, 0], vs[:, 1])
  return l

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

def workload():
  nt = 4
  xs = np.arange(nt)
  ys = np.ones(nt)
  lbcolor='blue'
  fontdict = dict(verticalalignment='center', horizontalalignment='center',
                  fontsize='22', color=lbcolor)
  fontdict1 = dict(verticalalignment='center', horizontalalignment='center',
                   fontsize='30')
  colors = ['lightskyblue', 'tan', 'lightgrey', 'pink', 'white']
  hatches = ['/', '..', '\\', 'x', None]
  hlcolor='darkgrey'
  fig, ax = plt.subplots()

  nc = 1.5
  lw = 1.5
  # b
  xs = np.arange(nt)*nc+2
  ys = np.ones(nt)
  recs = ax.bar(xs, ys, width=nc, bottom=5, color=hlcolor, linewidth=lw)
  for i, rec in enumerate(recs):
    x, y = rec.get_xy()
    w = rec.get_width()
    h = rec.get_height()
    cx = x + w/2
    cy = y + h/2
    # t = ax.text(cx, cy, str(i), fontdict=fontdict)
    if i == 1:
      l = boxcolmajor(rec, 8, 4)
      l.set_color('blue')
      ax.add_artist(l)
  boxlabel(ax, recs[1], '$K_c$', 'w', distance=.25, fontdict=fontdict)
  boxlabel(ax, recs[1], '$\\frac{N_c}{NT}$', 's', distance=.35, fontdict=fontdict)
  boxlabel(ax, recs[0], '$B_1$', 'w', distance=.5, fontdict=fontdict1, arrows=False)
  # a
  xs = np.ones(nt)
  ys = np.arange(nt)
  recs = ax.barh(ys, xs, height=1, color=['none','none',hlcolor,'none'], linewidth=lw)
  for i, rec in enumerate(recs):
    x, y = rec.get_xy()
    w = rec.get_width()
    h = rec.get_height()
    cx = x + w/2
    cy = y + h/2
    # t = ax.text(cx, cy, str(nt-i-1), fontdict=fontdict)
    if nt-i-1 == 1:
      l = boxrowmajor(rec, 4, 8)
      l.set_color('blue')
      ax.add_artist(l)
  boxlabel(ax, recs[nt-1-1], '$K_c$', 'n', distance=.25, fontdict=fontdict)
  boxlabel(ax, recs[nt-1-1], '$M_c$', 'e', distance=.25, fontdict=fontdict)
  boxlabel(ax, recs[-1], '$A_1$', 'n', distance=.5, fontdict=fontdict1, arrows=False)
  # c
  xs = np.arange(nt)*nc+2
  ys = np.ones(nt)
  for r in range(nt):
    color = 'none'
    if nt-r-1 == 1:
      color = hlcolor
    recs = ax.bar(xs, ys, width=nc, bottom=r, color=color, linewidth=lw)
    for c, rec in enumerate(recs):
      x, y = rec.get_xy()
      w = rec.get_width()
      h = rec.get_height()
      cx = x + w/2
      cy = y + h/2
      # t = ax.text(cx, cy, str(nt-r-1), fontdict=fontdict)
  t = ax.text(1.5, 4.5, '$C_1$', fontdict=fontdict1)

  ax.set_aspect(1)
  ax.set_axis_off()
  # ax.tick_params(length=0)
  # ax.set_frame_on(False)
  # ax.set_xticks(xs+.5)
  # ax.set_xticklabels(xs, fontsize='x-large')
  # ax.set_xlabel('Subtasks of packing $B_2$', fontsize='xx-large')
  # ax.get_yaxis().set_visible(False);
  fig.savefig('workload.pdf')

if __name__ == '__main__':
  workload()
