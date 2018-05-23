#!/usr/bin/python3

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mp
import matplotlib.transforms as mtr
import matplotlib.text as mtext
import matplotlib.lines as mline
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def grid(row, col, width=1, height=1, width_ratios=None, height_ratios=None, 
         left=0, right=0, top=0, bottom=0):
  hratios = None
  wratios = None
  if not height_ratios:
    height_ratios = np.ones(row)
    hratios = np.arange(0, row+1)
  else:
    hratios = np.empty(row+1)
    hratios[1:] = np.cumsum(height_ratios)
  if not width_ratios:
    width_ratios = np.ones(col)
    wratios = np.arange(0, col+1)
  else:
    wratios = np.empty(col+1)
    wratios[1:] = np.cumsum(width_ratios)
  runit = (height-top-bottom)/hratios[-1]
  cunit = (width-left-right)/wratios[-1]
  hratios = hratios[:-1]
  wratios = wratios[:-1]
  rbases = runit * hratios
  cbases = cunit * wratios
  xys = np.array([(left+c, height-top-r) for r in rbases for c in cbases])
  xys = xys.reshape((row, col, 2))
  whs = np.array([(w*cunit, -h*runit) for h in height_ratios for w in width_ratios])
  whs = whs.reshape((row, col, 2))
  bbs = np.empty((row, col, 4))
  bbs[:,:,:2] = xys
  bbs[:,:,2:] = whs
  return bbs

# def offsettrans(x=0, y=0):
#   trans = mtr.Affine2D.from_values(1,0,0,1,x,y)

def box(bounds):
  return mtr.Bbox.from_bounds(*bounds)
def canonicalbox(box):
  return mtr.Bbox.from_bounds(box.xmin, box.ymin, abs(box.width), abs(box.height))
def padbox(box, ratio=-.1):
  margin = ratio * min(box.width, box.height)
  return box.padded(margin)
def hcutbox(box, refbox):
  return mtr.Bbox.from_bounds(refbox.xmin, box.ymin, refbox.width, box.height)
def vcutbox(box, refbox):
  return mtr.Bbox.from_bounds(box.xmin, refbox.ymin, box.width, refbox.height)
def fitboxes(abox, ashape, bbox, bshape, cbox, cshape):
  def fitshape(box, shape):
    hwratio = shape.height/shape.width
    w, h = box.width, box.height
    if w * hwratio > h:
      box, _ = box.splitx(h/w/hwratio)
    else:
      _, box = box.splity(1-w*hwratio/h)
    return box
  def shrinkbox(box, ratio):
    box, _ = box.splitx(ratio)
    _, box = box.splity(1-ratio)
    return box

  abox = fitshape(abox, ashape)
  bbox = fitshape(bbox, bshape)
  cbox = fitshape(cbox, cshape)
  smallest = abox
  if bbox.height < abox.width:
    if bbox.width < cbox.width:
      smallest = bbox
    else:
      smallest = cbox
  else:
    if abox.height < cbox.height:
      smallest = abox
    else:
      smallest = cbox
  if smallest == abox:
    #print('shrink c by', abox.height/cbox.height)
    cbox = shrinkbox(cbox, abox.height/cbox.height)
    #print('shrink b by', abox.width/bbox.height)
    bbox = shrinkbox(bbox, abox.width/bbox.height)
  elif smallest == bbox:
    #print('shrink a by', bbox.height/abox.width)
    abox = shrinkbox(abox, bbox.height/abox.width)
    #print('shrink c by', bbox.width/cbox.width)
    cbox = shrinkbox(cbox, bbox.width/cbox.width)
  else:
    #print('shrink a by', cbox.height/abox.height)
    abox = shrinkbox(abox, cbox.height/abox.height)
    #print('shrink b by', cbox.width/bbox.width)
    bbox = shrinkbox(bbox, cbox.width/bbox.width)
  return abox, bbox, cbox
  
def transbox(trans, box):
  x, y = trans.transform_point(box.min)
  return mtr.Bbox.from_bounds(x, y, abs(box.width), abs(box.height))

def boxctext(box, s, **kargs):
  x, y = (box.xmin+box.xmax)/2, (box.ymin+box.ymax)/2
  return mtext.Text(x, y, s, ha='center', va='center', fontsize=20, **kargs)
def boxltext(box, s, **kargs):
  x, y = box.xmin, (box.ymin+box.ymax)/2
  return mtext.Text(x, y, s, ha='left', va='center', fontsize=16, **kargs)
def boxvtext(box, s, **kargs):
  x, y = (box.xmin+box.xmax)/2, (box.ymin+box.ymax)/2
  return mtext.Text(x, y, s, ha='center', va='center', rotation='vertical', fontsize=16, **kargs)
  # return boxctext(box, s, rotation='vertical', **kargs)
def boxrbrace(box, label, height_ratio, width_ratio=.1, **kargs):
  _, labelbox, _ = box.splity((1-height_ratio)/2, (1+height_ratio)/2)
  t = boxvtext(labelbox, label)
  halfx = (box.xmin + box.xmax)/2
  x = halfx - box.width*width_ratio
  # x = box.xmin*width_ratio + box.xmax*(1-width_ratio) - halfx
  vs = [(x, box.ymax), (halfx, box.ymax), (halfx, labelbox.ymax)]
  xs = [x for x, y in vs]
  ys = [y for x, y in vs]
  upbrace = mline.Line2D(xs, ys, color='black')
  vs = [(halfx, labelbox.ymin), (halfx, box.ymin), (x, box.ymin)]
  xs = [x for x, y in vs]
  ys = [y for x, y in vs]
  downbrace = mline.Line2D(xs, ys, color='black')
  return t, upbrace, downbrace
def boxtlabel(box, s, **kargs):
  if 'fontsize' not in kargs:
    kargs['fontsize'] = 15
  x, y = (box.xmin+box.xmax)/2, box.ymax
  return mtext.Text(x, y, s, va='bottom', ha='center', **kargs)
def boxblabel(box, s, **kargs):
  if 'fontsize' not in kargs:
    kargs['fontsize'] = 15
  x, y = (box.xmin+box.xmax)/2, box.ymin
  return mtext.Text(x, y, s, va='top', ha='center', **kargs)
def boxllabel(box, s, **kargs):
  if 'fontsize' not in kargs:
    kargs['fontsize'] = 15
  x, y = box.xmin, (box.ymin+box.ymax)/2
  return mtext.Text(x, y, s, va='center', ha='right', **kargs)
def boxrlabel(box, s, **kargs):
  if 'fontsize' not in kargs:
    kargs['fontsize'] = 15
  x, y = box.xmax, (box.ymin+box.ymax)/2
  return mtext.Text(x, y, s, va='center', ha='left', **kargs)

def zoom(ubox, dbox):
  (blx, bly), _, (brx, bry), _ = ubox.corners()
  _, (tlx, tly), _, (trx, try_) = dbox.corners()
  return (blx, bly, tlx-blx, tly-bly), (brx, bry, trx-brx, try_-bry)

def boxrowmajor(box, row, col):
  vs = np.empty((row, col, 4))
  runit = box.height/row
  for i in range(row):
    topbox = mtr.Bbox.from_bounds(box.xmin, box.ymin+runit*(row-1-i), 
                                  box.width, runit)
    topbox = padbox(topbox, -.1)
    cunit = topbox.width/(col-1)
    for j in range(col):
      x0 = x1 = topbox.xmin + j*cunit
      y0 = topbox.ymax
      y1 = topbox.ymin
      vs[i,j] = np.array([x0,y0, x1,y1])
  vs = vs.reshape((row*col*2, 2))
  l = mline.Line2D(vs[:, 0], vs[:, 1])
  return l
def boxcolmajor(box, row, col):
  vs = np.empty((col, row, 4))
  cunit = box.width/col
  for i in range(col):
    topbox = mtr.Bbox.from_bounds(box.xmin+cunit*i, box.ymin,
                                  cunit, box.height)
    topbox = padbox(topbox, -.1)
    runit = topbox.height/(row-1)
    for j in range(row):
      y0 = y1 = topbox.ymax - j*runit
      x0 = topbox.xmin
      x1 = topbox.xmax
      vs[i,j] = np.array([x0,y0, x1,y1])
  vs = vs.reshape((row*col*2, 2))
  l = mline.Line2D(vs[:, 0], vs[:, 1])
  return l


# def boxvline(box, pos=.5, **kargs):
#   x = box.xmin * (1-pos) + box.xmax * pos
#   l = mline.Line2D((x, x), (box.ymin, box.ymax), **kargs)
#   return l
# def boxhline(box, pos=.5, **kargs):
#   y = box.ymin * (1-pos) + box.ymax * pos
#   l = mline.Line2D((box.xmin, box.xmax), (y, y), **kargs)
#   return l
# def boxvlabel(box, s):
#   t = boxtext(box, s, rotation='vertical', ha='center', va='center')
#   l = boxvline(box)
#   return t,l
# def boxhlabel(box, s):
#   t = boxtext(box, s, ha='center', va='center')
#   l = boxhline(box)
#   return t,l

def mat(box, **kargs):
  return mp.Rectangle(box.p0, box.width, box.height, **kargs)


if __name__ == '__main__':
  width=1.2
  height=1
  width_ratios = [.02,1,.02, .02,1,.02, .02,1,.02, .1,4.5,.02, .02,.2,.2]
  height_ratios = [.3, .1,1,.1, .1,1,.1, .1,1,.1, .1,1,.1, .1,1,.1, .1,1,.1, .1,1,.1]
  col = len(width_ratios)
  row = len(height_ratios)
  zoomline = dict(ls='dotted', color='cyan', head_width=0)

  g = grid(row, col, width=width, height=height, 
           width_ratios=width_ratios, height_ratios=height_ratios,
           left=0.1, right=0.05, top=0.05, bottom=0.05)
  fig = plt.figure(figsize=(14,16))
  ax = fig.add_axes((0,0,1,1))
  ax.set_aspect(1)
  ax.set_xlim(0, width)
  ax.set_ylim(0, height)

  boxes = np.empty((row, col), dtype=mtr.Bbox)
  for r, c in [(x, y) for x in range(row) for y in range(col)]:
    boxes[r, c] = canonicalbox(box(g[r, c]))

  # banner
  abox, bbox, cbox, codebox = boxes[0, [1,4,7,10]]
  ax.add_artist(boxctext(abox, r'$A$'))
  ax.add_artist(boxctext(bbox, r'$B$'))
  ax.add_artist(boxctext(cbox, r'$C$'))
  ax.add_artist(boxctext(codebox, r'GEMM Blocked Algorithm'))
  # ax.add_artist(mat(codebox, facecolor='none'))

  # loop 1 
  tabox, tbbox, tcbox = boxes[1, [1,4,7]]
  labox, abox, lbbox, bbox, lcbox, cbox, codebox = boxes[2, [0,1,3,4,6, 7,10]]
  bbbox, bcbox = boxes[3, [4,7]]

  abox = padbox(abox)
  bbox = padbox(bbox)
  cbox = padbox(cbox)
  k = min(abs(abox.width), abs(bbox.height))
  abox, _ = abox.splitx(k/abox.width)
  bbox, _ = bbox.splity(k/bbox.height)
  for b in abox, bbox, cbox:
    ax.add_artist(mat(b, facecolor='none'))
  a0box = abox
  b0box, _ = bbox.splitx(.5)
  c0box, _ = cbox.splitx(.5)
  for b in b0box, c0box:
    ax.add_artist(mat(b, facecolor='darkgrey'))

  ta = boxtlabel(abox, r'$K$')
  tb = boxtlabel(bbox, r'$N$')
  tc = boxtlabel(cbox, r'$N$')
  la = boxllabel(abox, r'$M$')
  lb = boxllabel(bbox, r'$K$')
  lc = boxllabel(cbox, r'$M$')
  bb = boxblabel(b0box, r'$N_c$')
  bc = boxblabel(c0box, r'$N_c$')
  for t in ta, tb, tc, la, lb, lc, bb, bc:
    ax.add_artist(t)

  c2box, c1box = codebox.splity(.5)
  c1 = boxltext(c1box, r'for $jj$ = $0$:$N_c$:$N\!\!-\!\!1$')
  c2 = boxltext(c2box, r'  $A_0$=$A$; $B_0$ = $B$[:][$jj$:$jj\!\!+\!\!N_c\!\!-\!\!1$]; $C_0$ = $C$[:][$jj$:$jj\!\!+\!\!N_c\!\!-\!\!1$];')
  for c in c1, c2:
    ax.add_artist(c)
  # ax.add_artist(mat(codebox, facecolor='none'))
  
  # loop 2
  tabox, tbbox, tcbox = boxes[4, [4,4,7]]
  labox, abox, lbbox, bbox, rbbox, lcbox, cbox, codebox = boxes[5, [0,1,3,4,5,6, 7,10]]
  babox, = boxes[6, [1]]

  abox = padbox(abox)
  bbox = padbox(bbox)
  cbox = padbox(cbox)
  abox, bbox, cbox = fitboxes(abox, a0box, bbox, b0box, cbox, c0box)
  for b in abox, bbox, cbox:
    ax.add_artist(mat(b, facecolor='none'))
  l0, l1 = zoom(b0box, bbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  l0, l1 = zoom(c0box, cbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  a0box, _ = abox.splitx(.3)
  _, b0box = bbox.splity(1-.3)
  for b in a0box, b0box:
    ax.add_artist(mat(b, facecolor='darkgrey'))
  
  ta = boxtlabel(abox, r'$K$')
  tb = boxtlabel(bbox, r'$N_c$')
  tc = boxtlabel(cbox, r'$N_c$')
  la = boxllabel(abox, r'$M$')
  lb = boxllabel(bbox, r'$K$')
  lc = boxllabel(cbox, r'$M$')
  ba = boxblabel(a0box, r'$K_c$')
  rb = boxrlabel(b0box, r'$K_c$')
  for t in ta, tb, tc, la, lb, lc, ba, rb:
    ax.add_artist(t)
  # ax.add_artist(mat(babox, facecolor='none'))

  c2box, c1box = codebox.splity(.5)
  c1 = boxltext(c1box, r'  for $kk$ = $0$:$K_c$:$K\!\!-\!\!1$')
  c2 = boxltext(c2box, r'    $A_1$=$A_0$[:][$kk$:$kk\!\!+\!\!K_c\!\!-\!\!1$]; $B_1$=$B_0$[$kk$:$kk\!\!+\!\!K_c\!\!-\!\!1$][:]; $C_1$=$C_0$;')
  for c in c1, c2:
    ax.add_artist(c)

  # loop 3
  tabox, tbbox, tcbox = boxes[7, [4,4,7]]
  labox, abox, lbbox, bbox, rbbox, lcbox, cbox, codebox = boxes[8, [0,1,3,4,5,6, 7,10]]
  babox, = boxes[9, [1]]

  abox = padbox(abox)
  bbox = padbox(bbox)
  cbox = padbox(cbox)
  abox, bbox, cbox = fitboxes(abox, a0box, bbox, b0box, cbox, c0box)
  for b in abox, bbox, cbox:
    ax.add_artist(mat(b, facecolor='none'))
  l0, l1 = zoom(a0box, abox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  l0, l1 = zoom(b0box, bbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  _, a0box = abox.splity(.5)
  _, c0box = cbox.splity(.5)
  b0box=bbox
  for b in a0box, c0box:
    ax.add_artist(mat(b, facecolor='darkgrey'))
  
  ta = boxtlabel(abox, r'$K_c$')
  tb = boxtlabel(bbox, r'$N_c$')
  tc = boxtlabel(cbox, r'$N_c$')
  la = boxllabel(abox, r'$M$')
  lb = boxllabel(bbox, r'$K_c$')
  lc = boxllabel(cbox, r'$M$')
  ra = boxrlabel(a0box, r'$M_c$')
  rc = boxrlabel(c0box, r'$M_c$')
  for t in ta, tb, tc, la, lb, lc, ra, rc:
    ax.add_artist(t)

  c2box, c1box = codebox.splity(.5)
  c1 = boxltext(c1box, r'    parallel for $ii$ = $1$:$M_c$:$M\!\!-\!\!1$')
  c2 = boxltext(c2box, r'      $A_2$=pack($A_1$[$ii$:$ii\!\!+\!\!M_c\!\!-\!\!1$][:]); $B_2$=pack($B_1$); $C_2$=$C_1$[$ii$:$ii\!\!+\!\!M_c\!\!-\!\!1$][:];')
  for c in c1, c2:
    ax.add_artist(c)

  # loop 4
  tabox, tbbox, tcbox = boxes[10, [4,4,7]]
  labox, abox, lbbox, bbox, rbbox, lcbox, cbox, codebox = boxes[11, [0,1,3,4,5,6, 7,10]]
  babox, = boxes[12, [1]]

  abox = padbox(abox)
  bbox = padbox(bbox)
  cbox = padbox(cbox)
  abox, bbox, cbox = fitboxes(abox, a0box, bbox, b0box, cbox, c0box)
  for b in abox, bbox, cbox:
    ax.add_artist(mat(b, facecolor='none'))
  l0, l1 = zoom(a0box, abox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  l0, l1 = zoom(c0box, cbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)

  l = boxrowmajor(abox, 5, 6)
  ax.add_artist(l)
  l = boxcolmajor(bbox, 6, 5)
  ax.add_artist(l)
  
  b0box=bbox
  a0box = abox
  b0box, _ = bbox.splitx(.2)
  c0box, _ = cbox.splitx(.2)
  for b in b0box, c0box:
    ax.add_artist(mat(b, facecolor='darkgrey'))
  
  ta = boxtlabel(abox, r'$K_c$')
  tb = boxtlabel(bbox, r'$N_c$')
  tc = boxtlabel(cbox, r'$N_c$')
  la = boxllabel(abox, r'$M_c$')
  lb = boxllabel(bbox, r'$K_c$')
  lc = boxllabel(cbox, r'$M_c$')
  bb = boxblabel(b0box, r'$N_r$')
  bc = boxblabel(c0box, r'$N_r$')
  for t in ta, tb, tc, la, lb, lc, bb, bc:
    ax.add_artist(t)

  c2box, c1box = codebox.splity(.5)
  c1 = boxltext(c1box, r'      for $j$ = $1$:$N_r$:$N_c\!\!-\!\!1$')
  c2 = boxltext(c2box, r'        $A_3$=$A_2$; $B_3$=$B_2$[:][$j$:$j\!\!+\!\!N_r\!\!-\!\!1$]; $C_3$=$C_2$[:][$j$:$j\!\!+\!\!N_r\!\!-\!\!1$];')
  for c in c1, c2:
    ax.add_artist(c)

  # loop 5
  tabox, tbbox, tcbox = boxes[13, [4,4,7]]
  labox, abox, lbbox, bbox, rbbox, lcbox, cbox, codebox = boxes[14, [0,1,3,4,5,6, 7,10]]
  babox, = boxes[15, [1]]

  abox = padbox(abox)
  bbox = padbox(bbox)
  cbox = padbox(cbox)
  abox, bbox, cbox = fitboxes(abox, a0box, bbox, b0box, cbox, c0box)
  for b in abox, bbox, cbox:
    ax.add_artist(mat(b, facecolor='none'))
  l0, l1 = zoom(b0box, bbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  l0, l1 = zoom(c0box, cbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)

  l = boxrowmajor(abox, 5, 6)
  ax.add_artist(l)
  l = boxcolmajor(bbox, 6, 1)
  ax.add_artist(l)

  _, a0box = abox.splity(1-.2)
  b0box = bbox
  _, c0box = cbox.splity(1-.2)
  for b in a0box, c0box:
    ax.add_artist(mat(b, facecolor='darkgrey'))
  
  ta = boxtlabel(abox, r'$K_c$')
  tb = boxtlabel(bbox, r'$N_r$')
  tc = boxtlabel(cbox, r'$N_r$')
  la = boxllabel(abox, r'$M_c$')
  lb = boxllabel(bbox, r'$K_c$')
  lc = boxllabel(cbox, r'$M_c$')
  ra = boxrlabel(a0box, r'$M_r$')
  rc = boxrlabel(c0box, r'$M_r$')
  for t in ta, tb, tc, la, lb, lc, ra, rc:
    ax.add_artist(t)

  c2box, c1box = codebox.splity(.5)
  c1 = boxltext(c1box, r'        for $i$ = $0$:$M_r$:$M_c\!\!-\!\!1$')
  c2 = boxltext(c2box, r'          $A_4$=$A_3$[$i$:$i\!\!+\!\!M_r\!\!-\!\!1$][:]; $B_4$=$B_3$; $C_4$=$C_3$[$i$:$i\!\!+\!\!M_r\!\!-\!\!1$][:];')
  for c in c1, c2:
    ax.add_artist(c)

  # loop 6
  tabox, tbbox, tcbox = boxes[16, [4,4,7]]
  labox, abox, lbbox, bbox, rbbox, lcbox, cbox, codebox = boxes[17, [0,1,3,4,5,6, 7,10]]
  babox, = boxes[18, [1]]

  abox = padbox(abox)
  bbox = padbox(bbox)
  cbox = padbox(cbox)
  abox, bbox, cbox = fitboxes(abox, a0box, bbox, b0box, cbox, c0box)
  for b in abox, bbox, cbox:
    ax.add_artist(mat(b, facecolor='none'))
  l0, l1 = zoom(a0box, abox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  l0, l1 = zoom(c0box, cbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)

  l = boxrowmajor(abox, 1, 10)
  ax.add_artist(l)
  l = boxcolmajor(bbox, 10, 1)
  ax.add_artist(l)

  a0box, _ = abox.splitx(.1)
  _, b0box = bbox.splity(1-.1)
  c0box = cbox
  for b in a0box, b0box:
    ax.add_artist(mat(b, facecolor='darkgrey'))
  
  ta = boxtlabel(abox, r'$K_c$')
  tb = boxtlabel(bbox, r'$N_r$')
  tc = boxtlabel(cbox, r'$N_r$')
  la = boxllabel(abox, r'$M_r$')
  lb = boxllabel(bbox, r'$K_c$')
  lc = boxllabel(cbox, r'$M_r$')
  ba = boxblabel(a0box, r'$1$')
  rb = boxrlabel(b0box, r'$1$')
  for t in ta, tb, tc, la, lb, lc, ba, rb:
    ax.add_artist(t)

  c2box, c1box = codebox.splity(.5)
  c1 = boxltext(c1box, r'          for $k$ = $0$:$1$:$K_c\!\!-\!\!1$')
  c2 = boxltext(c2box, r'            $A_5$=$A_4$[:][$k$]; $B_5$=$B_4$[$k$][:]; $C_5$=$C_4$;')
  for c in c1, c2:
    ax.add_artist(c)

  # nano kernel
  tabox, tbbox, tcbox = boxes[19, [4,4,7]]
  labox, abox, lbbox, bbox, rbbox, lcbox, cbox, codebox = boxes[20, [0,1,3,4,5,6, 7,10]]
  babox, = boxes[21, [1]]

  abox = padbox(abox)
  bbox = padbox(bbox)
  cbox = padbox(cbox)
  abox, bbox, cbox = fitboxes(abox, a0box, bbox, b0box, cbox, c0box)
  for b in abox, bbox, cbox:
    ax.add_artist(mat(b, facecolor='none'))
  l0, l1 = zoom(a0box, abox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  l0, l1 = zoom(b0box, bbox)
  ax.arrow(*l0, **zoomline)
  ax.arrow(*l1, **zoomline)
  
  ta = boxtlabel(abox, r'$1$')
  tb = boxtlabel(bbox, r'$N_r$')
  tc = boxtlabel(cbox, r'$N_r$')
  la = boxllabel(abox, r'$M_r$')
  lb = boxllabel(bbox, r'$1$')
  lc = boxllabel(cbox, r'$M_r$')
  for t in ta, tb, tc, la, lb, lc:
    ax.add_artist(t)

  c = boxltext(codebox, r'            $C_5\!\!+\!\!= \alpha A_5 B_5$;')
  ax.add_artist(c)

  # braces
  # topbox = boxes[19, -3]
  # botbox = boxes[-1, -3]
  # left, bottom = botbox.min
  # right, top = topbox.max
  # bracebox = mtr.Bbox.from_extents(left, bottom, right, top)
  # t, upbrace, downbrace = boxrbrace(bracebox, r'nkernel', height_ratio=.7)
  # ax.add_artist(t)
  # ax.add_artist(upbrace)
  # ax.add_artist(downbrace)

  # topbox = boxes[16, -2]
  # botbox = boxes[-1, -2]
  # left, bottom = botbox.min
  # right, top = topbox.max
  # bracebox = mtr.Bbox.from_extents(left, bottom, right, top)
  # t, upbrace, downbrace = boxrbrace(bracebox, r'$\mu$kernel', height_ratio=.35)
  # ax.add_artist(t)
  # ax.add_artist(upbrace)
  # ax.add_artist(downbrace)

  topbox = boxes[11, -1]
  botbox = boxes[-1, -1]
  left, bottom = botbox.min
  right, top = topbox.max
  bracebox = mtr.Bbox.from_extents(left, bottom, right, top)
  t, upbrace, downbrace = boxrbrace(bracebox, 'Kernel (GEBP)', height_ratio=.32)
  ax.add_artist(t)
  ax.add_artist(upbrace)
  ax.add_artist(downbrace)

  topbox = boxes[2, -1]
  botbox = boxes[8, -1]
  left, bottom = botbox.min
  right, top = topbox.max
  bracebox = mtr.Bbox.from_extents(left, bottom, right, top)
  t, upbrace, downbrace = boxrbrace(bracebox, 'Strategy', height_ratio=.3)
  ax.add_artist(t)
  ax.add_artist(upbrace)
  ax.add_artist(downbrace)

  # lines
  hlines = []
  for i in range(3, len(height_ratios), 3):
    sbox = boxes[i][0]
    hlines.append(sbox.ymin)
    label = boxllabel(sbox, 'layer '+str(len(hlines)), fontsize=16)
    ax.add_artist(label)
    # ax.add_artist(mat(sbox, facecolor='none'))
  rightend = boxes[0, 10].xmax
  ax.hlines(hlines, 0, rightend, linestyle='dashed', color='darkgrey')

  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  ax.set_frame_on(False)
  fig.savefig('gemm.pdf')
  #plt.show(fig)

  # for r, c in [(x, y) for x in range(row) for y in range(col)]:
  #   b = box(g[r, c])
  #   m = mat(b, facecolor='none')
  #   ax.add_artist(m)

