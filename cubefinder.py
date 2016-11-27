#!/usr/bin/env python2

import argparse
import cv
import cv2
import json
import logging
import os
import sys
from math import sin, cos, pi, atan2, sqrt
from numpy import matrix
from pprint import pformat
from time import sleep

log = logging.getLogger(__name__)


# To capture a single png from the webcam
# fswebcam --device /dev/video0 --no-timestamp --no-title --no-subtitle --no-banner --no-info -r 640x480 --png 3 image.png

# To convert a jpg to png
# mogrify -format png *.jpg


def intersect_seg(x1, x2, x3, x4, y1, y2, y3, y4):
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    if abs(den) < 0.1:
        return (False, (0, 0), (0, 0))

    ua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
    ub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
    ua = ua / den
    ub = ub / den
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (True, (ua, ub), (x, y))


def ptdst(p1, p2):
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def ptdstw(p1, p2):
    # return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

    # test if hue is reliable measurement
    if p1[1] < 100 or p2[1] < 100:
        # hue measurement will be unreliable. Probably white stickers are present
        # leave this until end
        return 300.0 + abs(p1[0] - p2[0])
    else:
        return abs(p1[0] - p2[0])


def ptdst3(p1, p2):
    dist = sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1])
                * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]))
    if (p1[0] > 245 and p1[1] > 245 and p1[2] > 245):
        # the sticker could potentially be washed out. Lets leave it to the end
        dist = dist + 300.0
    return dist


def compfaces(f1, f2):
    totd = 0
    for p1 in f1:
        mind = 10000
        for p2 in f2:
            d = ptdst(p1, p2)
            if d < mind:
                mind = d
        totd += mind
    return totd / 4


def avg(p1, p2):
    return (0.5 * p1[0] + 0.5 * p2[0], 0.5 * p2[1] + 0.5 * p2[1])


def areclose(t1, t2, t):
    # is t1 close to t2 within t?
    return abs(t1[0] - t2[0]) < t and abs(t1[1] - t2[1]) < t


def winded(p1, p2, p3, p4):
    # return the pts in correct order based on quadrants
    avg = (0.25 * (p1[0] + p2[0] + p3[0] + p4[0]), 0.25 * (p1[1] + p2[1] + p3[1] + p4[1]))
    ps = [(atan2(p[1] - avg[1], p[0] - avg[0]), p) for p in [p1, p2, p3, p4]]
    ps.sort(reverse=True)
    return [p[1] for p in ps]

def neighbors(f, s):
    """
    return tuple of neighbors given face and sticker indeces
    """
    if f == 0 and s == 0:
        return ((1, 2), (4, 0))
    if f == 0 and s == 1:
        return ((4, 3),)
    if f == 0 and s == 2:
        return ((4, 6), (3, 0))
    if f == 0 and s == 3:
        return ((1, 5),)
    if f == 0 and s == 5:
        return ((3, 3),)
    if f == 0 and s == 6:
        return ((1, 8), (5, 2))
    if f == 0 and s == 7:
        return ((5, 5),)
    if f == 0 and s == 8:
        return ((3, 6), (5, 8))

    if f == 1 and s == 0:
        return ((2, 2), (4, 2))
    if f == 1 and s == 1:
        return ((4, 1),)
    if f == 1 and s == 2:
        return ((4, 0), (0, 0))
    if f == 1 and s == 3:
        return ((2, 5),)
    if f == 1 and s == 5:
        return ((0, 3),)
    if f == 1 and s == 6:
        return ((2, 8), (5, 0))
    if f == 1 and s == 7:
        return ((5, 1),)
    if f == 1 and s == 8:
        return ((0, 6), (5, 2))

    if f == 2 and s == 0:
        return ((4, 8), (3, 2))
    if f == 2 and s == 1:
        return ((4, 5),)
    if f == 2 and s == 2:
        return ((4, 2), (1, 0))
    if f == 2 and s == 3:
        return ((3, 5),)
    if f == 2 and s == 5:
        return ((1, 3),)
    if f == 2 and s == 6:
        return ((3, 8), (5, 6))
    if f == 2 and s == 7:
        return ((5, 3),)
    if f == 2 and s == 8:
        return ((1, 6), (5, 0))

    if f == 3 and s == 0:
        return ((4, 6), (0, 2))
    if f == 3 and s == 1:
        return ((4, 7),)
    if f == 3 and s == 2:
        return ((4, 8), (2, 0))
    if f == 3 and s == 3:
        return ((0, 5),)
    if f == 3 and s == 5:
        return ((2, 3),)
    if f == 3 and s == 6:
        return ((0, 8), (5, 8))
    if f == 3 and s == 7:
        return ((5, 7),)
    if f == 3 and s == 8:
        return ((2, 6), (5, 6))

    if f == 4 and s == 0:
        return ((1, 2), (0, 0))
    if f == 4 and s == 1:
        return ((1, 1),)
    if f == 4 and s == 2:
        return ((1, 0), (2, 2))
    if f == 4 and s == 3:
        return ((0, 1),)
    if f == 4 and s == 5:
        return ((2, 1),)
    if f == 4 and s == 6:
        return ((0, 2), (3, 0))
    if f == 4 and s == 7:
        return ((3, 1),)
    if f == 4 and s == 8:
        return ((3, 2), (2, 0))

    if f == 5 and s == 0:
        return ((1, 6), (2, 8))
    if f == 5 and s == 1:
        return ((1, 7),)
    if f == 5 and s == 2:
        return ((1, 8), (0, 6))
    if f == 5 and s == 3:
        return ((2, 7),)
    if f == 5 and s == 5:
        return ((0, 7),)
    if f == 5 and s == 6:
        return ((2, 6), (3, 8))
    if f == 5 and s == 7:
        return ((3, 7),)
    if f == 5 and s == 8:
        return ((3, 6), (0, 8))


class RubiksFinder(object):

    def __init__(self, S1, S2):
        den = 2
        self.width = S1 / den
        self.height = S2 / den

        self.sg = cv.CreateImage((self.width, self.height), 8, 3)
        self.sgc = cv.CreateImage((self.width, self.height), 8, 3)
        self.hsv = cv.CreateImage((self.width, self.height), 8, 3)
        self.dst2 = cv.CreateImage((self.width, self.height), 8, 1)
        self.d = cv.CreateImage((self.width, self.height), cv.IPL_DEPTH_16S, 1)
        self.d2 = cv.CreateImage((self.width, self.height), 8, 1)
        self.b = cv.CreateImage((self.width, self.height), 8, 1)

        self.lastdetected = 0
        self.THR = 70
        self.dects = 50  # ideal number of number of lines detections

        self.hue = cv.CreateImage((self.width, self.height), 8, 1)
        self.sat = cv.CreateImage((self.width, self.height), 8, 1)
        self.val = cv.CreateImage((self.width, self.height), 8, 1)

        # stores the coordinates that make up the face. in order: p,p1,p3,p2 (i.e.) counterclockwise winding
        self.prevface = [(0, 0), (5, 0), (0, 5)]
        self.dodetection = True
        self.onlyBlackCubes = False

        self.succ = 0  # number of frames in a row that we were successful in finding the outline
        self.tracking = False
        self.win_size = 5
        self.flags = 0
        self.detected = 0

        self.grey = cv.CreateImage((self.width, self.height), 8, 1)
        self.prev_grey = cv.CreateImage((self.width, self.height), 8, 1)
        self.pyramid = cv.CreateImage((self.width, self.height), 8, 1)
        self.prev_pyramid = cv.CreateImage((self.width, self.height), 8, 1)

        self.ff = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, shear=0, thickness=1, lineType=8)

        self.undetectednum = 100
        self.extract = False
        self.selected = 0
        self.colors = [[] for i in range(6)]
        self.center_pixels = [[] for i in range(6)]
        self.hsvs = [[] for i in range(6)]
        self.assigned = [[-1 for i in range(9)] for j in range(6)]

        for i in range(6):
            self.assigned[i][4] = i

        self.didassignments = False

        # Used only for visualization purposes
        self.mycols = [(0, 127, 255),   # orange
                       (20, 240, 20),   # green
                       (0, 0, 255),     # red
                       (200, 0, 0),     # blue
                       (0, 255, 255),   # yellow
                       (255, 255, 255)] # white

    def analyze_frame(self, frame):
        cv.Resize(frame, self.sg)
        cv.Copy(self.sg, self.sgc)
        cv.CvtColor(self.sg, self.grey, cv.CV_RGB2GRAY)

        # tracking mode
        if self.tracking:
            self.verify_still_tracking()

        # detection mode
        if not self.tracking:
            self.detect()

        if self.tracking:
            # we are in tracking mode, we need to fill in pt[] array
            # calculate the pt array for drawing from features
            p = self.features[0]
            p1 = self.features[1]
            p2 = self.features[2]

            v1 = (p1[0] - p[0], p1[1] - p[1])
            v2 = (p2[0] - p[0], p2[1] - p[1])

            self.pt = [(p[0] - v1[0] - v2[0], p[1] - v1[1] - v2[1]),
                       (p[0] + 2 * v2[0] - v1[0], p[1] + 2 * v2[1] - v1[1]),
                       (p[0] + 2 * v1[0] - v2[0], p[1] + 2 * v1[1] - v2[1])]

            self.prevface = [self.pt[0], self.pt[1], self.pt[2]]

        # use pt[] array to do drawing
        if (self.detected or self.undetectednum < 1) and self.dodetection:
            self.draw_circles_and_lines()

        # draw faces of the extracted cubes (hit the spacebar to extract
        # the face of the currently detected cube face)
        self.draw_extracted_cube_faces()

        self.lastdetected = len(self.li)
        # swapping for LK
        self.prev_grey, self.grey = self.grey, self.prev_grey
        self.prev_pyramid, self.pyramid = self.pyramid, self.prev_pyramid

        # Display a circle around each detected square (this bars on EV3)
        # cv.ShowImage("Fig", self.sg)

    def verify_still_tracking(self):
        self.detected = 2

        # compute optical flow
        self.features, status, track_error = cv.CalcOpticalFlowPyrLK(
            self.prev_grey, self.grey, self.prev_pyramid, self.pyramid,
            self.features,
            (self.win_size, self.win_size), 3,
            (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 20, 0.03),
            self.flags)

        # set back the points we keep
        self.features = [p for (st, p) in zip(status, self.features) if st]

        if len(self.features) < 4:
            self.tracking = False  # we lost it, restart search
            log.info("tracking -> not tracking: len features %d < 4" % len(self.features))
        else:
            # make sure that in addition the distances are consistent
            ds1 = ptdst(self.features[0], self.features[1])
            ds2 = ptdst(self.features[2], self.features[3])

            if max(ds1, ds2) / min(ds1, ds2) > 1.4:
                self.tracking = False
                log.info("tracking -> not tracking: max/min ds1 %s, ds2 %s > 1.4" % (ds1, ds2))

            ds3 = ptdst(self.features[0], self.features[2])
            ds4 = ptdst(self.features[1], self.features[3])

            if max(ds3, ds4) / min(ds3, ds4) > 1.4:
                self.tracking = False
                log.info("tracking -> not tracking: max/min ds3 %s, ds4 %s > 1.4" % (ds3, ds4))

            if ds1 < 10 or ds2 < 10 or ds3 < 10 or ds4 < 10:
                self.tracking = False
                log.info("tracking -> not tracking: ds1 %s, ds2 %s, ds3 %s, ds4 %s" % (ds1, ds2, ds3, ds4))

            if not self.tracking:
                self.detected = 0

    def detect(self):
        self.detected = 0
        cv.Smooth(self.grey, self.dst2, cv.CV_GAUSSIAN, 3)
        cv.Laplace(self.dst2, self.d)
        cv.CmpS(self.d, 8, self.d2, cv.CV_CMP_GT)

        if self.onlyBlackCubes:
            # can also detect on black lines for improved robustness
            cv.CmpS(grey, 100, b, cv.CV_CMP_LT)
            cv.And(b, d2, d2)

        # these weights should be adaptive. We should always detect 100 lines
        if self.lastdetected > self.dects:
            self.THR = self.THR + 1

        if self.lastdetected < self.dects:
            self.THR = max(2, self.THR - 1)

        self.li = cv.HoughLines2(self.d2, cv.CreateMemStorage(), cv.CV_HOUGH_PROBABILISTIC, 1, 3.1415926 / 45, self.THR, 10, 5)

        # store angles for later
        angs = []
        for (p1, p2) in self.li:
            # cv.Line(sg,p1,p2,(0,255,0))
            a = atan2(p2[1] - p1[1], p2[0] - p1[0])
            if a < 0:
                a += pi
            angs.append(a)

        # log.info("THR %d, lastdetected %d, dects %d, houghlines %d, angles: %s" % (self.THR, self.lastdetected, self.dects, len(self.li), pformat(angs)))

        # lets look for lines that share a common end point
        t = 10
        totry = []

        for i in range(len(self.li)):
            p1, p2 = self.li[i]

            for j in range(i + 1, len(self.li)):
                q1, q2 = self.li[j]

                # test lengths are approximately consistent
                dd1 = sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
                dd2 = sqrt((q2[0] - q1[0]) * (q2[0] - q1[0]) + (q2[1] - q1[1]) * (q2[1] - q1[1]))

                if max(dd1, dd2) / min(dd1, dd2) > 1.3:
                    continue

                matched = 0
                if areclose(p1, q2, t):
                    IT = (avg(p1, q2), p2, q1, dd1)
                    matched = matched + 1

                if areclose(p2, q2, t):
                    IT = (avg(p2, q2), p1, q1, dd1)
                    matched = matched + 1

                if areclose(p1, q1, t):
                    IT = (avg(p1, q1), p2, q2, dd1)
                    matched = matched + 1

                if areclose(p2, q1, t):
                    IT = (avg(p2, q1), q2, p1, dd1)
                    matched = matched + 1

                if matched == 0:
                    # not touching at corner... try also inner grid segments hypothesis?
                    self.p1 = (float(p1[0]), float(p1[1]))
                    self.p2 = (float(p2[0]), float(p2[1]))
                    self.q1 = (float(q1[0]), float(q1[1]))
                    self.q2 = (float(q2[0]), float(q2[1]))
                    success, (ua, ub), (x, y) = intersect_seg(self.p1[0], self.p2[0], self.q1[0], self.q2[0], self.p1[1], self.p2[1], self.q1[1], self.q2[1])

                    if success and ua > 0 and ua < 1 and ub > 0 and ub < 1:
                        # if they intersect
                        # cv.Line(sg, p1, p2, (255,255,255))
                        ok1 = 0
                        ok2 = 0

                        if abs(ua - 1.0 / 3) < 0.05:
                            ok1 = 1

                        if abs(ua - 2.0 / 3) < 0.05:
                            ok1 = 2

                        if abs(ub - 1.0 / 3) < 0.05:
                            ok2 = 1

                        if abs(ub - 2.0 / 3) < 0.05:
                            ok2 = 2

                        if ok1 > 0 and ok2 > 0:
                            # ok these are inner lines of grid
                            # flip if necessary
                            if ok1 == 2:
                                self.p1, self.p2 = self.p2, self.p1

                            if ok2 == 2:
                                self.q1, self.q2 = self.q2, self.q1

                            # both lines now go from p1->p2, q1->q2 and
                            # intersect at 1/3
                            # calculate IT
                            z1 = (self.q1[0] + 2.0 / 3 * (self.p2[0] - self.p1[0]), self.q1[1] + 2.0 / 3 * (self.p2[1] - self.p1[1]))
                            z2 = (self.p1[0] + 2.0 / 3 * (self.q2[0] - self.q1[0]), self.p1[1] + 2.0 / 3 * (self.q2[1] - self.q1[1]))
                            z = (self.p1[0] - 1.0 / 3 * (self.q2[0] - self.q1[0]), self.p1[1] - 1.0 / 3 * (self.q2[1] - self.q1[1]))
                            IT = (z, z1, z2, dd1)
                            matched = 1

                # only single one matched!! Could be corner
                if matched == 1:

                    # also test angle
                    a1 = atan2(p2[1] - p1[1], p2[0] - p1[0])
                    a2 = atan2(q2[1] - q1[1], q2[0] - q1[0])

                    if a1 < 0:
                        a1 += pi

                    if a2 < 0:
                        a2 += pi

                    ang = abs(abs(a2 - a1) - pi / 2)

                    if ang < 0.5:
                        totry.append(IT)
                        # cv.Circle(sg, IT[0], 5, (255,255,255))

        # now check if any points in totry are consistent!
        # t=4
        res = []
        for i in range(len(totry)):

            p, p1, p2, dd = totry[i]
            a1 = atan2(p1[1] - p[1], p1[0] - p[0])
            a2 = atan2(p2[1] - p[1], p2[0] - p[0])

            if a1 < 0:
                a1 += pi

            if a2 < 0:
                a2 += pi

            dd = 1.7 * dd
            evidence = 0

            # cv.Line(sg,p,p2,(0,255,0))
            # cv.Line(sg,p,p1,(0,255,0))

            # affine transform to local coords
            A = matrix([[p2[0] - p[0], p1[0] - p[0], p[0]], [p2[1] - p[1], p1[1] - p[1], p[1]], [0, 0, 1]])
            Ainv = A.I

            v = matrix([[p1[0]], [p1[1]], [1]])

            # check likelihood of this coordinate system. iterate all lines
            # and see how many align with grid
            for j in range(len(self.li)):

                # test angle consistency with either one of the two angles
                a = angs[j]
                ang1 = abs(abs(a - a1) - pi / 2)
                ang2 = abs(abs(a - a2) - pi / 2)

                if ang1 > 0.1 and ang2 > 0.1:
                    continue

                # test position consistency.
                q1, q2 = self.li[j]
                qwe = 0.06

                # test one endpoint
                v = matrix([[q1[0]], [q1[1]], [1]])
                vp = Ainv * v

                # project it
                if vp[0, 0] > 1.1 or vp[0, 0] < -0.1:
                    continue

                if vp[1, 0] > 1.1 or vp[1, 0] < -0.1:
                    continue

                if abs(vp[0, 0] - 1 / 3.0) > qwe and abs(vp[0, 0] - 2 / 3.0) > qwe and \
                        abs(vp[1, 0] - 1 / 3.0) > qwe and abs(vp[1, 0] - 2 / 3.0) > qwe:
                        continue

                # the other end point
                v = matrix([[q2[0]], [q2[1]], [1]])
                vp = Ainv * v

                if vp[0, 0] > 1.1 or vp[0, 0] < -0.1:
                    continue

                if vp[1, 0] > 1.1 or vp[1, 0] < -0.1:
                    continue

                if abs(vp[0, 0] - 1 / 3.0) > qwe and abs(vp[0, 0] - 2 / 3.0) > qwe and \
                        abs(vp[1, 0] - 1 / 3.0) > qwe and abs(vp[1, 0] - 2 / 3.0) > qwe:
                        continue

                # cv.Circle(sg, q1, 3, (255,255,0))
                # cv.Circle(sg, q2, 3, (255,255,0))
                # cv.Line(sg,q1,q2,(0,255,255))
                evidence += 1

            res.append((evidence, (p, p1, p2)))

        minch = 10000
        res.sort(reverse=True)
        # log.info("dects %s, res:\n%s" % (self.dects, pformat(res)))

        if len(res) > 0:
            minps = []
            pt = []

            # among good observations find best one that fits with last one
            for i in range(len(res)):

                if res[i][0] > 0.05 * self.dects:
                    # OK WE HAVE GRID
                    p, p1, p2 = res[i][1]
                    p3 = (p2[0] + p1[0] - p[0], p2[1] + p1[1] - p[1])

                    # cv.Line(sg,p,p1,(0,255,0),2)
                    # cv.Line(sg,p,p2,(0,255,0),2)
                    # cv.Line(sg,p2,p3,(0,255,0),2)
                    # cv.Line(sg,p3,p1,(0,255,0),2)
                    # cen=(0.5*p2[0]+0.5*p1[0],0.5*p2[1]+0.5*p1[1])
                    # cv.Circle(sg, cen, 20, (0,0,255),5)
                    # cv.Line(sg, (0,cen[1]), (320,cen[1]),(0,255,0),2)
                    # cv.Line(sg, (cen[0],0), (cen[0],240), (0,255,0),2)

                    w = [p, p1, p2, p3]
                    p3 = (self.prevface[2][0] + self.prevface[1][0] - self.prevface[0][0],
                          self.prevface[2][1] + self.prevface[1][1] - self.prevface[0][1])
                    tc = (self.prevface[0], self.prevface[1], self.prevface[2], p3)
                    ch = compfaces(w, tc)

                    # log.info("ch %s, minch %s" % (ch, minch))
                    if ch < minch:
                        minch = ch
                        minps = (p, p1, p2)

            # log.info("minch %d, minps:\n%s" % (minch, pformat(minps)))

            if len(minps) > 0:
                self.prevface = minps

                if minch < 10:
                    # good enough!
                    self.succ += 1
                    self.pt = self.prevface
                    self.detected = 1
                    # log.info("detected %d, succ %d" % (self.detected, self.succ))

            else:
                self.succ = 0

            # log.info("succ %d\n\n" % self.succ)

            # we matched a few times same grid
            # coincidence? I think NOT!!! Init LK tracker
            if self.succ > 2:

                # initialize features for LK
                pt = []
                for i in [1.0 / 3, 2.0 / 3]:
                    for j in [1.0 / 3, 2.0 / 3]:
                        pt.append((self.p0[0] + i * self.v1[0] + j * self.v2[0], self.p0[1] + i * self.v1[1] + j * self.v2[1]))

                self.features = pt
                self.tracking = True
                self.succ = 0
                log.info("non-tracking -> tracking: succ %d" % self.succ)

    def draw_circles_and_lines(self):

        # undetectednum 'fills in' a few detection to make
        # things look smoother in case we fall out one frame
        # for some reason
        if not self.detected:
            self.undetectednum += 1
            self.pt = self.lastpt

        if self.detected:
            self.undetectednum = 0
            self.lastpt = self.pt

        # convert to HSV
        cv.CvtColor(self.sgc, self.hsv, cv.CV_RGB2HSV)
        cv.Split(self.hsv, self.hue, self.sat, self.val, None)

        pt_int = []
        for (foo, bar) in self.pt:
            pt_int.append((int(foo), int(bar)))

        # do the drawing. pt array should store p,p1,p2
        self.p3 = (self.pt[2][0] + self.pt[1][0] - self.pt[0][0], self.pt[2][1] + self.pt[1][1] - self.pt[0][1])
        p2_int = (int(self.p2[0]), int(self.p2[1]))
        p3_int = (int(self.p3[0]), int(self.p3[1]))

        cv.Line(self.sg, pt_int[0], pt_int[1], (0, 255, 0), 2)
        cv.Line(self.sg, pt_int[1], p3_int, (0, 255, 0), 2)
        cv.Line(self.sg, p3_int, pt_int[2], (0, 255, 0), 2)
        cv.Line(self.sg, pt_int[2], pt_int[0], (0, 255, 0), 2)

        # first sort the points so that 0 is BL 1 is UL and 2 is BR
        pt = winded(self.pt[0], self.pt[1], self.pt[2], self.p3)

        # find the coordinates of the 9 places we want to extract over
        self.v1 = (pt[1][0] - pt[0][0], pt[1][1] - pt[0][1])
        self.v2 = (pt[3][0] - pt[0][0], pt[3][1] - pt[0][1])
        self.p0 = (pt[0][0], pt[0][1])

        ep = []
        i = 1
        j = 5
        for k in range(9):
            ep.append((self.p0[0] + i * self.v1[0] / 6.0 + j * self.v2[0] / 6.0,
                       self.p0[1] + i * self.v1[1] / 6.0 + j * self.v2[1] / 6.0))
            i = i + 2
            if i == 7:
                i = 1
                j = j - 2

        rad = ptdst(self.v1, (0.0, 0.0)) / 6.0
        cs = []
        center_pixels = []
        hsvcs = []
        den = 2

        for i, p in enumerate(ep):
            if p[0] > rad and p[0] < self.width - rad and p[1] > rad and p[1] < self.height - rad:

                # valavg=val[int(p[1]-rad/3):int(p[1]+rad/3),int(p[0]-rad/3):int(p[0]+rad/3)]
                # mask=cv.CreateImage(cv.GetDims(valavg), 8, 1 )

                col = cv.Avg(self.sgc[int(p[1] - rad / den):int(p[1] + rad / den),
                                      int(p[0] - rad / den):int(p[0] + rad / den)])

                col = cv.Avg(self.sgc[int(p[1] - rad / den):int(p[1] + rad / den),
                                      int(p[0] - rad / den):int(p[0] + rad / den)])

                p_int = (int(p[0]), int(p[1]))
                cv.Circle(self.sg, p_int, int(rad), col, -1)

                if i == 4:
                    cv.Circle(self.sg, p_int, int(rad), (0, 255, 255), 2)
                else:
                    cv.Circle(self.sg, p_int, int(rad), (255, 255, 255), 2)

                hueavg = cv.Avg(self.hue[int(p[1] - rad / den):int(p[1] + rad / den),
                                         int(p[0] - rad / den):int(p[0] + rad / den)])
                satavg = cv.Avg(self.sat[int(p[1] - rad / den):int(p[1] + rad / den),
                                         int(p[0] - rad / den):int(p[0] + rad / den)])

                cv.PutText(self.sg, repr(int(hueavg[0])), (p_int[0] + 70, p_int[1]), self.ff, (255, 255, 255))
                cv.PutText(self.sg, repr(int(satavg[0])), (p_int[0] + 70, p_int[1] + 10), self.ff, (255, 255, 255))

                if self.extract:
                    cs.append(col)
                    center_pixels.append(p_int)
                    hsvcs.append((hueavg[0], satavg[0]))

        if self.extract:
            self.extract = False
            self.colors[self.selected] = cs
            self.center_pixels[self.selected] = center_pixels
            self.hsvs[self.selected] = hsvcs
            self.selected = min(self.selected + 1, 5)

    def draw_extracted_cube_faces(self):
        """
        Draw faces of the extracted cubes (hit the spacebar to extract
        the face of the currently detected cube face)
        """
        x = 20
        y = 20
        s = 13

        for i in range(6):
            if not self.colors[i]:
                x += 3 * s + 10
                continue

            # draw the grid on top
            cv.Rectangle(self.sg, (x - 1, y - 1), (x + 3 * s + 5, y + 3 * s + 5), (0, 0, 0), -1)
            x1, y1 = x, y
            x2, y2 = x1 + s, y1 + s

            for j in range(9):
                if self.didassignments:
                    cv.Rectangle(self.sg, (x1, y1), (x2, y2), self.mycols[self.assigned[i][j]], -1)
                else:
                    cv.Rectangle(self.sg, (x1, y1), (x2, y2), self.colors[i][j], -1)
                x1 += s + 2
                if j == 2 or j == 5:
                    x1 = x
                    y1 += s + 2
                x2, y2 = x1 + s, y1 + s
            x += 3 * s + 10

        # draw the selection rectangle
        x = 20
        y = 20
        for i in range(self.selected):
            x += 3 * s + 10
        cv.Rectangle(self.sg, (x - 1, y - 1), (x + 3 * s + 5, y + 3 * s + 5), (0, 0, 255), 2)

    def process_colors(self, useRGB=True):
        """
        User hits the letter p, assign a color to each square
        """
        # assign all colors
        bestj = 0
        besti = 0
        matchesto = 0
        bestd = 10001
        taken = [0 for i in range(6)]
        done = 0
        opposite = {0: 2, 1: 3, 2: 0, 3: 1, 4: 5, 5: 4}  # dict of opposite faces

        # possibilities for each face
        poss = {}
        for j, f in enumerate(self.hsvs):
            for i, s in enumerate(f):
                poss[j, i] = range(6)

        # we are looping different arrays based on the useRGB flag
        toloop = self.hsvs
        if useRGB:
            toloop = self.colors

        while done < 8 * 6:
            bestd = 10000
            forced = False

            for j, f in enumerate(toloop):
                for i, s in enumerate(f):
                    if i != 4 and self.assigned[j][i] == -1 and (not forced):

                        # this is a non-center sticker.
                        # find the closest center
                        considered = 0
                        for k in poss[j, i]:

                            # all colors for this center were already assigned
                            if taken[k] < 8:

                                # use Euclidean in RGB space or more elaborate
                                # distance metric for Hue Saturation
                                if useRGB:
                                    dist = ptdst3(s, toloop[k][4])
                                else:
                                    dist = ptdstw(s, toloop[k][4])

                                considered += 1
                                if dist < bestd:
                                    bestd = dist
                                    bestj = j
                                    besti = i
                                    matchesto = k

                        # IDEA: ADD PENALTY IF 2ND CLOSEST MATCH IS CLOSE TO FIRST
                        # i.e. we are uncertain about it
                        if besti == i and bestj == j:
                            bestcon = considered

                        if considered == 1:
                            # this sticker is forced! Terminate search
                            # for better matches
                            forced = True
                            log.info('sticker (%s, %s) had color forced!' % (i, j))

            # assign it
            done = done + 1

            self.assigned[bestj][besti] = matchesto

            op = opposite[matchesto]  # get the opposite side

            # remove this possibility from neighboring stickers
            # since we cant have red-red edges for example
            # also corners have 2 neighbors. Also remove possibilities
            # of edge/corners made up of opposite sides
            ns = neighbors(bestj, besti)
            log.info("neighbors: %s" % pformat(ns))
            log.info("poss: %s" % pformat(poss))

            for neighbor in ns:
                if neighbor in poss:
                    p = poss[neighbor]

                    if matchesto in p:
                        p.remove(matchesto)
                    if op in p:
                        p.remove(op)
                else:
                    log.warning("process_colors %s not in poss" % pformat(neighbor))

            taken[matchesto] += 1

        self.didassignments = True

    def process_keyboard_input(self):
        c = cv.WaitKey(10) % 0x100

        if c == 27: # ESC
            return False

        # processing depending on the character
        if 32 <= c and c < 128:
            cc = chr(c).lower()

            # EXTRACT COLORS!!!
            if cc == ' ':
                log.info("extract colors")
                self.extract = True

            elif cc == 'r':
                log.info("reset")
                # reset
                self.extract = False
                self.selected = 0
                self.colors = [[] for i in range(6)]
                self.didassignments = False
                self.assigned = [[-1 for i in range(9)] for j in range(6)]
                for i in range(6):
                    self.assigned[i][4] = i
                self.didassignments = False

            elif cc == 'n':
                log.info("n - shift left")
                self.selected = self.selected - 1

                if self.selected < 0:
                    self.selected = 5

            elif cc == 'm':
                log.info("m - shift right")
                self.selected = self.selected + 1

                if self.selected > 5:
                    self.selected = 0

            elif cc == 'b':
                self.onlyBlackCubes = not self.onlyBlackCubes
                log.info("onlyBlackCubes is now %s" % self.onlyBlackCubes)

            elif cc == 'd':
                self.dodetection = not self.dodetection
                log.info("dodetection is now %s" % self.dodetection)

            elif cc == 'q':
                print(self.hsvs)

            elif cc == 'p':
                log.info("extract colors")
                self.process_colors()

            elif cc == 'u':
                self.didassignments = not self.didassignments
                log.info("didassignments is now %s" % self.didassignments)

            elif cc == 's':
                from time import time
                log.info("save image")
                cv.SaveImage(repr(time()) + ".jpg", self.sgc)

        return True


def analyze_file(filename):
    """
    Assuming filename is a png that contains a rubiks cube, return
    the RGB values for all 9 squares
    """
    img = cv.LoadImage(filename)

    (S1, S2) = cv.GetSize(img)
    rf = RubiksFinder(S1, S2)
    display_window = False

    if display_window:
        cv.NamedWindow("Fig", cv.CV_WINDOW_NORMAL)
        cv.ResizeWindow("Fig", rf.width, rf.height)

    ATTEMPTS = 100
    rf.extract = True

    for x in xrange(ATTEMPTS):
        rf.analyze_frame(img)

        # we can "track" (find the cube in the pic) but it takes ~30 attempts...why ~30?
        log.info("analyze_frame %d/%d: tracking %s, THR %s" % (x, ATTEMPTS-1, rf.tracking, rf.THR))

        if rf.tracking:
            # log.warning("analyze_frame selected %s\ncolors\n%s\n\nhsvs\n%s\n\n" %
            #          (rf.selected,
            #           pformat(rf.colors),
            #           pformat(rf.hsvs)))

            if display_window:
                cv.ShowImage('foobar', img)
                cv.WaitKey(0)
            break
    else:
        raise Exception("Could not find the cube in %s" % filename)

    if display_window:
        cv.DestroyWindow("Fig")

    # When analyzing a file we are only examining one side so use index 0 for colors and center_pixels
    colors = rf.colors[0]
    center_pixels = rf.center_pixels[0]
    colors_final = []
    # log.warning("rf.colors\n %s" % pformat(rf.colors))
    # log.warning("rf.center_pixels\n%s" % pformat(rf.center_pixels))
    # log.warning("colors\n %s" % pformat(colors))
    # log.warning("center_pixels\n%s" % pformat(center_pixels))

    # opencv returns BGR, not RGB
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#basic-ops
    for ((x, y), (blue, green, red, _)) in zip(center_pixels, colors):

        # Save in RGB, this makes the colors much easier to print on a web page
        # for troubleshooting, it is also the format expected by rubiks_rgb_solver.py
        colors_final.append({
            'x' : x, 
            'y' : y, 
            'red': int(red),
            'green' : int(green),
            'blue' : int(blue)
        })

    return colors_final


def analyze_webcam(width, height):
    print("""
    ' ' : extract colors of detected face
    'b' : toggle onlyBlackCubes
    'd' : toggle dodetection
    'm' : shift right
    'n' : shift left
    'r' : reset everything
    'q' : print hsvs
    'p' : resolve colors
    'u' : toggle didassignments
    's' : save image
""")

    # 0 for laptop camera
    # 1 for usb camera
    capture = cv.CreateCameraCapture(0)

    # Set the capture resolution
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, width)
    cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, height)

    # Create the window and set the size to match the capture resolution
    cv.NamedWindow("Fig", cv.CV_WINDOW_NORMAL)
    cv.ResizeWindow("Fig", width, height)

    frame = cv.QueryFrame(capture)
    rf = RubiksFinder(width, height)

    while True:
        frame = cv.QueryFrame(capture)

        if not frame:
            cv.WaitKey(0)
            break

        rf.analyze_frame(frame)

        if not rf.process_keyboard_input():
            break

    cv.DestroyWindow("Fig")


if __name__ == '__main__':
    '''
    Notes on resolutions
    352x240  : The default, this is what runs smoothly on EV3
    640x480  : what this program used originally
    800x600  : highest I can use (on my laptop) far that works smoothly
    1024x768 : works but takes a second to get going
    1280x720 : max for my camera, takes too much cpu
    '''
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)7s %(filename)12s: %(message)s')
    parser = argparse.ArgumentParser("Find a Rubiks cube in an image or video feed")
    parser.add_argument("--width", default=352, type=int)
    parser.add_argument("--height", default=240, type=int)
    parser.add_argument("-f", "--filename", type=str)

    args = parser.parse_args()
    log = logging.getLogger(__name__)

    # Color the errors and warnings in red
    logging.addLevelName(logging.ERROR, "\033[91m  %s\033[0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.WARNING, "\033[91m%s\033[0m" % logging.getLevelName(logging.WARNING))

    if args.filename:
        rgb_all_squares = analyze_file(args.filename)
        print(json.dumps(rgb_all_squares, indent=4))

    else:
        analyze_webcam(args.width, args.height)
