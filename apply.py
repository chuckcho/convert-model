#!/usr/bin/env python

def apply(dx, fb):
    print "hello"
    print type(dx)
    print type(fb)
    for n in range(0, 30):
        dx_layer = dx.layer[n].blobs
        fb_layer = fb.layers[n].blobs
        if len(dx_layer) != len(fb_layer):
            print "PANIC", n, len(dx_layer), len(fb_layer)
            return
        for bn in range(0, len(dx_layer)):
            dx_blob = dx_layer[bn]
            fb_blob = fb_layer[bn]
            if len(dx_blob.data) != len(fb_blob.data):
                print dx.layer[n].name, fb.layers[n].name
                print "PANIC", n, bn, len(dx_blob.data), len(fb_blob.data)
                return
            print "COPYING", n, bn, len(dx_blob.data), len(fb_blob.data)
            for i in range(0, len(dx_blob.data)):
                dx_blob.data[i] = fb_blob.data[i]
            print "COPIED"
