#!/usr/bin/env python

import facebook_caffe_pb2 as fb
import dextro_caffe_pb2 as dx

def convert_from_dx_to_fb(dx, fb):
    print type(dx)
    print type(fb)
    for n in range(len(fb.layers)):
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

# WIP
def populate(dx, fb):
    print type(dx)
    print type(fb)
    for n in range(len(fb.layers)):
        fb_layer = fb.layers[n].blobs
        dx_layer = dx.layer[n].blobs
        # skip empty blob (non-conv or non-FC layers)
        if len(fb_layer) == 0:
            continue
        for bn in range(len(fb_layer)):
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

def main():
    fb_model_filename = 'c3d_dextro_benchmark_8conv_finetune_iter_100000'
    dx_model_filename = 'c3d_dextro_benchmark_8conv_finetune_iter_84000.caffemodel'
    converted_dx_model_filename = 'c3d_dextro_benchmark_8conv_finetune_converted2.caffemodel'

    fb_model = fb.NetParameter()
    print "Loading FB"
    with open(fb_model_filename, "rb") as f:
        fb_model.ParseFromString(f.read())
    print "Loaded FB"

    dx_model = dx.NetParameter()

    print "Loading DX"
    with open(dx_model_filename, "rb") as f:
        dx_model.ParseFromString(f.read())
    print "Loaded DX"

    convert_from_dx_to_fb(dx_model, fb_model)

    f = open(converted_dx_model_filename, "wb")
    f.write(dx_model.SerializeToString())
    f.close()

if __name__ == "__main__":
  main()
