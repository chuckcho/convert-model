#!/usr/bin/env python

#import sys
#sys.path.append("..")
import facebook_caffe_pb2 as fb
import dextro_caffe_pb2 as dx
import numpy as np

def convert_from_dx_to_fb(dx, fb, last_layer_name=None, simulate=False):

    if simulate:
        print "--------------------"
        print " Simulation only!!!"
        print "--------------------"
    print type(dx)
    print type(fb)

    # by default, it processes through the end of the layers
    last_layer_idx = len(fb.layers)

    # if specified, find layer index corresponding to the given last_layer_name
    # for fb model
    if last_layer_name:
        for n in range(len(fb.layers)):
            if fb.layers[n].name == last_layer_name:
                last_layer_idx = n+1
                break

    for n in range(last_layer_idx):
        dx_layer = dx.layer[n].blobs
        fb_layer = fb.layers[n].blobs
        if len(dx_layer) != len(fb_layer):
            print "PANIC", n, len(dx_layer), len(fb_layer)
            return
        for bn in range(len(dx_layer)):
            dx_blob = dx_layer[bn]
            fb_blob = fb_layer[bn]
            if len(dx_blob.data) != len(fb_blob.data):
                print dx.layer[n].name, fb.layers[n].name
                print "PANIC", n, bn, len(dx_blob.data), len(fb_blob.data)
                return
            # more elaborate check
            dx_ndim = np.prod(dx_blob.shape.dim)
            print "dx_blob.shape.dim={} (#={})".format(
                dx_blob.shape.dim, dx_ndim)
            fb_ndim = fb_blob.num * fb_blob.channels * fb_blob.length * \
                fb_blob.height * fb_blob.width
            print "fb_blob.(n,c,l,h,w)=({},{},{},{},{}) (#={})".format(
                fb_blob.num,
                fb_blob.channels,
                fb_blob.length,
                fb_blob.height,
                fb_blob.width,
                fb_ndim)
            if dx_ndim != fb_ndim:
                print "PANIC", dx_ndim, fb_ndim
            if dx_ndim != len(dx_blob.data):
                print "PANIC", dx_ndim, len(dx_blob.data)
            print "COPYING layer index-{}-{}, data_len={}: {} -> {}".format(
                n, bn, len(dx_blob.data),
                dx.layer[n].name, fb.layers[n].name)
            if not simulate:
                # copy data
                for i in range(0, len(dx_blob.data)):
                    dx_blob.data[i] = fb_blob.data[i]
            print "COPIED"

def main():

    # old NCHWL format
    #dx_model_filename = 'dx_empty_8conv_model.caffemodel'

    # new NCLHW format
    #dx_model_filename = '/home/chuck/projects/debug-C3D/chuck-Vision-NCLHW/examples/c3d_ucf101/TEMP_c3d_ucf101_create_8conv_model_iter_1.caffemodel'
    #dx_model_filename = '/home/chuck/projects/debug-C3D/chuck-Vision-NCLHW/examples/c3d_ucf101/TEMP_c3d_ucf101_create_8conv_model_v2_iter_1.caffemodel'
    dx_model_filename = '/home/chuck/projects/Vision/examples/c3d_ucf101/TEMP_c3d_ucf101_create_8conv_model_v2_iter_20.caffemodel'
    #dx_model_filename = '/home/chuck/projects/Vision/examples/c3d_dextro_benchmark/TEMP_c3d_ucf101_create_8conv_model_v2_iter_10.caffemodel'
    fb_model_filename = 'c3d_ucf101_finetune_whole_iter_20000'
    #fb_model_filename = '/home/chuck/projects/facebook-C3D/examples/c3d_dextro_benchmark/c3d_dextro_benchmark_8conv_finetune_iter_100000'
    #converted_dx_model_filename = './c3d_ucf101_finetune_iter_100000.caffemodel'
    converted_dx_model_filename = './c3d_ucf101_finetune_iter_20000_v2.caffemodel'

    dx_model = dx.NetParameter()
    print "Loading DX"
    with open(dx_model_filename, "rb") as f:
        dx_model.ParseFromString(f.read())
    print "Loaded DX"

    fb_model = fb.NetParameter()
    print "Loading FB"
    with open(fb_model_filename, "rb") as f:
        fb_model.ParseFromString(f.read())
    print "Loaded FB"

    #last_layer_name = 'fc7'
    convert_from_dx_to_fb(dx_model, fb_model,
    #                      last_layer_name=last_layer_name,
    #                      simulate=False
                          )
    #convert_from_dx_to_fb(dx_model, fb_model)

    f = open(converted_dx_model_filename, "wb")
    f.write(dx_model.SerializeToString())
    f.close()

if __name__ == "__main__":
  main()
