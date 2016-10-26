#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:58:46 2016

@author: chuck
"""

# check FB model and the converted DX model
# to see where the conversion went wrong

def check_array_identity(a, b, verbose=False):

    import numpy as np

    eps = 0.00001
    a = a.squeeze()
    b = b.squeeze()

    if a.shape != b.shape:
        print "[error] the shapes of two input ndarrays must match!"

    if verbose:
        diff_indices = np.where(abs(a-b) > eps)
        # count consistent / inconsistent elements
        print "[info] #elementes same:{} vs diff:{}".format(
            np.prod(a.shape) - len(diff_indices[0]),
            len(diff_indices[0]),
            )

        # check pattern on inconsistent indices
        if len(diff_indices[0]):
            from collections import Counter
            cnt = []
            print "-"*59
            N_common = 5
            for dim in range(len(diff_indices)):
                cnt.append(Counter(diff_indices[dim]))
                #print "cnt for dim={}: {}".format(dim, cnt[dim])
                print "most common={}".format(cnt[dim].most_common(N_common))
                print "least common={}".format(cnt[dim].most_common()[:-N_common-1:-1])
                print "-"*5

    return (abs(a-b) < eps).all()

def compare_fb_dx():

    import numpy as np
    import cPickle as pickle
    import matplotlib.pyplot as plt
    import collections

    fb_all_pkl = 'fb_all.pkl'
    f = open(fb_all_pkl,"r")
    fb_blobs = pickle.load(f)
    fb_wgts = pickle.load(f)
    fb_bias = pickle.load(f)
    f.close()

    #dx_all_pkl = 'dx_all.pkl'
    dx_all_pkl = 'dx_all_v2.pkl'
    f = open(dx_all_pkl,"r")
    dx_blobs = pickle.load(f)
    dx_wgts = pickle.load(f)
    dx_bias = pickle.load(f)
    f.close()    

    # compare weights and biases
    for k, v in dx_wgts.iteritems():
        print "[info] inspecting filter={}".format(k)
        wgts_same = check_array_identity(dx_wgts[k], fb_wgts[k], verbose=True)
        if wgts_same:
            print "[info] filter weights match!"
        else:
            print "[panic] filter weights don't match!"
        bias_same = check_array_identity(dx_bias[k], fb_bias[k], verbose=True)
        if bias_same:
            print "[info] filter biases match!"
        else:
            print "[panic] filter biases don't match!"

    # compare layer outputs
    for k, v in dx_blobs.iteritems():
        print "[info] inspecting layer output={}".format(k)
        blobs_same = check_array_identity(dx_blobs[k], fb_blobs[k], verbose=True)
        if blobs_same:
            print "[info] layer outputs match!"
        else:
            print "[panic] layer outputs don't match!"

def main():
    #import facebook_caffe_pb2 as fb
    #import dextro_caffe_pb2 as dx
    import numpy as np
    import cPickle as pickle
    import collections
    import sys

    fb_def_filename = 'c3d_dextro_benchmark_train_test_8conv_finetune_deploy_facebook.prototxt'
    fb_wgt_filename = 'c3d_ucf101_finetune_whole_iter_20000'

    # new NCLHW format
    dx_def_filename = 'c3d_dextro_benchmark_train_test_8conv_finetune_deploy_Vision_v2.prototxt'
    converted_dx_wgt_filename = 'c3d_ucf101_finetune_iter_20000_v2.caffemodel'

    #################################################
    # load FB model
    sys.path.insert(0, "/home/chuck/projects/facebook-C3D/python")
    import caffe as fb_caffe
    fb_net = fb_caffe.Net(
        fb_def_filename,
        fb_wgt_filename
        )
    fb_net.set_phase_test()
    fb_net.set_mode_cpu()
    print "-"*79
    print "[info] fb model layers:"
    fb_layers = [(k, v.data.shape) for k, v in fb_net.blobs.items()]
    for k, v in fb_layers:
        print "[info] layer name {}: input shape {}".format(k, v)
    for k, v in fb_net.params.iteritems():
        print "[info] filter name {}: weight shape {}, bias shape {}".format(
            k,
            v[0].data.shape,
            v[1].data.shape,
            )
    print "-"*79

    input_shape = fb_net.blobs['data'].data.shape
    fb_net.blobs['data'].data[...] = np.ones(input_shape)
    fb_net.forward()

    fb_blobs = collections.OrderedDict()
    for k, v in fb_layers:
        fb_blobs[k] = np.array(fb_net.blobs[k].data[:], dtype=float, copy=True)
    fb_wgts = collections.OrderedDict()
    fb_bias = collections.OrderedDict()
    for k, v in fb_net.params.iteritems():
        fb_wgts[k] = np.array(v[0].data[:], dtype=float, copy=True)
        fb_bias[k] = np.array(v[1].data[:], dtype=float, copy=True)

    fb_all_pkl = 'fb_all.pkl'
    f = open(fb_all_pkl, "w")
    pickle.dump(fb_blobs, f)
    pickle.dump(fb_wgts, f)
    pickle.dump(fb_bias, f)
    f.close()

    #################################################
    # load converted DX model
    sys.path.insert(0, "/home/chuck/projects/debug-C3D/chuck-Vision-NCLHW/python")
    import caffe as dx_caffe
    dx_caffe.set_device(3)
    dx_caffe.set_mode_gpu()
    dx_net = dx_caffe.Net(
        dx_def_filename,
        converted_dx_wgt_filename,
        dx_caffe.TEST,
        )
    print "-"*79
    print "[info] dx model layers:"
    dx_layers = [(k, v.data.shape) for k, v in dx_net.blobs.items()]
    for k, v in dx_layers:
        print "[info] layer name {}: input shape {}".format(k, v)
    for k, v in dx_net.params.iteritems():
        print "[info] filter name {}: weight shape {}, bias shape {}".format(
            k,
            v[0].data.shape,
            v[1].data.shape,
            )
    print "-"*79

    input_shape = dx_net.blobs['data'].data.shape
    dx_net.blobs['data'].data[...] = np.ones(input_shape)
    dx_net.forward()

    dx_blobs = collections.OrderedDict()
    for k, v in dx_layers:
        dx_blobs[k] = np.array(dx_net.blobs[k].data[:], dtype=float, copy=True)
    dx_wgts = collections.OrderedDict()
    dx_bias = collections.OrderedDict()
    for k, v in dx_net.params.iteritems():
        dx_wgts[k] = np.array(v[0].data[:], dtype=float, copy=True)
        dx_bias[k] = np.array(v[1].data[:], dtype=float, copy=True)

    #dx_all_pkl = 'dx_all.pkl'
    dx_all_pkl = 'dx_all_v2.pkl'
    f = open(dx_all_pkl, "w")
    pickle.dump(dx_blobs, f)
    pickle.dump(dx_wgts, f)
    pickle.dump(dx_bias, f)
    f.close()

if __name__ == "__main__":
    main()
