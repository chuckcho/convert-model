#!/usr/bin/env python

import facebook_caffe_pb2 as fb
import dextro_caffe_pb2 as dx

fb_model = fb.NetParameter()
print "Loading FB"
with open("facebook_caffe.model", "rb") as f:
    fb_model.ParseFromString(f.read())
print "Loaded FB"

dx_model = dx.NetParameter()

print "Loading DX"
with open("dextro_caffe.model", "rb") as f:
    dx_model.ParseFromString(f.read())
print "Loaded DX"
