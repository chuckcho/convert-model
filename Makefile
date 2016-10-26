default:
	protoc --python_out=$(PWD) dextro_caffe.proto
	protoc --python_out=$(PWD) facebook_caffe.proto
