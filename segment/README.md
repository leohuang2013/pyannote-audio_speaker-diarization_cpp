
try to export whole process as module.nn to onnx model, but failed. 

# problem
1. like numpy.unfold function, it is considered as constant when pytorch.export to onnx model, which demonstrated
in simply.py as simplest demo.

solution is custom operator of onnx,
https://github.com/onnx/tutorials/blob/main/PyTorchCustomOperator/README.md

implement operator in python extension when export, and implement operator in c++ side when doing infer.

However, this is not only one that need to implemented, there gonna be a lot in export.py

Therefore, stop here, but leave code, work here. So that if we want to continue its work. 

Current solution is only export model partially, then implement remaining in c++. See cpp folder.

# export
'''
$> conda activate sd_embeddings
$> python export2.py
'''


