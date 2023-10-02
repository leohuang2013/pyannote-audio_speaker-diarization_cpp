# Compile
```
$> mkdir build && cd build
$> cmake -DCMAKE_BUILD_TYPE=DEBUG ..

$> cmake ..
$> make
```
If want build GPU version
```
$> cmake -DGPU=ON ..
$> make
```

# Run
./speakerDiarizer ../model/segment2.onnx ../model/emd4.onnx ../data/multi-speaker_1min.wav

# onnxruntime to GPU
It seems no need change code, instead set cuda when convert to model. For segment.onnx model, change source 
a) add following line to model = ....
model.cuda() 
b) change 
dummy_input = torch.zeros(3, 1, 32000)
-->
dummy_input = torch.zeros(3, 1, 32000).cuda()

change onnx.cmake to download gpu version of onnxruntime

# Verification
Since whole project is to translate pyannote-audio speaker diarization pipleline from python to C++, strategy I adopted here is 
write input/output of each small function in python to txt file, and do same for C++, then load txt file into python to compare 
and check difference. Target is to make each input and output is same.
For this purpose, script/verifyEveryStepResult.py is created.
``` bash
$> python verifyEveryStepResult.py
```
Above command is to compare txt files generated /tmp. and command below is to delete all the txt files.
``` bash
$> python verifyEveryStepResult.py clean
```


# For hierichical clustering
tried following
- hclust-cpp/fastcluster
https://github.com/cdalitz/hclust-cpp
result is wrong, including distance of clusters and result 'fcluster'

- agglomerative-hierarchical-clustering
https://github.com/gyaikhom/agglomerative-hierarchical-clustering/tree/master
centroid_linkage is empty

- alglib
https://www.alglib.net/dataanalysis/clustering.php
does not support centriod
