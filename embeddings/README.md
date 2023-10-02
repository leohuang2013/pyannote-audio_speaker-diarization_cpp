

# Steps to export onnx model
1. activate env
```
$> conda activate sd_embeddings
```
2. export 
```
$> python export2.py
```

Due to aten::stft issue, use develop branch,
with system wide python(3.10), it will encounter error: some assert error, so we use 3.8
```
conda create --prefix=./embeddings python=3.8
- or
conda create -n embeddings python=3.8
conda env list
conda activate /home/leo/storage/sharedFolderVirtualbox/experiment/speaker_diarization-embeddings/embeddings
```

- install pyannote-audio, 

- uninstall speechbarin, 
when install pyannote-audio, it will install speechbrain. see reason below to learn why.

- install speechbrain==0.5.14, 

- uninstall torch, torchaudio, torchvision, 
when install pyannot-audio and speechbrain, it will install torch, torchaudio and torchvision.

- install preview version torch, torchaudio, torchvision 

Then install preview version torch, only which provides stft
```
$> pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu117
```
If got an error, may change /cu117 to /cu118

Check installed version of torch
```
$> import torch
$> torch.version.__version__
```

and another issue is aten::view_as_real is not supported yet
https://github.com/pytorch/pytorch/issues/49793
https://github.com/pytorch/pytorch/pull/92087
https://pytorch.org/docs/stable/onnx_supported_aten_ops.html


# Speechbrain version
Use 0.5.14, dont use 0.5.15 because it requires additional function view_as_real which is not supported pytorch/onnx
Error:
------------------------There are some unconvertable operations-----------
{'aten::view_as_real'}
https://github.com/onnx/onnx/issues/4785

# How to extract output of each layer
https://github.com/microsoft/onnxruntime/issues/1455

# map between pytorch op and onnx op
understand this is to map graph(onnx op) visulized in netron.app to source code(pytorch op) in python

https://zhuanlan.zhihu.com/p/422290231

---------------------------------------------------------------------------------
1. Wavs check is done: wavs same
2. feats check is done: some minior differnce, like 3rd number after decimal. e.g. -19.901580810546875, -19.90271759033203
    Mismatched elements: 3444 / 960080 (0.359%)
    Max absolute difference: 0.11842346
    Max relative difference: 14.923263
   This could be caused by 
   Differences in Operator Implementations
   -------------------------
   Due to differences in implementations of operators, running the exported model on different runtimes 
   may produce different results from each other or from PyTorch. Normally these differences are numerically small,
   so this should only be a concern if your application is sensitive to these small differences. 
   ------------------------
   above text from https://pytorch.org/docs/stable/onnx.html
   In addition, if input dimension is different, exported model is different. For example, input is [1,
   192000], while test input [1, 92000], then exported onnx model is different although model file
   size is exactly same. To check this, run command,
   $> md5file embeddings2.onnx
   to show diff
   $> vimdiff embeddings2.onnx embeddings2_2mins.onnx
   this is because torch.onnx.export default use model with trace, then
   generate static graph, which is static graph, all if/else and loops in script will be executed
   based the input data, more specificly, input decide code goes if or else, and how many loops will
   be executed. Another is method is script, that is pass model_script = torch.jit.script( model )
   into torch.onnx.torch, but current implemenation of speechbrain does not support it. To make
   speechbain to support script method, 

   I tried to fix errors when torch.onnx.export( model_script,
   ... but after having fixed 3 errors, and they seem more coming, so I stopped.

   Due to above exportede model difference, when export with N length signal, then test with M length signal, 
   will get totally different result.
   ------------------------------------------
        Mismatched elements: 192 / 192 (100%)
        Max absolute difference: 23.246006
        Max relative difference: 49.524498
         x: array([[[ 1.286359e+01,  1.619633e+01,  2.043188e+01, -7.442868e+00,
                  1.213802e+01, -2.301076e+01, -6.661740e+01, -5.565681e+00,
                  2.544946e+01, -2.250953e+01,  1.666232e+01,  3.077267e+00,...
         y: array([[[  9.896905,   6.993248,  16.629763, -18.276844,  14.344133,
                 -21.955336, -60.730415,  -4.880554,  38.35284 , -16.19262 ,
                  22.38634 ,   1.895411,  14.681747, -19.542442,  20.844149,...

    Conclusion here is that, mean_var_norm module leads to discrepancy.
    Above discrepancy caused by wrong conversion from pytorch to onnx. More specificly, in 
    def _compute_current_stats(self, x), 
    current_mean = torch.mean(x, dim=0).detach().data is converted as constant, that means 
    current_mean is constant, in above result of 'vimdiff embeddings2.onnx embeddings2_2mins.onnx'
    the difference shown is is constant, because signal data passed in torch.onnx.export different,
    therefore, this constant is different. Okay, solution is avoid let conversion consider this is
    variable not constant. In graph in netron.app, operate is Sub, its input B, named  /Constant_1_output_0
    click plus sign, can see it is constant value. This issue caused by it uses 'data' of tensor
    which should be avoided ilustrated as 
    https://pytorch.org/docs/stable/onnx.html ---> search 'Avoid Tensor.data'
    also see this,
    https://zhuanlan.zhihu.com/p/422290231
    Another issue is error when load onnx model:
    FAIL : Load model from embeddings2.onnx failed:Type Error: Type parameter (Tind) of Optype (Slice) bound to different types (tensor(int64) and tensor(int32) in node (/Slice)
    This is caused wrong type casted to for actual_size
        actual_size = torch.round(lengths[snt_id] * x.shape[1]).int()
    should changed to,
        actual_size = torch.round(lengths[snt_id] * x.shape[1]).long()

    Summary of changes,
    - use our own EncoderClassifier --> MyEmbedding.py
    - use our own Nomalization --> MyNormalization, simplized version of original one
    - dont use data 
    - change cast from int to long 

3. mean_var_norm check is done: result is totally different. see difference betwen using x and
signal as input
4. embedding check is done: if give same feats, then output same embeddings
