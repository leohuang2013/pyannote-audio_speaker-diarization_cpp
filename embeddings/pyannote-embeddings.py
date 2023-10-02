# 1. visit hf.co/pyannote/embedding and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model
import torch

hf_auth_token="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"
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=hf_auth_token)

from pyannote.audio import Inference

signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav')
print( signal )
print( signal.shape )
#exit( 0 ) 

inference = Inference(model, window="whole")
embedding1 = inference("/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav")
# `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

# `distance` is a `float` describing how dissimilar speakers 1 and 2 are.

symbolic_names = {0: "batch_size", 1: "max_seq_len"}
x = torch.randn( 1, 1920000 )
print( x )
print( x.shape )
#exit( 0 ) 
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "embeddings.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  verbose=False,
                  input_names = ['signal'],   # the model's input names
                  output_names = ['embeddings'], # the model's output names
                  dynamic_axes={'signal' : symbolic_names,    # variable length axes
                                })
