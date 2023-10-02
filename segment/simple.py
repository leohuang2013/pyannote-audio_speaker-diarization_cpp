import torch.nn as nn
import torch
from typing import Tuple
import numpy as np

class MySimple(nn.Module):
    def __init__(self, device:torch.device = None):
        super(MySimple, self).__init__()

    def calc( self, waveforms:torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        r1 = waveforms.squeeze(dim=1)
        r2 = waveforms.T
        return r1, r2

    def slide(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ws = 16
        step = 4
        d = waveform.unfold( 1, ws, step )
        #i = 0
        #a = []
        #while i + ws < waveform.shape[1]:
        #    a.append( waveform[0][i:i+ws] )
        #    i += step
        #d = torch.stack( tuple( a ))
        #c = np.reshape( n, ( 32, 1, -1 ))
        #d = torch.from_numpy( n ) // failed
        #d = torch.reshape( waveform, ( sample_rate, 1, -1 )) // working
        return d


    def forward( self, waveform:torch.Tensor ):
        res = []
        r1, r2 = self.calc( waveform )
        res.append( r1 )
        res.append( r2 )
        feat = self.slide( waveform, 32 )
        return feat

opsetVer = 17
outModel = 'simple.onnx'

def export():
    # Create dummy input
    signal = torch.randn( 1, 128 ) 
    print( signal.shape )

    model = MySimple()

    # Export the model
    print( '\n---- start export ----' )
    symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    torch.onnx.export(model,               # model being run
                      signal,                         # model input (or a tuple for multiple inputs)
                      outModel,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opsetVer,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      input_names = ['signal'],   # the model's input names
                      output_names = ['segments'], # the model's output names
                      dynamic_axes={'signal' : symbolic_names,    # variable length axes
                                    })
    print( f'---- model has been saved to {outModel} ----\n' )


export()
