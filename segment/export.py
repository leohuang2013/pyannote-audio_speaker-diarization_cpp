
import torchaudio
import torch
import numpy as np
from mysegment import MySegment

opsetVer = 17
outModel = 'segment.onnx'

def export():
    # Create dummy input
    #audio = '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav'
    audio = '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_4-speakers_Jennifer_Aniston_and_Adam_Sandler_talk.wav'
    signal, fs = torchaudio.load( audio )
    print( signal.shape )

    self.model =  Model.from_pretrained(
            "pyannote/segmentation@2022.07",
            strict=False,
            use_auth_token=hf_auth_token,
        )
    self.model.eval()
    self.model.to( torch.device( 'cuda' ))
    model = MySegment()

    # Show all unconvertable ops, output would be like,
    #       {'aten::view_as_real', 'aten::chunk', 'aten::stft'}
    #torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
    #    model, signal, opset_version=opsetVer
    #)
    #if( len( unconvertible_ops ) > 0 ):
    #    print( '------------------------There are some unconvertable operations-----------' )
    #    print(set(unconvertible_ops))
    #    exit( 0 )
    #print( '---- all operations convertable ----' )

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
