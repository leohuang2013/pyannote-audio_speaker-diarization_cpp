
import torchaudio
import torch
import numpy as np
from pyannote.audio.core.model import Model
from hf_token import *
import torchaudio
import onnx
import onnxruntime

opsetVer = 17
outModel = 'segment2.onnx'
batch = 32
channel = 1

def export():
    model =  Model.from_pretrained(
            "pyannote/segmentation@2022.07",
            strict=False,
            use_auth_token=hf_auth_token,
        )
    model.eval()

    # Create dummy input
    dummy_input = torch.zeros(batch, channel, 80000)

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
    #symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    symbolic_names = {0: "B", 1: "C", 2: "T"}
    torch.onnx.export(model,               # model being run
                      dummy_input,                         # model input (or a tuple for multiple inputs)
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

def verify():
    print( '\n\n\n====================verify=======================' )
    model =  Model.from_pretrained(
            "pyannote/segmentation@2022.07",
            strict=False,
            use_auth_token=hf_auth_token,
        )
    model.eval()

    onnx_model = onnx.load( outModel )
    onnx.checker.check_model(onnx_model)
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_4-speakers_Jennifer_Aniston_and_Adam_Sandler_talk.wav',
            ]


    # Load wav
    sequence_len = 80000
    signal, fs = torchaudio.load( wavs[0] )
    assert( signal.shape[1] >= batch * sequence_len )
    signal = signal[:,:batch*sequence_len].flatten().reshape( batch, -1 )
    signal = signal.unsqueeze(1)

    # Calc using pytorch
    #with torch.no_grad():
    segments = model(signal)
    segments1 = segments.detach().numpy()

    # Calc embedding using onnx model
    print( ort_session.get_inputs()[0])
    ort_inputs = {ort_session.get_inputs()[0].name: signal.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_segments = ort_outs[0]

    try:
        # Check if result is close enough
        np.testing.assert_allclose(
                segments1, 
                onnx_segments, 
                rtol=1e-03, 
                atol=1e-05,
                verbose = True)
        print( 'check passed' )
    except AssertionError as e:
        print( e )
        print( onnx_segments[0] )
    print( "-----------------------------" )

def debug():
    '''
    Load model
    check model
    print output shape
    print output 
    print outputs of all layers -- > this is very useful for debugging, 
        - check name against graph in netron.app 
        - python -m pdb export2.py to check each step input/output
    '''
    # Check exported model see if there is error
    # add all intermediate outputs to onnx net ------ 1
    ort_session = onnxruntime.InferenceSession(outModel)
    org_outputs = [x.name for x in ort_session.get_outputs()]

    print( f'\n---- Load model: {outModel} ----' )
    onnx_model = onnx.load( outModel )
    # Check model
    onnx.checker.check_model(onnx_model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(onnx_model, check_type=True)

    # Add all intermediate layer's output to model outputs
    for node in onnx_model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # Load test data
    signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav')
    print( signal.shape )
    # batch size = 2
    signal = signal[:160000]

    model =  Model.from_pretrained(
            "pyannote/segmentation@2022.07",
            strict=False,
            use_auth_token=hf_auth_token,
        )
    model.eval()

    # Calc embeddings using pytorch
    #with torch.no_grad():
    segments = model(signal)


    # Compare the output of the original model and the ONNX-converted model to ensure their equivalence.
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    #so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.log_severity_level = 1 
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)
    print( ort_session.get_inputs()[0])
    signal = signal[None,:]
    ort_inputs = {ort_session.get_inputs()[0].name: signal.numpy()}
    outputs = [x.name for x in ort_session.get_outputs()]
    #ort_inputs = {ort_session.get_inputs()[0].name: feats.numpy()}

    print( f'---- run model ----' )
    #ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_session.run(outputs, ort_inputs)
    onnx_segments = ort_outs[0]
    # ------ 3
    from collections import OrderedDict
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    # Check output is same( or close enough )

    print( '---- print all layers outputs ---- ' )
    for name in outputs:
        print( f'--- out for {name} ---' )
        print( ort_outs[name] )

    print( f'---- check result ----' )
    #fn = torch.from_numpy( ort_outs[0] )
    #out = model.mods.mean_var_norm( fn, wav_lens)
    np.testing.assert_allclose(
            segments.detach().numpy(), 
            onnx_segments, 
            rtol=1e-03, 
            atol=1e-05,
            verbose = True)
    print( f'---- check result is done ----' )

    import sys
    np.set_printoptions(threshold=sys.maxsize)
    '''
    print("Original Output:")
    a1 = feats.detach().numpy().flatten()
    print("Onnx model Output:")
    a2 = ort_outs[0].flatten()
    if len( a1 ) != len( a2 ):
        print( f"Size diff: {len(a1)}, {len(a2)}" )
    for i in range( len( a1 )):
        if abs( a1[i] - a2[i] ) > 1e-03:
            print( f"Diff: {a1[i]}, {a2[i]}" )
            return
    '''


def hints():
    print( '\n\n\n====================note=======================' )
    print( 'If you error like below' )
    print('''
            RuntimeError: cuDNN version incompatibility: PyTorch was compiled  against (8, 7, 0) but found runtime version (8, 6, 0). PyTorch already comes bundled with cuDNN. One option to resolving this error is to ensure PyTorch can find the bundled cuDNN.Looks like your LD_LIBRARY_PATH contains incompatible version of cudnnPlease either remove it from the path or install cudnn (8, 7, 0)
            ''')
    print( 'then execute following command' )
    print( '$> unset LD_LIBRARY_PATH' )
    print( '===============================================\n\n\n' )


hints()
export()
#debug()
verify()
