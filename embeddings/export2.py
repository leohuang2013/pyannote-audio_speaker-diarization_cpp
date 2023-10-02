import torchaudio
import onnx
import onnxruntime
from myembedding import MyEmbedding
import torch
import numpy as np
import torch.onnx.verification
from speechbrain.pretrained import EncoderClassifier


outModel = 'embeddings2.onnx'
opsetVer = 17


def simple():
    model = MyEmbedding.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")

    signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav')
    #signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav')
    #signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_7-speakers_SkillsFuture.wav')
    print( signal )
    print( signal.shape )
    #exit( 0 ) 

    wavs = signal
    if len(wavs.shape) == 1:
        wavs = wavs.unsqueeze(0)
    wav_lens = torch.ones(wavs.shape[0], device=model.device)
    wavs, wav_lens = wavs.to(model.device), wav_lens.to(model.device)
    wavs = wavs.float()
    feats = model.mods.compute_features(wavs)
    print( feats )
    #f = open( '/tmp/feats1.txt', 'w' )
    #df = feats.numpy().flatten()
    #for a in df:
    #    f.write( str( a ) + ',' )
    #f.close()
    feats = model.mods.mean_var_norm(feats, wav_lens)
    #exit( 0 ) 

    # Print output shape
    embeddings = model.encode_batch(signal, normalize=False)
    #print( embeddings )
    #print( embeddings.shape )
    #exit( 0 ) 
    #prediction = model.classify_batch(signal)
    #print(prediction)
    #exit(0)

def export( ):
    # Create dummy input
    # Batch of waveforms [batch, time, channels] or [batch, time]
    # speechbrain/pretrained/interfaces.py:903
    batch = 32
    #batch = 1
    max_sequence = 80000
    #x = torch.FloatTensor(batch, max_sequence).uniform_(-32768, 32767)
    x = torch.FloatTensor(batch, max_sequence).uniform_(-1, 1)
    print( x.shape )
    #exit( 0 ) 

    model = MyEmbedding.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    pretrained_dict = model.state_dict()
    #print( pretrained_dict )
    #exit(0)

    # TODO: test if eval() will solve precision problem
    # another potential cause is, one is on cpu, another is on cuda
    #model.eval()

    # Show all unconvertable ops, output would be like,
    # {'aten::view_as_real', 'aten::chunk', 'aten::stft'}
    torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
        model, 
        x, 
        opset_version=opsetVer
    )
    if( len( unconvertible_ops ) > 0 ):
        print( '------------------------There are some unconvertable operations-----------' )
        print(set(unconvertible_ops))
        exit( 0 )
    print( '\n---- all operations convertable ----\n' )

    #print( '\n---- Check model mismatch ----' )
    #torch.onnx.verification.find_mismatch( model,
    #        x,
    #        opset_version=opsetVer,
    #        do_constant_folding=True,
    #        verbose=True
    #        )
    #print( '---- Mismatch check is done ----\n' )

    signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav')
    print( '\n---- start export ----' )
    # Export the model
    '''
    There will be big difference between using signal and x for result of mean_var_norm
    '''
    #model_script = torch.jit.script( model )
    wav_lens = torch.Tensor([1.0])
    wav_lens1 = [1.0] * batch
    wav_lens1 = torch.Tensor( wav_lens1 )
    wav_lens1 = wav_lens1.to(torch.float32)
    torch.onnx.export( model,               # model being run
                      #(signal, wav_lens),                          # model input (or a tuple for multiple inputs)
                      (x, wav_lens1),                          # model input (or a tuple for multiple inputs)
                      outModel,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opsetVer,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      input_names = ['signal', 'wav_lens'],   # the model's input names
                      output_names = ['embeddings'], # the model's output names
                      dynamic_axes={
                          'signal': {0: "batch_size", 1: "max_seq_len"},    
                          'wav_lens': {0: "batch_size" }
                                    })
    print( f'---- model has been saved to {outModel} ----\n' )

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

    # ----- 2
    for node in onnx_model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # Load test data
    #signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav')
    signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/one_min16k.wav')
    print( signal.shape )
    #model = MyEmbedding.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")

    # Calc embeddings using pytorch
    wavs = signal
    if len(wavs.shape) == 1:
        wavs = wavs.unsqueeze(0)
    wav_lens = torch.ones(wavs.shape[0], device=model.device)
    wavs, wav_lens = wavs.to(model.device), wav_lens.to(model.device)
    wavs = wavs.float()
    feats = model.mods.compute_features(wavs)
    feats = model.mods.mean_var_norm(feats, wav_lens)
    d1 = signal.numpy()
    embeddings = model.encode_batch(signal, normalize=False)
    d2 = signal.numpy()
    # make sure above function call, signal not changed
    np.testing.assert_allclose(
            d1, 
            d2, 
            rtol=1e-03, 
            atol=1e-05,
            verbose = True)
    print( '\nSignal data is not changed\n' )

    # Compare the output of the original model and the ONNX-converted model to ensure their equivalence.
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    #so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.log_severity_level = 1 
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)
    print( ort_session.get_inputs()[0])
    ort_inputs = {ort_session.get_inputs()[0].name: signal.numpy()}
    outputs = [x.name for x in ort_session.get_outputs()]
    #ort_inputs = {ort_session.get_inputs()[0].name: feats.numpy()}

    print( f'---- run model ----' )
    #ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_session.run(outputs, ort_inputs)
    onnx_embeddings = ort_outs[0]
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
            embeddings.detach().numpy(), 
            #feats.detach().numpy(), 
            onnx_embeddings, 
            rtol=1e-03, 
            atol=1e-05,
            verbose = True)
    #np.testing.assert_allclose(
    #        feats.detach().numpy(), 
    #        ort_outs[0], 
    #        rtol=1e-03, 
    #        atol=1e-05,
    #        verbose=True)
    print( f'---- check result is done ----' )

    import sys
    np.set_printoptions(threshold=sys.maxsize)
    print("Original Output:")
    #print(feats.detach().numpy())
    a1 = feats.detach().numpy().flatten()
    print("Onnx model Output:")
    #print(ort_outs[0])
    a2 = ort_outs[0].flatten()
    if len( a1 ) != len( a2 ):
        print( f"Size diff: {len(a1)}, {len(a2)}" )
    for i in range( len( a1 )):
        if abs( a1[i] - a2[i] ) > 1e-03:
            print( f"Diff: {a1[i]}, {a2[i]}" )
            return


def verify():
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    #model = MyEmbedding.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
    #        savedir="pretrained")
    #model.eval()
    onnx_model = onnx.load( outModel )
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_4-speakers_Jennifer_Aniston_and_Adam_Sandler_talk.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]

    batch = 32
    for wav in wavs:
        print( f"Check wav file: {wav}" )

        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:,:batch*80000].flatten().reshape( batch, -1 )

        # Calc embeddings using pytorch
        print( signal.shape )
        wav_lens = torch.Tensor([1.0] * batch)
        embeddings = model.encode_batch(signal, wav_lens, normalize=False)

        # Calc embedding using onnx model
        print( ort_session.get_inputs()[0])
        print( ort_session.get_inputs()[1])
        ort_inputs = {ort_session.get_inputs()[0].name: signal.numpy(),
                ort_session.get_inputs()[1].name: wav_lens.numpy()
                }
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_embeddings = ort_outs[0]

        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    embeddings.detach().numpy(), 
                    onnx_embeddings, 
                    rtol=1e-03, 
                    atol=1e-05,
                    verbose = True)
            print( 'check passed' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------" )

def test():
    onnx_model = onnx.load( outModel )
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    # Load wav
    with open( '/tmp/cpp_signals0.txt' ) as f:
        lines = f.readlines()
    signals = []
    for line in lines:
        arr = line.split(',')
        arr = arr[:-1]
        assert( len( arr ) == 80000 )
        signal = [float(a) for a in arr ]
        signals.append( signal )
    signals = torch.Tensor( signals )

    with open( '/tmp/cpp_final_wav_lens0.txt' ) as f:
        content = f.read()
    arr = content.split(',')
    arr = arr[:-1]
    assert( len( arr ) == signals.shape[0])
    wav_lens = [float(a) for a in arr ]
    # Calc embedding using onnx model
    print( ort_session.get_inputs()[0])
    print( ort_session.get_inputs()[1])
    wav_lens = torch.Tensor( wav_lens )
    ort_inputs = {ort_session.get_inputs()[0].name: signals.numpy(),
            ort_session.get_inputs()[1].name: wav_lens.numpy()
            }
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_embeddings = ort_outs[0]
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    print( onnx_embeddings )


export()
#debug()
verify()
#test()
#simple()


