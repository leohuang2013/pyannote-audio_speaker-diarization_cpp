import torchaudio
import onnx
import onnxruntime
from threeModel import FeatureModel, EMDModel, FBankModel, MySTFT, MyEmbedding0
import torch
import numpy as np
import torch.onnx.verification
from speechbrain.pretrained import EncoderClassifier
from speechbrain.processing.features import STFT, Filterbank, spectral_magnitude


opsetVer = 17


def export( outModel ):
    # Create dummy input
    # Batch of waveforms [batch, time, channels] or [batch, time]
    # speechbrain/pretrained/interfaces.py:903
    #batch = 32
    batch = 1
    max_sequence = 80000
    #x = torch.FloatTensor(batch, max_sequence).uniform_(-32768, 32767)
    x = torch.FloatTensor(batch, max_sequence).uniform_(-1, 1)
    print( x.shape )
    #exit( 0 ) 

    model = FeatureModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
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
    torch.onnx.export( model,               # model being run
                      signal,                          # model input (or a tuple for multiple inputs)
                      #x,                          # model input (or a tuple for multiple inputs)
                      outModel,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opsetVer,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      input_names = ['signal'],   # the model's input names
                      output_names = ['features'], # the model's output names
                      dynamic_axes={
                          'signal': {0: "batch_size", 1: "max_seq_len"},    
                                    })
    print( f'---- model has been saved to {outModel} ----\n' )

def export1( outModel ):
    # Create dummy input
    #batch = 32
    batch = 1
    max_sequence = 80
    #x = torch.FloatTensor(batch, max_sequence).uniform_(-32768, 32767)
    x = torch.FloatTensor(batch, 6000, max_sequence).uniform_(-1, 1)
    print( x.shape )
    #exit( 0 ) 

    model = EMDModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    pretrained_dict = model.state_dict()
    #print( pretrained_dict )
    #exit(0)
    wav_lens = torch.Tensor([1.0])
    wav_lens1 = [1.0] * batch
    wav_lens1 = torch.Tensor( wav_lens1 )
    wav_lens1 = wav_lens1.to(torch.float32)

    # TODO: test if eval() will solve precision problem
    # another potential cause is, one is on cpu, another is on cuda
    #model.eval()

    # Show all unconvertable ops, output would be like,
    # {'aten::view_as_real', 'aten::chunk', 'aten::stft'}
    torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
        model, 
        (x, wav_lens1), 
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
    torch.onnx.export( model,               # model being run
                      #(signal, wav_lens),                          # model input (or a tuple for multiple inputs)
                      (x, wav_lens1),                          # model input (or a tuple for multiple inputs)
                      outModel,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opsetVer,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      input_names = ['signal'],   # the model's input names
                      output_names = ['features'], # the model's output names
                      dynamic_axes={
                          'signal': {0: "batch_size", 1: "max_seq_len"},    
                                    })
    print( f'---- model has been saved to {outModel} ----\n' )

def export2( outModel ):
    # Create dummy input
    batch = 32
    x = torch.FloatTensor(batch, 501, 201, 2).uniform_(-1, 1)
    print( x.shape )
    #exit( 0 ) 

    model = MyEmbedding0.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    wav_lens1 = [1.0] * batch
    wav_lens1 = torch.Tensor( wav_lens1 )
    wav_lens1 = wav_lens1.to(torch.float32)

    #print( '\n---- Check model mismatch ----' )
    #torch.onnx.verification.find_mismatch( model,
    #        x,
    #        opset_version=opsetVer,
    #        do_constant_folding=True,
    #        verbose=True
    #        )
    #print( '---- Mismatch check is done ----\n' )

    print( '\n---- start export ----' )

    # Export the model
    torch.onnx.export( model,               # model being run
                      (x, wav_lens1),                          # model input (or a tuple for multiple inputs)
                      outModel,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opsetVer,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      input_names = ['feats', 'wav_lens'],   # the model's input names
                      output_names = ['embedding'], # the model's output names
                      dynamic_axes={
                          'feats': {0: "B", 1: "T", 2: "N", 3: "M"},    
                          'wav_lens': {0: "B" }
                                    })
    print( f'---- model has been saved to {outModel} ----\n' )

def export_stft( outModel ):
    #batch = 32
    batch = 1
    max_sequence = 80000
    x = torch.FloatTensor(batch, max_sequence).uniform_(-1, 1)
    print( x.shape )

    model = MySTFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400) 
    #model.eval()

    #print( '\n---- Check model mismatch ----' )
    #torch.onnx.verification.find_mismatch( model,
    #        x,
    #        opset_version=opsetVer,
    #        do_constant_folding=True,
    #        verbose=True
    #        )
    #print( '---- Mismatch check is done ----\n' )

    print( '\n---- start export ----' )
    torch.onnx.export( model,               # model being run
                      x,                          # model input (or a tuple for multiple inputs)
                      outModel,   # where to save the model (can be a file or file-like object)
                      #export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opsetVer,          # the ONNX version to export the model to
                      #do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      input_names = ['signal'],   # the model's input names
                      output_names = ['features'], # the model's output names
                      dynamic_axes={
                          'signal': {0: "batch_size", 1: "max_seq_len"},    
                                    })
    print( f'---- model has been saved to {outModel} ----\n' )

def export_fbank( outModel ):
    #batch = 32
    batch = 1
    x = torch.FloatTensor(batch, 501, 201, 2).uniform_(-1, 1)
    print( x.shape )

    model = FBankModel() 

    #print( '\n---- Check model mismatch ----' )
    #torch.onnx.verification.find_mismatch( model,
    #        x,
    #        opset_version=opsetVer,
    #        do_constant_folding=True,
    #        verbose=True
    #        )
    #print( '---- Mismatch check is done ----\n' )

    print( '\n---- start export ----' )
    torch.onnx.export( model,               # model being run
                      x,                          # model input (or a tuple for multiple inputs)
                      outModel,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opsetVer,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      input_names = ['signal'],   # the model's input names
                      output_names = ['features'], # the model's output names
                      dynamic_axes={
                          'signal': {0: "B", 1: "T", 2: "N", 3: "M"},    
                                    })
    print( f'---- model has been saved to {outModel} ----\n' )

def verify_fbank( outModel ):
    model = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)

    onnx_model = onnx.load( outModel )
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]

    for wav in wavs:
        print( f"Check wav file: {wav}" )

        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:,:80000]

        # Calc embeddings using pytorch
        print( signal.shape )
        compute_STFT = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
        feats_stft = compute_STFT(signal)
        features = spectral_magnitude(feats_stft)
        compute_fbanks = Filterbank(n_mels=80)
        feats = compute_fbanks(features)

        # Calc embedding using onnx model
        print( ort_session.get_inputs()[0])
        ort_inputs = {ort_session.get_inputs()[0].name: feats_stft.numpy(),
                }
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_feats = ort_outs[0]

        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    feats.detach().numpy(), 
                    onnx_feats, 
                    rtol=1e-03, 
                    atol=1e-05,
                    verbose = True)
            print( 'check passed' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------" )

def debug( outModel ):
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
    signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav')
    print( signal.shape )
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
    #feats = model.mods.mean_var_norm(feats, wav_lens)
    #embeddings = model.encode_batch(signal, normalize=False)

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
    onnx_feats = ort_outs[0]
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
            feats.detach().numpy(), 
            #feats.detach().numpy(), 
            onnx_feats, 
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


def verify( outModel ):
    #model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
    #        savedir="pretrained")
    model = FeatureModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    onnx_model = onnx.load( outModel )
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]

    visualisation = {}
    def hook_fn(m, i, o):
        print( f"hook for: {m}" )
        visualisation[m] = o

    def get_all_layers(net):
        for name, layer in net._modules.items():
            #If it is a sequential, don't register a hook on it
            # but recursively register hook on all it's module children
            if isinstance(layer, torch.nn.Sequential):
                get_all_layers(layer)
            else:
                # it's a non sequential. Register a hook
                layer.register_forward_hook(hook_fn)

    get_all_layers(model.mods.compute_features)

    for wav in wavs:
        print( f"Check wav file: {wav}" )

        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:,:80000]

        # Calc embeddings using pytorch
        print( signal.shape )
        wav_lens = torch.Tensor([1.0])
        feats = model.encode_batch(signal)

        print( visualisation.keys()  )

        # Calc embedding using onnx model
        print( ort_session.get_inputs()[0])
        ort_inputs = {ort_session.get_inputs()[0].name: signal.numpy(),
                }
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_feats = ort_outs[0]

        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    feats.detach().numpy(), 
                    onnx_feats, 
                    rtol=1e-03, 
                    atol=1e-05,
                    verbose = True)
            print( 'check passed' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------" )

def genFeats():
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_1min.wav',
            ]

    n = 0
    for wav in wavs:
        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:,:80000]

        # Calc embeddings using pytorch
        print( signal.shape )
        feats = model.mods.compute_features(signal)
        emd = feats.detach().numpy()
        emd = emd.flatten()
        # [1, 6001, 80]
        np.savetxt(f'feats{n}.txt', emd, fmt='%f')
        n += 1

def verify1( outModel ):
    #model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
    #        savedir="pretrained")
    model = EMDModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    onnx_model = onnx.load( outModel )
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    featsN = ['feats0.txt', 'feats1.txt', 'feats2.txt', 'feats3.txt']


    batch = 1
    wav_lens1 = [1.0] * batch
    wav_lens1 = torch.Tensor( wav_lens1 )
    wav_lens1 = wav_lens1.to(torch.float32)
    for f in featsN:
        print( "\n-----------------------------" )
        feats = np.loadtxt( f, dtype=float)
        feats = feats.reshape( 1, -1, 80 )
        feats = torch.Tensor( feats )

        # Calc embeddings using pytorch
        print( feats.shape )
        wav_lens = torch.Tensor([1.0])
        embeddings = model.encode_batch( feats )

        # Calc embedding using onnx model
        print( ort_session.get_inputs()[0])
        print( ort_session.get_inputs()[1])
        ort_inputs = {ort_session.get_inputs()[0].name: feats.numpy(),
                ort_session.get_inputs()[1].name: wav_lens1.numpy()
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
            print( f'check passed for {f}' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------\n" )

def verify2( outModel ):
    model_origin = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model_origin.eval()
    model = MyEmbedding0.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    onnx_model = onnx.load( outModel )
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_4-speakers_Jennifer_Aniston_and_Adam_Sandler_talk.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_1min.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]


    batch = 32
    wav_lens1 = [1.0] * batch
    wav_lens1 = torch.Tensor( wav_lens1 )
    wav_lens1 = wav_lens1.to(torch.float32)
    for wav in wavs:
        print( "\n-----------------------------" )
        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:,:batch*80000].flatten().reshape( batch, -1 )

        # Calc embeddings using pytorch
        print( signal.shape )
        compute_STFT = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
        feats_stft = compute_STFT(signal)
        np.savetxt(f's4_feats.txt', feats_stft.flatten(), fmt='%f', delimiter=',')

        # Calc embeddings using pytorch
        print( feats_stft.shape )
        #wav_lens = torch.Tensor([1.0])
        embeddings_origin = model_origin.encode_batch(signal, normalize=False)
        np.savetxt(f's4_emd.txt', embeddings_origin.flatten(), fmt='%f', delimiter=',')
        print( embeddings_origin  )
        embeddings = model.encode_batch( feats_stft, wav_lens1 )

        # Calc embedding using onnx model
        print( ort_session.get_inputs()[0])
        print( ort_session.get_inputs()[1])
        ort_inputs = {ort_session.get_inputs()[0].name: feats_stft.numpy(),
                ort_session.get_inputs()[1].name: wav_lens1.numpy()
                }
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_embeddings = ort_outs[0]
        print(onnx_embeddings)

        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    embeddings.detach().numpy(), 
                    onnx_embeddings, 
                    rtol=1e-03, 
                    atol=1e-05,
                    verbose = True)
            print( f'check passed for {wav}' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------\n" )

def test():
    from speechbrain.processing.features import STFT
    model = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)
    #model = MySTFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_1min.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            #'/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]

    i = 0
    for wav in wavs:
        # Load wav
        signal, fs = torchaudio.load( wav )
        #signal = signal[:,:80000]
        signal = signal[:,:400]
        #signal = signal[:,:800]
        #signal = signal[:,:320000]
        #signal = signal.flatten().reshape( 4, -1 )
        print( signal.shape )
        feats = model(signal)
        print('--------------------')
        print( feats.shape )
        #print(feats)
        np.savetxt(f'/tmp/stft_wav{i}.txt', signal.detach().numpy().flatten(), fmt='%f')
        np.savetxt(f'/tmp/stft_feats{i}.txt', feats.detach().numpy().flatten(), fmt='%f')
        i += 1

def manualFbank():
    import kaldi_native_fbank as knf
    model = FeatureModel.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]

    for wav in wavs:
        print( f"Check wav file: {wav}" )

        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:,:80000]

        # Calc embeddings using pytorch
        print( signal.shape )
        wav_lens = torch.Tensor([1.0])
        feats = model.encode_batch(signal)
        feats.detach().numpy(), 

        # Calc using fbank 
        sampling_rate = 16000
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.remove_dc_offset = False
        opts.frame_opts.snip_edges = False
        opts.mel_opts.num_bins = 80
        opts.mel_opts.debug_mel = False
        opts.mel_opts.low_freq = 0
        opts.mel_opts.high_freq = 8000
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(sampling_rate, signal[0].tolist())
        ffeats = []
        for i in range(fbank.num_frames_ready):
            f = torch.from_numpy(fbank.get_frame(i))
            ffeats.append( f )


        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    feats[0][0], 
                    ffeats[0], 
                    rtol=1e-03, 
                    atol=1e-05,
                    verbose = True)
            print( 'check passed' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------" )

def verify_stft( outModel ):
    from speechbrain.processing.features import STFT
    model = STFT(sample_rate=16000, win_length=25, hop_length=10, n_fft=400)

    onnx_model = onnx.load( outModel )
    providers = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), so, providers=providers)

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]

    for wav in wavs:
        print( f"Check wav file: {wav}" )

        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:,:80000]

        # Calc embeddings using pytorch
        print( signal.shape )
        feats = model(signal)

        # Calc embedding using onnx model
        print( ort_session.get_inputs()[0])
        ort_inputs = {ort_session.get_inputs()[0].name: signal.numpy(),
                }
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_feats = ort_outs[0]

        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    feats.detach().numpy(), 
                    onnx_feats, 
                    rtol=1e-03, 
                    atol=1e-05,
                    verbose = True)
            print( 'check passed' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------" )

outModel = 'feature3.onnx'
#export(outModel )
#debug(outModel )
#verify(outModel )

'''
input: features 
output: embedding
This model works
'''
outModel = 'emd3.onnx'
#export1(outModel )
#verify1(outModel )

'''
input: output of stft 
output: embedding
This model works
'''
outModel = 'emd4.onnx'
export2(outModel )
verify2(outModel )

'''
This model does not work
'''
outModel = 'stft.onnx'
#export_stft( outModel )
#verify_stft( outModel )

'''
This model works
'''
outModel = 'fbank.onnx'
#export_fbank( outModel )
#verify_fbank( outModel )

# Calc fbank features using kaldi-native-fbank
# https://github.com/csukuangfj/kaldi-native-fbank
#manualFbank()

#test()
#simple()

#genFeats()
