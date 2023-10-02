import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
import numpy as np

model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained")

signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav')
#signal, fs = torchaudio.load('/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_7-speakers_SkillsFuture.wav')
print( signal )
print( signal.shape )
#exit( 0 ) 

wavs = signal
if len(wavs.shape) == 1:
    wavs = wavs.unsqueeze(0)
wav_lens = torch.ones(wavs.shape[0], device=model.device)
wavs = wavs.float()
feats = model.mods.compute_features(wavs)
feats = model.mods.mean_var_norm(feats, wav_lens)

# Print output shape
#embeddings = model.encode_batch(signal)
embeddings = model.encode_batch(signal, normalize=False)
#print( embeddings )
#print( embeddings.shape )
#exit( 0 ) 
#prediction = model.classify_batch(signal)
#print(prediction)
#exit(0)


def export():
    # Create dummy input
    # Batch of waveforms [batch, time, channels] or [batch, time]
    # speechbrain/pretrained/interfaces.py:903
    #x = torch.randn( 1, 1920000 )
    x = torch.randn( 1, 200, 80 )
    print( x )
    print( x.shape )
    #exit( 0 ) 

    pretrained_dict = model.state_dict()
    #print( pretrained_dict )
    #exit(0)

    # Show all unconvertable ops, output would be like,
    # {'aten::view_as_real', 'aten::chunk', 'aten::stft'}
    torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
        model, x, opset_version=17
    )
    if( len( unconvertible_ops ) > 0 ):
        print( '------------------------There are some unconvertable operations-----------' )
        print(set(unconvertible_ops))
        exit( 0 )
    print( '---- all operations convertable ----' )

    # Export the model
    #symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    symbolic_names = {0: "batch_size", 1: "max_seq_len", 2: "dim"}
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

def verify():
    # Check exported model see if there is error
    import onnx
    onnx_model = onnx.load("embeddings.onnx")
    onnx.checker.check_model(onnx_model)

    # Compare the output of the original model and the ONNX-converted model to ensure their equivalence.
    import onnxruntime
    rep = onnx.shape_inference.infer_shapes(onnx_model)
    #print(rep)
    providers = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=providers)
    print( ort_session.get_inputs()[0])
    #ort_inputs = {ort_session.get_inputs()[0].name: signal.numpy()}
    ort_inputs = {ort_session.get_inputs()[0].name: feats.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    # Check output is same( or close enough )
    np.testing.assert_allclose(embeddings.detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Original Output:", embeddings)
    print("Onnx model Output:", ort_outs[0])


#export()
verify()
