import torch
import torchaudio
from myembedding import MyEmbedding
from speechbrain.pretrained import EncoderClassifier
import numpy as np

outModel = 'embedding.pt'
def export():
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

    model = MyEmbedding.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    model.eval()
    pretrained_dict = model.state_dict()
    #print( pretrained_dict )
    #exit(0)

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
    traced_model = torch.jit.trace(model, [x, wav_lens1])
    torch.jit.save(traced_model, outModel)
    print( f'---- model has been saved to {outModel} ----\n' )

def verify():
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained")
    loaded_model = torch.jit.load( outModel )
    loaded_model.eval()
    loaded_model.to(torch.device('cuda'))

    wavs = [
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching1.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/shortTeaching2.wav',
            '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav',
            ]

    for wav in wavs:
        print( f"Check wav file: {wav}" )

        # Load wav
        signal, fs = torchaudio.load( wav )
        signal = signal[:80000]

        # Calc embeddings using pytorch
        print( signal.shape )
        wav_lens = torch.Tensor([1.0])
        embeddings = model.encode_batch(signal, wav_lens, normalize=False)

        # Calc embedding using onnx model
        pt_embeddings = loaded_model(signal, wav_lens)

        try:
            # Check if result is close enough
            np.testing.assert_allclose(
                    embeddings.detach().numpy(), 
                    pt_embeddings, 
                    rtol=1e-03, 
                    atol=1e-05,
                    verbose = True)
            print( 'check passed' )
        except AssertionError as e:
            print( e )
        print( "-----------------------------" )

#export()
verify()
