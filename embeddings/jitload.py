
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from hyperpyyaml import load_hyperpyyaml


ckpt = "pretrained_models/EncoderClassifier-8f6f7fdaa9628acf73e21ad1f99d5f83/hyperparams.yaml"

hparams = load_hyperpyyaml(
    open( ckpt ),
)

model = EncoderClassifier(modules=hparams['modules'], hparams=dict(hparams))

model.eval()

x = torch.randn( 1, 1920000 )
print( x )
print( x.shape )
#exit( 0 )

traced_script_module = torch.jit.trace(model, x)
traced_script_module.save("traced_model.pt")

