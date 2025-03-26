import torchaudio
from speechbrain.inference.separation import SepformerSeparation
# Load SepFormer pre-trained model
separator = SepformerSeparation.from_hparams(
    source="./results/sepformer-vivos/1234/save",
    savedir='./results/sepformer-vivos/1234/save'
    # source="speechbrain/sepformer-libri2mix",
    # savedir='./pretrained_models/sepformer-libri2mix'
)


# signal, fs = torchaudio.load('./examples/test/mix_clean/61-70968-0013_8555-284449-0003.wav')
# file_path = '../../../data/vivos/processed/processed_test/mix/mix_test_00000.wav'
file_path = '../../../data/vivos/mixture/test/data/mixture_01998.wav'
# file_path = './examples/test/mix_clean/61-70968-0013_8555-284449-0003.wav'

est_sources = separator.separate_file(path=file_path) 

torchaudio.save("./results/split_audio/sep/finetune/split1.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("./results/split_audio/sep/finetune/split2.wav", est_sources[:, :, 1].detach().cpu(), 8000)