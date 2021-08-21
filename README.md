# Update for Speech2Text (s2t)
# original work from paper on "fairseq S2T: Fast Speech-to-Text Modeling with fairseq" 
# https://paperswithcode.com/paper/fairseq-s2t-fast-speech-to-text-modeling-with
# github repo: https://github.com/pytorch/fairseq.git

```bash
git clone https://github.com/pytorch/fairseq.git
```

Install additional requirements for Speech Translation (ST)
```bash
pip install pandas sentencepiece
pip install fastBPE
pip install sacremoses

```

Try Speech Translation
```bash
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```

Run following command for make required files
Speech1Text here: https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/docs/librispeech_example.md

Update:
fairseq/data/audio/audio_utils.py
```bash
line#36
def convert_to_mon to def _convert_to_mon
line#82
waveform = _convert_to_mono(waveform, sample_rate)
```
Update on the examples/speech_to_text/prep_librispeech_data.py
```bash
SPLITS = [
    "train-clean-100",
   # "train-clean-360",
   # "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

line #52
sample_id = f"{spk_id}-{100}-{utt_no}"
line#69
sample_id = f"{spk_id}-{100}-{utt_no}"
```


```bash
python3 examples/speech_to_text/prep_librispeech_data.py \
  --output-root out_dir --vocab-type unigram --vocab-size 10000
```


Run training
```bash
fairseq-train out_dir --save-dir save_dir \
  --config-yaml config.yaml --train-subset train-clean-100 --valid-subset dev-clean,dev-other \
  --num-workers 1 --max-tokens 50000 --max-update 500000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8

```

Download weights/model file:
```bash
wget "https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_s.pt"
```

Run Inference for Audio file
```bash
CHECKPOINT_FILENAME=librispeech_transformer_s.pt
fairseq-interactive out_dir --config-yaml config.yaml --task speech_to_text \
  --path save_dir/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5
```



For soundfile read error 
```bash
sudo apt-get install libsndfile1
```


