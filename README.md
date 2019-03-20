# MiniS2S

Minimum sequence-to-sequence model to demonstrate how to use TensorFlow API.

Visit https://github.com/kiidax/minis2s for the original copy.

This implements basic sequence-to-sequence model with LSTM encoder and decoder to
translate an English word into phone symbols using CMU pronuncing dictionary
(http://www.speech.cs.cmu.edu/cgi-bin/cmudict). The model is very small and will run on
a notebook with 4GB memory.

# Prerequisites

- Python 3.6
- TensorFlow 1.12.0

# Data preparation

```
$ python minis2s.py --mode=datagen --data_dir=data
```