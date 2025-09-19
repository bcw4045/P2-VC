
# P²-VC: Speech Disentanglement by Cross-Factor Perturbation and Perturbed Latent Mixup for Zero-Shot Voice Conversion

This is the official implementation of the paper P²-VC: Speech Disentanglement by Cross-Factor Perturbation and Perturbed Latent Mixup for Zero-Shot Voice Conversion

## Abstract
A successful realization of speech disentanglement plays a crucial role in zero-shot voice conversion (ZSVC), but feature leakage between disentanglement factors, such as content and speaker, limits the performance of ZSVC. To mitigate this problem, this paper first proposes a cross-factor perturbation (CFP) approach that perturbs one disentanglement factor while others are maintained. In this work, the content and speaker information are considered as disentanglement factors; thus, the CFP helps the content encoder learn representations invariant to speaker characteristics, while the speaker encoder learns representations invariant to linguistic content.

In addition, a perturbed latent mixup (PLM) method, which is a kind of data augmentation, is proposed to reduce generalization errors by blending unperturbed and perturbed embeddings. The proposed CFP and PLM are executed through a single training step, circumventing the need for parallel data or complex multi-stage procedures and yielding high-quality voice conversion. Experimental results confirm that the ZSVC trained by the proposed methods achieves strong content preservation, validating the effectiveness of the CFP and PLM on voice conversion performance. Furthermore, performance evaluations on an unseen-to-unseen dataset indicate that the proposed ZSVC model produces speech with higher speaker similarity and perceptual quality than conventional models.

## Data Preprocess

### Download Dataset
We use the VCTK dataset as training data. Download VCTK from the official website.

### Dataset Path Configuration
<pre>
<code>ln -s {your_dataset_location} {WorkingDir} </code>
</pre>

### Data Preparation
The dataset should be prepared in the format ```data_path|text|spk_id```, as shown below. The generated metadata should be located in the ```resource/filelist``` path.
```
# train.txt
VCTK/test.wav|This is test|0
VCTK/test2.wav|This is test2|0
VCTK/test3.wav|This is test3|0
```

## Configuration
Settings required for learning can be modified in params.py. Enter the paths of the metadata created for training in train_filelist_path, valid_filelist_path, and test_filelist_path, respectively.

```
train_filelist_path = 'resources/filelists/train.txt'
valid_filelist_path = 'resources/filelists/test2.txt'
test_filelist_path = 'resources/filelists/test3.txt'
```


## Training
```
python train.py
```

## Inference
```
python inference.py --ckpt ./logs/P2VC/p2vc.pt --src ./src.wav --tgt ./tgt.wav --out ./converted.wav
```