
import os
import torch
import torchaudio

class AudioDataset(torchaudio.datasets.SPEECHCOMMANDS):
  def __init__(self, subset: str = None):
    super().__init__("./", download=True)
    def load_list(filename):
      filepath = os.path.join(self._path, filename)
      with open(filepath) as fileobj:
        return [os.path.join(self._path, line.strip()) for line in fileobj]
    if subset == "validation":
      self._walker = load_list("validation_list.txt")
    elif subset == "testing":
      self._walker = load_list("testing_list.txt")
    elif subset == "training":
      excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
      excludes = set(excludes)
      self._walker = [w for w in self._walker if w not in excludes]

  @classmethod
  def get_loader(cls, type, device, batch_size, collate_fn):
    assert(type=='train' or type=='test')
    nworkers = 1 if device == "cuda" else 0
    pinmem = True if device == "cuda" else False
    return torch.utils.data.DataLoader(
        AudioDataset("training") if type == 'train' else AudioDataset("testing"),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=nworkers,
        pin_memory=pinmem,
    )

  @classmethod
  def get_labels(cls):
    return sorted(list(set(dp[2] for dp in AudioDataset("training"))))

  @classmethod
  def get_sample_rate(cls):
    return 16000