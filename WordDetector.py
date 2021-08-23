
from AudioDataset import AudioDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchaudio.transforms as T
from tqdm.notebook import tqdm

class DankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.firstBN = nn.BatchNorm2d(num_features=3, eps=1e-05, momentum=0.1, affine=True)
        self.second = torchvision.models.resnet50(pretrained=False)
        self.second.fc = nn.Linear(in_features=2048, out_features=35, bias=True)

    def forward(self, x):
        x = self.first(x)
        x = self.firstBN(x)
        x = self.second(x)
        return x

class WordDetectorSpectrogram():
    def __init__(self, device, batch_size):
        self._device = device
        self._net = DankNet().to(self._device)
        self._train_loader = AudioDataset.get_loader('train', device, batch_size, self._collate_fn)
        self._test_loader = AudioDataset.get_loader('test', device, batch_size, self._collate_fn)
        self._train_labels = AudioDataset.get_labels()
        self._mfcc_tform = T.MFCC(
            sample_rate=AudioDataset.get_sample_rate(),
            n_mfcc=128,
            melkwargs={
                'n_fft': 4096,
                'n_mels': 128,
                'hop_length': round((AudioDataset.get_sample_rate() * 7) / 1000),
                'win_length': round((AudioDataset.get_sample_rate() * 50) / 1000),
                'mel_scale': 'htk',
            }
        ).to(self._device)
        print(f"WordDetector with mel spectrograms ready on device: {self._device}")

    def _collate_fn(self, batch):
        tensors = []
        targets = []

        # (waveform, sample_rate, label, speaker_id, utterance_num)
        for waveform, _, label, _, _ in batch:
            tensors += [waveform]
            targets += [torch.tensor(self._train_labels.index(label))]

        # Pad with zeros to make sure each tensor is the same length.
        tensors = [i.t() for i in tensors]
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.)
        tensors = tensors.permute(0, 2, 1)

        # Compute mel-spectrograms
        tensors = [self._mfcc_tform(tensor) for tensor in tensors]

        targets = torch.stack(targets)
        return tensors, targets

    def _exec_callback(self, is_train, ):
        pass

    def _exec(self, epoch, loader, optimizer, batch_callback):
        is_train = loader == self._train_loader
        self._net.train() if is_train else self._net.eval()

        correct = 0
        for idx, (data, target) in enumerate(loader):
            data = data.to(self._device)
            target = target.to(self._device)
            output = self._net(data)

            if is_train:
                loss = F.cross_entropy(torch.flatten(output), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 20 == 0:
                    print(f"Train Epoch: {epoch} [{idx * len(data)}/{len(loader.dataset)} ({100. * idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}")
            else:
                prediction = output.argmax(dim=-1)
                correct += prediction.squeeze().eq(target).sum().item()
            batch_callback()

        if is_train:
            print(f"Train Epoch: {epoch} [{idx * len(data)}/{len(loader.dataset)} ({100. * idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}")
        else:
            print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)\n")
        batch_callback()

    def train(self, epochs=1):
        with tqdm(total=epochs) as progress_bar:
            pbar_update = 1 / (len(self._train_loader) + len(self._test_loader))
            def cb():
                progress_bar.update(pbar_update)

            optimizer = optim.Adam(self._net.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                      max_lr=1e-3,
                                                      epochs=epochs,
                                                      steps_per_epoch=100)
            for epoch in range(0, epochs):
                self._exec(epoch, self._train_loader, optimizer, cb)
                self._exec(epoch, self._test_loader, optimizer, cb)
                scheduler.step()
