def __getitem__(self, idx):
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(self.audio_files[idx])

    # 调整音频长度（这里假设你想将音频限制为12秒）
    waveform = pad_or_trim_waveform(waveform, sample_rate * 12)

    # 提取梅尔频谱图 (channels, n_mels, time)
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=512, n_mels=40)
    mel_specgram = mel_transform(waveform)
    
    # 梅尔频谱图的标准化
    mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()

    # 提取MFCC (channels, n_mfcc, time)，确保 n_mfcc 和 n_mels 一致
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)
    mfcc = mfcc_transform(waveform)

    # MFCC的标准化
    mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()

    # 确保梅尔频谱图和MFCC的形状一致后进行拼接 (channels, n_mels+n_mfcc, time)
    feature = torch.cat([mel_specgram_norm, mfcc_norm], dim=1)

    # 去除channels这个维度 (channels, n_features, time) -> (n_features, time)
    # 并将 (n_features, time) 转换为 (time, n_features)
    feature = feature[0].permute(1, 0)

    return feature, self.labels[idx]
