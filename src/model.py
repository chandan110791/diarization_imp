import torch
import torch.nn as nn
from transformers import WavLMModel
from pyannote.audio import Model
from collections import namedtuple

# Fix for "Resolution" error in Pyannote
Resolution = namedtuple('Resolution', ['duration', 'step'])

class WavLMSegmentation(Model):
    def __init__(self, model_name="microsoft/wavlm-base-plus", lstm_hidden_size=128):
        # 16k sample rate, 1 channel (mono)
        super().__init__(sample_rate=16000, num_channels=1)
        
        print(f"🏗️ Initializing WavLM: {model_name}")
        self.wavlm = WavLMModel.from_pretrained(model_name)
        
        # FREEZE WavLM (We only train the head to save memory/time)
        # If you have a massive GPU, you can unfreeze this later.
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
        self.input_dim = 768 # Base-Plus output dimension
        
        # The "Diarization Head" (LSTM + Classifier)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Output: 2 classes (Speaker vs No Speaker) or more depending on task
        # But wait! For segmentation task, Pyannote handles the classifier dimension automatically via .build()
        self.lstm_out_dim = lstm_hidden_size * 2
        self.classifier = None 
        self.activation = nn.Sigmoid()

    def build(self):
        # This is called automatically by the Task when training starts
        # It sets the output size based on the number of speakers in the config
        num_classes = len(self.specifications.classes)
        self.classifier = nn.Linear(self.lstm_out_dim, num_classes)
        self.activation = self.default_activation()

    def num_frames(self, num_samples: int) -> int:
            import torch

            # 'self.wavlm' must match the variable name of your WavLMModel in __init__
            # We pass the input length to the HF helper method to get the EXACT output length
            with torch.no_grad():
                output_lengths = self.wavlm._get_feat_extract_output_lengths(torch.tensor([num_samples]))

            return output_lengths.item()
            
    def forward(self, waveforms: torch.Tensor, weights: torch.Tensor = None):
        # 1. WavLM Expects (Batch, Time), but we might have (Batch, 1, Time)
        if waveforms.dim() == 3:
            inputs = waveforms.squeeze(1)
        else:
            inputs = waveforms

        # 2. Extract Features (Frozen)
        with torch.no_grad():
            outputs = self.wavlm(inputs)
            features = outputs.last_hidden_state

        # 3. Pass through LSTM Head (Trainable)
        lstm_out, _ = self.lstm(features)
        
        # 4. Classify
        return self.activation(self.classifier(lstm_out))

    # --- REQUIRED PYANNOTE PROPERTIES ---
    @property
    def receptive_field(self):
        # WavLM outputs 1 frame every 20ms (0.02s)
        return Resolution(duration=0.02, step=0.02)

    @property
    def dimension(self):
        return self.input_dim