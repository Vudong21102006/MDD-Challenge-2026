import os

import librosa
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer
from pyctcdecode import build_ctcdecoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MDDDataset, make_collate_fn
from model import PL
from utils import (
    BLANK_TOKEN_ID,
    SAMPLE_RATE,
    PAD_TOKEN_ID,
    CTC_LABELS,
    build_feature_extractor,
    create_decoder,
    get_device,
    load_vocab,
    text_to_tensor,
)


class MDDTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        print(f"Training device: {self.device}")
        self.feature_extractor = build_feature_extractor()
        self.vocab = load_vocab(args.vocab_path)

        self.df_train = pd.read_csv(args.train_csv)
        self.df_dev = pd.read_csv(args.dev_csv)

        self.train_dataset = MDDDataset(self.df_train, wav_dir=args.train_wav_dir, vocab=self.vocab)
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=make_collate_fn(self.feature_extractor, self.device),
        )

        self.dev_dataset = MDDDataset(self.df_dev, wav_dir=self.args.dev_wav_dir, vocab=self.vocab)
        self.dev_loader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=make_collate_fn(self.feature_extractor, self.device),
        )

        model_cls = PL
        # pass vocab size into model so final linear layer matches tokenizer/vocab
        vocab_size = len(self.vocab)
        self.model = model_cls.from_pretrained(args.pretrained_model, vocab_size=vocab_size)
        self.model.freeze_feature_extractor()
        self.model = self.model.to(self.device)
        self._print_trainable_parameters()

        # Build CTC decoder using the same label ordering as the model vocab
        labels = [None] * len(self.vocab)
        for tok, idx in self.vocab.items():
            labels[idx] = tok
        self.decoder_ctc = build_ctcdecoder(labels=CTC_LABELS)
        self.id2token = {idx: tok for tok, idx in self.vocab.items()}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.ctc_loss = nn.CTCLoss(blank=BLANK_TOKEN_ID)
        self.min_wer = 100.0

        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.args.num_epoch):
            running_loss = self._train_one_epoch(epoch)
            print(f"Training loss: {sum(running_loss) / len(running_loss)}")
            if epoch >= self.args.eval_start_epoch:
                epoch_wer = self._evaluate()
                if epoch_wer < self.min_wer:
                    print('save_checkpoint...')
                    self.min_wer = epoch_wer
                    ckpt_name = "checkpoint_wl.pth"
                    ckpt_path = os.path.join(self.args.checkpoint_dir, ckpt_name)
                    torch.save(self.model.state_dict(), ckpt_path)

                print(f"wer checkpoint {epoch}: {epoch_wer}")
                print(f"min_wer: {self.min_wer}")


    def _train_one_epoch(self, epoch):
        self.model.train().to(self.device)
        running_loss = []
        print(f"EPOCH {epoch}:")
        worderrorrate = []

        for batch_idx, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            input_values, linguistic, labels, target_lengths, wav_lengths = data

            logits = self.model(input_values, linguistic)

            logits = logits.transpose(0, 1)
            logits = F.log_softmax(logits, dim=2)
            input_lengths = torch.full(
                (logits.shape[1],),
                logits.shape[0],
                dtype=torch.long,
                device=self.device,
            )


            loss = self.ctc_loss(logits, labels, input_lengths, target_lengths)

            running_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return running_loss

    def _print_trainable_parameters(self):
        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        backbone_total = sum(parameter.numel() for parameter in self.model.wav2vec2.parameters())
        backbone_trainable = sum(parameter.numel() for parameter in self.model.wav2vec2.parameters() if parameter.requires_grad)
        print(
            f"Trainable params: {trainable_params}/{total_params} | "
            f"wav2vec2 trainable: {backbone_trainable}/{backbone_total}"
        )

    def _greedy_decode_tokens(self, frame_logits: torch.Tensor) -> str:
        pred_ids = torch.argmax(frame_logits, dim=-1).tolist()
        collapsed_ids = []
        prev = None
        for token_id in pred_ids:
            if token_id != prev:
                collapsed_ids.append(token_id)
            prev = token_id

        tokens = []
        for token_id in collapsed_ids:
            if token_id == BLANK_TOKEN_ID:
                continue
            token = self.id2token.get(token_id, '')
            if token and token != '$':
                tokens.append(token)
        return ' '.join(tokens)

    @staticmethod
    def _get_column_value(dataframe, index: int, names, default=''):
        for name in names:
            if name in dataframe.columns:
                return dataframe[name][index]
        return default

    def _resolve_dev_wav_path(self, point: int) -> str:
        audio_path = self._get_column_value(self.df_dev, point, ['AudioPath', 'audio_path'])
        if audio_path:
            return str(audio_path)

        raw_path = str(self._get_column_value(self.df_dev, point, ['Path', 'path']))
        if raw_path.lower().endswith('.wav'):
            if os.path.isabs(raw_path):
                return raw_path
            return os.path.join(self.args.dev_wav_dir, raw_path)
        return os.path.join(self.args.dev_wav_dir, f"{raw_path}.wav")

    def _evaluate(self):
        self.model.eval().to(self.device)
        worderrorrate = []

        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(self.dev_loader), total=len(self.dev_loader)):
                input_values, linguistic, labels, target_lengths, wav_lengths = data
                input_lengths = self.model._get_feat_extract_output_lengths(wav_lengths)

                logits = self.model(input_values, linguistic)
                logits = F.log_softmax(logits, dim=2)

                for b in range(logits.shape[0]):
                    valid_len = input_lengths[b].item()
                    sample_logits = logits[b, :valid_len, :]
                    hypothesis = self._greedy_decode_tokens(sample_logits)
                    with open("debug.txt", "a") as f:
                        f.write(f"{hypothesis} \n")
                    global_idx = batch_idx * self.dev_loader.batch_size + b
                    if global_idx >= len(self.df_dev):
                        break
                    
                    transcript_raw = self._get_column_value(self.df_dev, global_idx, ['Transcript', 'transcript'])
                    transcript = ' '.join(t for t in transcript_raw.split(' ') if t and t != '$')

                    worderrorrate.append(wer(transcript, hypothesis))

        return sum(worderrorrate) / len(worderrorrate) if worderrorrate else 0.0

    def inference(self, csv_path: str, wav_dir: str = None, out_path: str = 'results.csv', batch_size: int = None, checkpoint = "checkpoint/checkpoint_wl.pth"):
        """Run inference on a CSV with a `path` column and write predictions to `out_path`.

        Each line in the output CSV contains the space-separated token hypothesis for the
        corresponding input audio path.
        """
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        df = pd.read_csv(csv_path)
        paths = df['path'].tolist() if 'path' in df.columns else df['Path'].tolist()
        canonicals_all = df['canonical'].tolist() if 'canonical' in df.columns else df['Canonical'].tolist()
        wav_dir = wav_dir or self.args.train_wav_dir
        batch_size = batch_size or max(1, getattr(self.args, 'batch_size', 1) * 2)

        results = []

        # padding id for linguistic embedding (model embedding uses vocab_size as padding idx)
        pad_id = len(self.vocab)
        with torch.no_grad():
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i : i + batch_size]
                waveforms = []
                wav_lengths = []

                # load waveforms
                for p in batch_paths:
                    raw = str(p)
                    if raw.lower().endswith('.wav'):
                        if os.path.isabs(raw):
                            wav_path = raw
                        else:
                            wav_path = os.path.join(wav_dir, raw)
                    else:
                        wav_path = os.path.join(wav_dir, f"{raw}.wav")

                    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

                    waveforms.append(audio)
                    wav_lengths.append(len(audio))

                # feature extraction
                inputs = self.feature_extractor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors='pt')
                input_values = inputs.input_values.to(self.device)
                wav_lengths = torch.tensor(wav_lengths, dtype=torch.long, device=self.device)

                # build canonical token tensor from CSV column and pad to batch max length
                batch_can = canonicals_all[i : i + batch_size]
                token_lists = [text_to_tensor(t, self.vocab) for t in batch_can]
                max_can_len = max((len(t) for t in token_lists), default=1)
                canonical = torch.full((input_values.shape[0], max_can_len), fill_value=PAD_TOKEN_ID, dtype=torch.long, device=self.device)
                for j, toks in enumerate(token_lists):
                    if toks:
                        canonical[j, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=self.device)

                # forward
                with torch.no_grad():
                    logits = self.model(input_values, canonical)
                    logits = F.log_softmax(logits, dim=2)
                    input_lengths = torch.full(
                        (logits.shape[0],),
                        logits.shape[1],
                        dtype=torch.long,
                        device=self.device,
                    )

                for b in range(logits.shape[0]):
                    valid_len = int(input_lengths[b].item())
                    sample_logits = logits[b, :valid_len, :]
                    hypothesis = self._greedy_decode_tokens(sample_logits)
                    results.append(hypothesis)

        # write results
        out_df = pd.DataFrame({'predict': results})
        out_df.to_csv(out_path, index=False)
        return out_path