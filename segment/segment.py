from pyannote.audio.core.model import Model
from utils import SlidingWindow, SlidingWindowFeature, Segment
from typing import Optional, Tuple, Union
from hf_token import *
import torchaudio
import numpy as np
import torch
from einops import rearrange
import itertools
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class Segmentation:
    def __init__(self, device:torch.device = None):
        self.duration = 5.0
        self.step = 0.5
        self.batch_size = 32
        self.sample_rate = 16000
        self.embedding_batch_size = 32 # from  speaker_diarization config.yaml
        # minimum number of samples needed to extract an embedding
        # (a lower number of samples would result in an error)
        # This value determined by pyannote/audio/pipelines/speaker_verification.py:264
        self.min_num_samples = 640 # self._embedding.min_num_samples
        self.model =  Model.from_pretrained(
                "pyannote/segmentation@2022.07",
                strict=False,
                use_auth_token=hf_auth_token,
            )
        self.model.eval()
        self.model.to( torch.device( 'cuda' ))
        if device is None:
            self.device = self.model.device
        else:
            self.device = device
        #self.device = torch.device('cuda')

        # read from 
        # ~/.cache/torch/pyannote/models--pyannote--speaker-diarization/snapshots/2c6a571d14c3794623b098a065ff95fa22da7f25/config.yaml
        self.diarization_segmentation_threashold = 0.4442333667381752
        self.diarization_segmentation_min_duration_off = 0.5817029604921046



    def slide(self, waveform: torch.Tensor, sample_rate: int) -> SlidingWindowFeature:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.

        Returns
        -------
        output : SlidingWindowFeature
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """

        window_size: int = round(self.duration * sample_rate)
        step_size: int = round(self.step * sample_rate)
        num_channels, num_samples = waveform.shape

        # Those code only to calc num_frames_per_chunk
        specifications = self.model.specifications
        introspection = self.model.introspection 
        num_frames_per_chunk, _ = introspection(window_size)

        # prepare complete chunks
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0 

        # prepare last incomplete chunk
        has_last_chunk = (num_samples < window_size) or (
            num_samples - window_size
        ) % step_size > 0
        if has_last_chunk:
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]

        outputs: Union[List[np.ndarray], np.ndarray] = list()

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]
            outputs.append(self.infer(batch))

        # process orphan last chunk
        if has_last_chunk:
            last_output = self.infer(last_chunk[None])
            pad = num_frames_per_chunk - last_output.shape[1]
            last_output = np.pad(last_output, ((0, 0), (0, pad), (0, 0)))

            outputs.append(last_output)

        outputs = np.vstack(outputs)

        # /home/leo/product/speaker_diarization/diarization/lib/python3.10/site-packages/pyannote/core/segment.py:441
        # /home/leo/product/speaker_diarization/diarization/lib/python3.10/site-packages/pyannote/core/feature.py:49
        frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
        return SlidingWindowFeature(outputs, frames)

    def infer(self, chunks: torch.Tensor) -> np.ndarray:
        """Forward pass

        Takes care of sending chunks to right device and outputs back to CPU

        Parameters
        ----------
        chunks : (batch_size, num_channels, num_samples) torch.Tensor
            Batch of audio chunks.

        Returns
        -------
        outputs : (batch_size, ...) np.ndarray
            Model output.
        """

        with torch.no_grad():
            try:
                outputs = self.model(chunks.to(self.device))
            except RuntimeError as exception:
                if is_oom_error(exception):
                    raise MemoryError(
                        f"batch_size ({self.batch_size: d}) is probably too large. "
                        f"Try with a smaller value until memory error disappears."
                    )
                else:
                    raise exception

        return outputs.cpu().numpy()


    def run( self, waveform:torch.Tensor, sample_rate:int ):
        # waveform, sample_rate = self.model.audio(file)
        segmentations = self.slide( waveform, sample_rate ) 
        #   shape: (num_chunks, num_frames, local_num_speakers)

        # estimate frame-level number of instantaneous speakers
        #count = self.speaker_count(
        #    segmentations,
        #    onset=self.diarization_segmentation_threashold,
        #    frames=self._frames,
        #)
        #   shape: (num_frames, 1)
        #   dtype: int

        # binarize segmentation
        binary_segmentations: SlidingWindowFeature = self.binarize_swf(
            segmentations,
            onset=self.diarization_segmentation_threashold,
            initial_state=False,
        )

        # Next, pass binarized_segmentations and audio file to embedding module
        # pyannote/audio/pipelines/speaker_diarization.py: get_embeddings(...)
        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, _ = binary_segmentations.data.shape

        # corresponding minimum number of frames
        num_samples = duration * 16000 # self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * self.min_num_samples / num_samples)

        # zero-out frames with overlapping speech
        clean_frames = 1.0 * (
            np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
        )
        clean_segmentations = SlidingWindowFeature(
            binary_segmentations.data * clean_frames,
            binary_segmentations.sliding_window,
        )

        def iter_waveform_and_mask():
            for (chunk, masks), (_, clean_masks) in zip(
                binary_segmentations, clean_segmentations
            ):
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray

                wvform, _ = self.crop(
                    waveform,
                    sample_rate,
                    chunk,
                    duration=duration,
                    mode="pad",
                )
                # wvform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    # mask: (num_frames, ) np.ndarray

                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask

                    yield wvform[None], torch.from_numpy(used_mask)[None]
                    # w: (1, 1, num_samples) torch.Tensor
                    # m: (1, num_frames) torch.Tensor

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )

        embedding_batches = []

        for batch in batches:
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

            waveform_batch = torch.vstack(waveforms)
            # (batch_size, 1, num_samples) torch.Tensor

            mask_batch = torch.vstack(masks)
            # (batch_size, num_frames) torch.Tensor

            embedding_batch: np.ndarray = self.embedding_mask(
                waveform_batch, masks=mask_batch
            )
            # (batch_size, dimension) np.ndarray
            embedding_batches.append(embedding_batch)

        #embedding_batches = np.vstack(embedding_batches)
        #embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

        #return embeddings

    def embedding_mask(
        self, waveforms: torch.Tensor, masks: torch.Tensor = None
        ) -> np.ndarray:
        """

        Parameters
        ----------
        waveforms : (batch_size, num_channels, num_samples)
            Only num_channels == 1 is supported.
        masks : (batch_size, num_samples), optional

        Returns
        -------
        embeddings : (batch_size, dimension)

        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1

        waveforms = waveforms.squeeze(dim=1)

        if masks is None:
            signals = waveforms.squeeze(dim=1)
            wav_lens = signals.shape[1] * torch.ones(batch_size)

        else:

            batch_size_masks, _ = masks.shape
            assert batch_size == batch_size_masks

            # TODO: speed up the creation of "signals"
            # preliminary profiling experiments show
            # that it accounts for 15% of __call__
            # (the remaining 85% being the actual forward pass)

            imasks = F.interpolate(
                masks.unsqueeze(dim=1), size=num_samples, mode="nearest"
            ).squeeze(dim=1)

            imasks = imasks > 0.5

            signals = pad_sequence(
                [waveform[imask] for waveform, imask in zip(waveforms, imasks)],
                batch_first=True,
            )
            wav_lens = imasks.sum(dim=1)

        max_len = wav_lens.max()

        # corner case: every signal is too short
        if max_len < self.min_num_samples:
            return np.NAN * np.zeros((batch_size, self.dimension))

        too_short = wav_lens < self.min_num_samples
        wav_lens = wav_lens / max_len
        wav_lens[too_short] = 1.0

        #**********************************************
        # This is the step call embedding model to get embeddings
        #**********************************************
        #embeddings = (
        #    self.classifier_.encode_batch(signals, wav_lens=wav_lens)
        #    .squeeze(dim=1)
        #    .cpu()
        #    .numpy()
        #)

        #embeddings[too_short.cpu().numpy()] = np.NAN

        #return embeddings
        return None


    # from pyannote/audio/core/io.py
    def crop(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment: Segment,
        duration: Optional[float] = None,
        mode="raise",
    ) -> Tuple[torch.Tensor, int]:

        # infer which samples to load from sample rate and requested chunk
        start_frame = math.floor(segment.start * sample_rate)
        frames = waveform.shape[1]

        if duration:
            num_frames = math.floor(duration * sample_rate)
            end_frame = start_frame + num_frames
        else:
            end_frame = math.floor(segment.end * sample_rate)
            num_frames = end_frame - start_frame

        pad_start = -min(0, start_frame)
        pad_end = max(end_frame, frames) - frames
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, frames)
        num_frames = end_frame - start_frame

        data = waveform[:, start_frame:end_frame]

        # pad with zeros
        if mode == "pad":
            data = F.pad(data, (pad_start, pad_end))

        return data, sample_rate


    def binarize_swf( 
            self,
        scores: SlidingWindowFeature,
        onset: float = 0.5,
        offset: Optional[float] = None,
        initial_state: Optional[bool] = None,
    ):
        """(Batch) hysteresis thresholding

        Parameters
        ----------
        scores : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)- or (num_frames, num_classes)-shaped scores.
        onset : float, optional
            Onset threshold. Defaults to 0.5.
        offset : float, optional
            Offset threshold. Defaults to `onset`.
        initial_state : np.ndarray or bool, optional
            Initial state.

        Returns
        -------
        binarized : same as scores
            Binarized scores with same shape and type as scores.

        """

        offset = offset or onset

        if scores.data.ndim == 2:
            num_frames, num_classes = scores.data.shape
            data = einops.rearrange(scores.data, "f k -> k f", f=num_frames, k=num_classes)
            binarized = self.binarize_ndarray(
                data, onset=onset, offset=offset, initial_state=initial_state
            )
            return SlidingWindowFeature(
                1.0
                * einops.rearrange(binarized, "k f -> f k", f=num_frames, k=num_classes),
                scores.sliding_window,
            )

        elif scores.data.ndim == 3:
            num_chunks, num_frames, num_classes = scores.data.shape
            data = rearrange(
                scores.data, "c f k -> (c k) f", c=num_chunks, f=num_frames, k=num_classes
            )
            binarized = self.binarize_ndarray(
                data, onset=onset, offset=offset, initial_state=initial_state
            )
            return SlidingWindowFeature(
                1.0
                * rearrange(
                    binarized, "(c k) f -> c f k", c=num_chunks, f=num_frames, k=num_classes
                ),
                scores.sliding_window,
            )

        else:
            raise ValueError(
                "Shape of scores must be (num_chunks, num_frames, num_classes) or (num_frames, num_classes)."
            )

    def binarize_ndarray(
            self,
            scores: np.ndarray,
            onset: float = 0.5,
            offset: Optional[float] = None,
            initial_state: Optional[Union[bool, np.ndarray]] = None,
        ):
        """(Batch) hysteresis thresholding

        Parameters
        ----------
        scores : numpy.ndarray
            (num_frames, num_classes)-shaped scores.
        onset : float, optional
            Onset threshold. Defaults to 0.5.
        offset : float, optional
            Offset threshold. Defaults to `onset`.
        initial_state : np.ndarray or bool, optional
            Initial state.

        Returns
        -------
        binarized : same as scores
            Binarized scores with same shape and type as scores.
        """

        offset = offset or onset

        batch_size, num_frames = scores.shape

        scores = np.nan_to_num(scores)

        if initial_state is None:
            initial_state = scores[:, 0] >= 0.5 * (onset + offset)

        elif isinstance(initial_state, bool):
            initial_state = initial_state * np.ones((batch_size,), dtype=bool)

        elif isinstance(initial_state, np.ndarray):
            assert initial_state.shape == (batch_size,)
            assert initial_state.dtype == bool

        initial_state = np.tile(initial_state, (num_frames, 1)).T

        on = scores > onset
        off_or_on = (scores < offset) | on

        # indices of frames for which the on/off state is well-defined
        well_defined_idx = np.array(
            list(itertools.zip_longest(*[np.nonzero(oon)[0] for oon in off_or_on], fillvalue=-1))
        ).T

        # corner case where well_defined_idx is empty
        if not well_defined_idx.size:
            return np.zeros_like(scores, dtype=bool) | initial_state

        # points to the index of the previous well-defined frame
        same_as = np.cumsum(off_or_on, axis=1)

        samples = np.tile(np.arange(batch_size), (num_frames, 1)).T

        return np.where(
            same_as, on[samples, well_defined_idx[samples, same_as - 1]], initial_state
        )




if __name__ == '__main__':
    #audio = '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/one_min16k.wav')
    #audio = '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_7-speakers_SkillsFuture.wav'
    audio = '/home/leo/storage/sharedFolderVirtualbox/audioForTesting/multi-speaker_4-speakers_Jennifer_Aniston_and_Adam_Sandler_talk.wav'
    signal, fs = torchaudio.load( audio )
    seg = Segmentation()
    seg.run( signal, fs )
