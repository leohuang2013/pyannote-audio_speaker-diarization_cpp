
import numbers
import warnings
from typing import Tuple, Optional, Union, Iterator, Iterable, List, Text
#from typing import Dict, Tuple, Iterable, List, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

# setting 'frozen' to True makes it hashable and immutable
@dataclass(frozen=True, order=True)
class Segment:
    """
    Time interval

    Parameters
    ----------
    start : float
        interval start time, in seconds.
    end : float
        interval end time, in seconds.


    Segments can be compared and sorted using the standard operators:

    >>> Segment(0, 1) == Segment(0, 1.)
    True
    >>> Segment(0, 1) != Segment(3, 4)
    True
    >>> Segment(0, 1) < Segment(2, 3)
    True
    >>> Segment(0, 1) < Segment(0, 2)
    True
    >>> Segment(1, 2) < Segment(0, 3)
    False

    Note
    ----
    A segment is smaller than another segment if one of these two conditions is verified:

      - `segment.start < other_segment.start`
      - `segment.start == other_segment.start` and `segment.end < other_segment.end`

    """
    start: float = 0.0
    end: float = 0.0

    @staticmethod
    def set_precision(ndigits: Optional[int] = None):
        """Automatically round start and end timestamps to `ndigits` precision after the decimal point

        To ensure consistency between `Segment` instances, it is recommended to call this method only 
        once, right after importing `pyannote.core.Segment`.

        Usage
        -----
        >>> from pyannote.core import Segment
        >>> Segment.set_precision(2)
        >>> Segment(1/3, 2/3)
        <Segment(0.33, 0.67)>
        """
        global AUTO_ROUND_TIME
        global SEGMENT_PRECISION

        if ndigits is None:
            # backward compatibility
            AUTO_ROUND_TIME = False
            # 1 μs (one microsecond)
            SEGMENT_PRECISION = 1e-6
        else:
            AUTO_ROUND_TIME = True
            SEGMENT_PRECISION = 10 ** (-ndigits)

    def __bool__(self):
        """Emptiness

        >>> if segment:
        ...    # segment is not empty.
        ... else:
        ...    # segment is empty.

        Note
        ----
        A segment is considered empty if its end time is smaller than its
        start time, or its duration is smaller than 1μs.
        """
        return bool((self.end - self.start) > SEGMENT_PRECISION)

    #def __post_init__(self):
    #    """Round start and end up to SEGMENT_PRECISION precision (when required)"""
    #    if AUTO_ROUND_TIME:
    #        object.__setattr__(self, 'start', int(self.start / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)
    #        object.__setattr__(self, 'end', int(self.end / SEGMENT_PRECISION + 0.5) * SEGMENT_PRECISION)

    @property
    def duration(self) -> float:
        """Segment duration (read-only)"""
        return self.end - self.start if self else 0.

    @property
    def middle(self) -> float:
        """Segment mid-time (read-only)"""
        return .5 * (self.start + self.end)

    def __iter__(self) -> Iterator[float]:
        """Unpack segment boundaries
        >>> segment = Segment(start, end)
        >>> start, end = segment
        """
        yield self.start
        yield self.end

    def copy(self) -> 'Segment':
        """Get a copy of the segment

        Returns
        -------
        copy : Segment
            Copy of the segment.
        """
        return Segment(start=self.start, end=self.end)

    # ------------------------------------------------------- #
    # Inclusion (in), intersection (&), union (|) and gap (^) #
    # ------------------------------------------------------- #

    def __contains__(self, other: 'Segment'):
        """Inclusion

        >>> segment = Segment(start=0, end=10)
        >>> Segment(start=3, end=10) in segment:
        True
        >>> Segment(start=5, end=15) in segment:
        False
        """
        return (self.start <= other.start) and (self.end >= other.end)

    def __and__(self, other):
        """Intersection

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment & other_segment
        <Segment(5, 10)>

        Note
        ----
        When the intersection is empty, an empty segment is returned:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> intersection = segment & other_segment
        >>> if not intersection:
        ...    # intersection is empty.
        """
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Segment(start=start, end=end)

    def intersects(self, other: 'Segment') -> bool:
        """Check whether two segments intersect each other

        Parameters
        ----------
        other : Segment
            Other segment

        Returns
        -------
        intersect : bool
            True if segments intersect, False otherwise
        """

        return (self.start < other.start and
                other.start < self.end - SEGMENT_PRECISION) or \
               (self.start > other.start and
                self.start < other.end - SEGMENT_PRECISION) or \
               (self.start == other.start)

    def overlaps(self, t: float) -> bool:
        """Check if segment overlaps a given time

        Parameters
        ----------
        t : float
            Time, in seconds.

        Returns
        -------
        overlap: bool
            True if segment overlaps time t, False otherwise.
        """
        return self.start <= t and self.end >= t

    def __or__(self, other: 'Segment') -> 'Segment':
        """Union

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(5, 15)
        >>> segment | other_segment
        <Segment(0, 15)>

        Note
        ----
        When a gap exists between the segment, their union covers the gap as well:

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment | other_segment
        <Segment(0, 20)
        """

        # if segment is empty, union is the other one
        if not self:
            return other
        # if other one is empty, union is self
        if not other:
            return self

        # otherwise, do what's meant to be...
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return Segment(start=start, end=end)

    def __xor__(self, other: 'Segment') -> 'Segment':
        """Gap

        >>> segment = Segment(0, 10)
        >>> other_segment = Segment(15, 20)
        >>> segment ^ other_segment
        <Segment(10, 15)

        Note
        ----
        The gap between a segment and an empty segment is not defined.

        >>> segment = Segment(0, 10)
        >>> empty_segment = Segment(11, 11)
        >>> segment ^ empty_segment
        ValueError: The gap between a segment and an empty segment is not defined.
        """

        # if segment is empty, xor is not defined
        if (not self) or (not other):
            raise ValueError(
                'The gap between a segment and an empty segment '
                'is not defined.')

        start = min(self.end, other.end)
        end = max(self.start, other.start)
        return Segment(start=start, end=end)

    def _str_helper(self, seconds: float) -> str:
        from datetime import timedelta
        negative = seconds < 0
        seconds = abs(seconds)
        td = timedelta(seconds=seconds)
        seconds = td.seconds + 86400 * td.days
        microseconds = td.microseconds
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '%s%02d:%02d:%02d.%03d' % (
            '-' if negative else ' ', hours, minutes,
            seconds, microseconds / 1000)

    def __str__(self):
        """Human-readable representation

        >>> print(Segment(1337, 1337 + 0.42))
        [ 00:22:17.000 -->  00:22:17.420]

        Note
        ----
        Empty segments are printed as "[]"
        """
        if self:
            return '[%s --> %s]' % (self._str_helper(self.start),
                                    self._str_helper(self.end))
        return '[]'

    def __repr__(self):
        """Computer-readable representation

        >>> Segment(1337, 1337 + 0.42)
        <Segment(1337, 1337.42)>
        """
        return '<Segment(%g, %g)>' % (self.start, self.end)

    def for_json(self):
        """Serialization

        See also
        --------
        :mod:`pyannote.core.json`
        """
        return {'start': self.start, 'end': self.end}

    @classmethod
    def from_json(cls, data):
        """Deserialization

        See also
        --------
        :mod:`pyannote.core.json`
        """
        return cls(start=data['start'], end=data['end'])

    def _repr_png_(self):
        """IPython notebook support

        See also
        --------
        :mod:`pyannote.core.notebook`
        """
        from .notebook import MATPLOTLIB_IS_AVAILABLE, MATPLOTLIB_WARNING
        if not MATPLOTLIB_IS_AVAILABLE:
            warnings.warn(MATPLOTLIB_WARNING.format(klass=self.__class__.__name__))
            return None

        from .notebook import repr_segment
        try:
            return repr_segment(self)
        except ImportError:
            warnings.warn(
                f"Couldn't import matplotlib to render the vizualization for object {self}. To enable, install the required dependencies with 'pip install pyannore.core[notebook]'")
            return None



class SlidingWindow:
    """Sliding window

    Parameters
    ----------
    duration : float > 0, optional
        Window duration, in seconds. Default is 30 ms.
    step : float > 0, optional
        Step between two consecutive position, in seconds. Default is 10 ms.
    start : float, optional
        First start position of window, in seconds. Default is 0.
    end : float > `start`, optional
        Default is infinity (ie. window keeps sliding forever)

    Examples
    --------

    >>> sw = SlidingWindow(duration, step, start)
    >>> frame_range = (a, b)
    >>> frame_range == sw.toFrameRange(sw.toSegment(*frame_range))
    ... True

    >>> segment = Segment(A, B)
    >>> new_segment = sw.toSegment(*sw.toFrameRange(segment))
    >>> abs(segment) - abs(segment & new_segment) < .5 * sw.step

    >>> sw = SlidingWindow(end=0.1)
    >>> print(next(sw))
    [ 00:00:00.000 -->  00:00:00.030]
    >>> print(next(sw))
    [ 00:00:00.010 -->  00:00:00.040]
    """

    def __init__(self, duration=0.030, step=0.010, start=0.000, end=None):

        # duration must be a float > 0
        if duration <= 0:
            raise ValueError("'duration' must be a float > 0.")
        self.__duration = duration

        # step must be a float > 0
        if step <= 0:
            raise ValueError("'step' must be a float > 0.")
        self.__step: float = step

        # start must be a float.
        self.__start: float = start

        # if end is not provided, set it to infinity
        if end is None:
            self.__end: float = np.inf
        else:
            # end must be greater than start
            if end <= start:
                raise ValueError("'end' must be greater than 'start'.")
            self.__end: float = end

        # current index of iterator
        self.__i: int = -1

    @property
    def start(self) -> float:
        """Sliding window start time in seconds."""
        return self.__start

    @property
    def end(self) -> float:
        """Sliding window end time in seconds."""
        return self.__end

    @property
    def step(self) -> float:
        """Sliding window step in seconds."""
        return self.__step

    @property
    def duration(self) -> float:
        """Sliding window duration in seconds."""
        return self.__duration

    def closest_frame(self, t: float) -> int:
        """Closest frame to timestamp.

        Parameters
        ----------
        t : float
            Timestamp, in seconds.

        Returns
        -------
        index : int
            Index of frame whose middle is the closest to `timestamp`

        """
        return int(np.rint(
            (t - self.__start - .5 * self.__duration) / self.__step
        ))

    '''
    def samples(self, from_duration: float, mode: Alignment = 'strict') -> int:
        """Number of frames

        Parameters
        ----------
        from_duration : float
            Duration in seconds.
        mode : {'strict', 'loose', 'center'}
            In 'strict' mode, computes the maximum number of consecutive frames
            that can be fitted into a segment with duration `from_duration`.
            In 'loose' mode, computes the maximum number of consecutive frames
            intersecting a segment with duration `from_duration`.
            In 'center' mode, computes the average number of consecutive frames
            where the first one is centered on the start time and the last one
            is centered on the end time of a segment with duration
            `from_duration`.

        """
        if mode == 'strict':
            return int(np.floor((from_duration - self.duration) / self.step)) + 1

        elif mode == 'loose':
            return int(np.floor((from_duration + self.duration) / self.step))

        elif mode == 'center':
            return int(np.rint((from_duration / self.step)))
    '''


    def segmentToRange(self, segment: Segment) -> Tuple[int, int]:
        warnings.warn("Deprecated in favor of `segment_to_range`",
                      DeprecationWarning)
        return self.segment_to_range(segment)

    def segment_to_range(self, segment: Segment) -> Tuple[int, int]:
        """Convert segment to 0-indexed frame range

        Parameters
        ----------
        segment : Segment

        Returns
        -------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.segment_to_range(Segment(10, 15))
            i0, n

        """
        # find closest frame to segment start
        i0 = self.closest_frame(segment.start)

        # number of steps to cover segment duration
        n = int(segment.duration / self.step) + 1

        return i0, n

    def rangeToSegment(self, i0: int, n: int) -> Segment:
        warnings.warn("This is deprecated in favor of `range_to_segment`",
                      DeprecationWarning)
        return self.range_to_segment(i0, n)

    def range_to_segment(self, i0: int, n: int) -> Segment:
        """Convert 0-indexed frame range to segment

        Each frame represents a unique segment of duration 'step', centered on
        the middle of the frame.

        The very first frame (i0 = 0) is the exception. It is extended to the
        sliding window start time.

        Parameters
        ----------
        i0 : int
            Index of first frame
        n : int
            Number of frames

        Returns
        -------
        segment : Segment

        Examples
        --------

            >>> window = SlidingWindow()
            >>> print window.range_to_segment(3, 2)
            [ --> ]

        """

        # frame start time
        # start = self.start + i0 * self.step
        # frame middle time
        # start += .5 * self.duration
        # subframe start time
        # start -= .5 * self.step
        start = self.__start + (i0 - .5) * self.__step + .5 * self.__duration
        duration = n * self.__step
        end = start + duration

        # extend segment to the beginning of the timeline
        if i0 == 0:
            start = self.start

        return Segment(start, end)

    def samplesToDuration(self, nSamples: int) -> float:
        warnings.warn("This is deprecated in favor of `samples_to_duration`",
                      DeprecationWarning)
        return self.samples_to_duration(nSamples)

    def samples_to_duration(self, n_samples: int) -> float:
        """Returns duration of samples"""
        return self.range_to_segment(0, n_samples).duration

    def durationToSamples(self, duration: float) -> int:
        warnings.warn("This is deprecated in favor of `duration_to_samples`",
                      DeprecationWarning)
        return self.duration_to_samples(duration)

    def duration_to_samples(self, duration: float) -> int:
        """Returns samples in duration"""
        return self.segment_to_range(Segment(0, duration))[1]

    def __getitem__(self, i: int) -> Segment:
        """
        Parameters
        ----------
        i : int
            Index of sliding window position

        Returns
        -------
        segment : :class:`Segment`
            Sliding window at ith position

        """

        # window start time at ith position
        start = self.__start + i * self.__step

        # in case segment starts after the end,
        # return an empty segment
        if start >= self.__end:
            return None

        return Segment(start=start, end=start + self.__duration)

    def next(self) -> Segment:
        return self.__next__()

    def __next__(self) -> Segment:
        self.__i += 1
        window = self[self.__i]

        if window:
            return window
        else:
            raise StopIteration()

    def __iter__(self) -> 'SlidingWindow':
        """Sliding window iterator

        Use expression 'for segment in sliding_window'

        Examples
        --------

        >>> window = SlidingWindow(end=0.1)
        >>> for segment in window:
        ...     print(segment)
        [ 00:00:00.000 -->  00:00:00.030]
        [ 00:00:00.010 -->  00:00:00.040]
        [ 00:00:00.020 -->  00:00:00.050]
        [ 00:00:00.030 -->  00:00:00.060]
        [ 00:00:00.040 -->  00:00:00.070]
        [ 00:00:00.050 -->  00:00:00.080]
        [ 00:00:00.060 -->  00:00:00.090]
        [ 00:00:00.070 -->  00:00:00.100]
        [ 00:00:00.080 -->  00:00:00.110]
        [ 00:00:00.090 -->  00:00:00.120]
        """

        # reset iterator index
        self.__i = -1
        return self

    def __len__(self) -> int:
        """Number of positions

        Equivalent to len([segment for segment in window])

        Returns
        -------
        length : int
            Number of positions taken by the sliding window
            (from start times to end times)

        """
        if np.isinf(self.__end):
            raise ValueError('infinite sliding window.')

        # start looking for last position
        # based on frame closest to the end
        i = self.closest_frame(self.__end)

        while (self[i]):
            i += 1
        length = i

        return length

    def copy(self) -> 'SlidingWindow':
        """Duplicate sliding window"""
        duration = self.duration
        step = self.step
        start = self.start
        end = self.end
        sliding_window = self.__class__(
            duration=duration, step=step, start=start, end=end
        )
        return sliding_window

    def __call__(self,
                 support: Union[Segment, 'Timeline'],
                 align_last: bool = False) -> Iterable[Segment]:
        """Slide window over support

        Parameter
        ---------
        support : Segment or Timeline
            Support on which to slide the window.
        align_last : bool, optional
            Yield a final segment so that it aligns exactly with end of support.

        Yields
        ------
        chunk : Segment

        Example
        -------
        >>> window = SlidingWindow(duration=2., step=1.)
        >>> for chunk in window(Segment(3, 7.5)):
        ...     print(tuple(chunk))
        (3.0, 5.0)
        (4.0, 6.0)
        (5.0, 7.0)
        >>> for chunk in window(Segment(3, 7.5), align_last=True):
        ...     print(tuple(chunk))
        (3.0, 5.0)
        (4.0, 6.0)
        (5.0, 7.0)
        (5.5, 7.5)
        """

        from pyannote.core import Timeline
        if isinstance(support, Timeline):
            segments = support

        elif isinstance(support, Segment):
            segments = Timeline(segments=[support])

        else:
            msg = (
                f'"support" must be either a Segment or a Timeline '
                f'instance (is {type(support)})'
            )
            raise TypeError(msg)

        for segment in segments:

            if segment.duration < self.duration:
                continue

            window = SlidingWindow(duration=self.duration,
                                   step=self.step,
                                   start=segment.start,
                                   end=segment.end)

            for s in window:
                # ugly hack to account for floating point imprecision
                if s in segment:
                    yield s
                    last = s

            if align_last and last.end < segment.end:
                yield Segment(start=segment.end - self.duration,
                              end=segment.end)

class SlidingWindowFeature(np.lib.mixins.NDArrayOperatorsMixin):
    """Periodic feature vectors

    Parameters
    ----------
    data : (n_frames, n_features) numpy array
    sliding_window : SlidingWindow
    labels : list, optional
        Textual description of each dimension.
    """

    def __init__(
        self, data: np.ndarray, sliding_window: SlidingWindow, labels: List[Text] = None
    ):
        self.sliding_window: SlidingWindow = sliding_window
        self.data = data
        self.labels = labels
        self.__i: int = -1

    def __len__(self):
        """Number of feature vectors"""
        return self.data.shape[0]

    @property
    def extent(self):
        return self.sliding_window.range_to_segment(0, len(self))

    @property
    def dimension(self):
        """Dimension of feature vectors"""
        return self.data.shape[1]

    def getNumber(self):
        warnings.warn("This is deprecated in favor of `__len__`", DeprecationWarning)
        return self.data.shape[0]

    def getDimension(self):
        warnings.warn(
            "This is deprecated in favor of `dimension` property", DeprecationWarning
        )
        return self.dimension

    def getExtent(self):
        warnings.warn(
            "This is deprecated in favor of `extent` property", DeprecationWarning
        )
        return self.extent

    def __getitem__(self, i: int) -> np.ndarray:
        """Get ith feature vector"""
        return self.data[i]

    def __iter__(self):
        self.__i = -1
        return self

    def __next__(self) -> Tuple[Segment, np.ndarray]:
        self.__i += 1
        try:
            return self.sliding_window[self.__i], self.data[self.__i]
        except IndexError as e:
            raise StopIteration()

    def next(self):
        return self.__next__()

    def iterfeatures(
        self, window: Optional[bool] = False
    ) -> Iterator[Union[Tuple[np.ndarray, Segment], np.ndarray]]:
        """Feature vector iterator

        Parameters
        ----------
        window : bool, optional
            When True, yield both feature vector and corresponding window.
            Default is to only yield feature vector

        """
        n_samples = self.data.shape[0]
        for i in range(n_samples):
            if window:
                yield self.data[i], self.sliding_window[i]
            else:
                yield self.data[i]


    def _repr_png_(self):
        from .notebook import MATPLOTLIB_IS_AVAILABLE, MATPLOTLIB_WARNING

        if not MATPLOTLIB_IS_AVAILABLE:
            warnings.warn(MATPLOTLIB_WARNING.format(klass=self.__class__.__name__))
            return None

        from .notebook import repr_feature

        return repr_feature(self)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array__(self) -> np.ndarray:
        return self.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use SlidingWindowFeature instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle SlidingWindowFeature objects.
            if not isinstance(x, self._HANDLED_TYPES + (SlidingWindowFeature,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x.data if isinstance(x, SlidingWindowFeature) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x.data if isinstance(x, SlidingWindowFeature) else x for x in out
            )
        data = getattr(ufunc, method)(*inputs, **kwargs)

        if type(data) is tuple:
            # multiple return values
            return tuple(
                type(self)(x, self.sliding_window, labels=self.labels) for x in data
            )
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return type(self)(data, self.sliding_window, labels=self.labels)



