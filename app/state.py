"""Session state: streaming advance, prefix confirmation, exact undo and local
frame revision.

Rules implemented here
----------------------
1. ``advance`` is a loop of single-frame beam steps.  Because
   :meth:`CTCBeamDecoder.advance` is exactly a loop over :meth:`step`, advancing
   to frame N in one call and advancing one frame at a time produce the same
   beam, bit for bit.
2. ``confirm`` fixes a prefix at the current frame.  From then on incompatible
   hypotheses are dropped *inside* the beam step, before pruning, so the beam
   fills with compatible candidates instead of losing slots to hypotheses that
   are about to be discarded.  No score is renormalised: surviving beams keep
   their absolute log probabilities, and the blank / non-blank path mass of the
   confirmed last character is recorded from an exact CTC forward pass.
3. ``undo`` restores the beam that would exist at the **current** frame had the
   confirmation never happened.  Merely lifting the text filter is incorrect:
   the restriction changed which hypotheses survived pruning.  The correct
   state is obtained by replaying the frames consumed since the confirmation
   from the stored pre-confirmation beam, with the earlier prefix restriction.
4. ``revise_frames`` replaces at most 16 consecutive already-consumed frames
   with recomputed probability rows and then re-runs the whole stream from
   frame 0 on the new data: the beam search and every confirmation are replayed
   at their original frame numbers.  The confirmed text is preserved; current
   beams, historical replay data, confirmation scores and the stored undo
   state (``beams_before``) are all rebuilt from the replay.  If the beam dies
   or a confirmation is no longer reachable, the earliest failure is reported
   and the session is left byte-for-byte untouched.
5. Operations for one session are submitted to a single-worker executor, so
   they are applied in arrival order: advance / undo / revision issued while a
   replay is running are queued and see the post-replay state.  Responses carry
   a monotonically increasing ``revision`` so the UI can discard late replies.
"""

from __future__ import annotations

import copy
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .decoder import (
    Beam,
    CTCBeamDecoder,
    DecodeError,
    compute_prefix_alpha,
    finite_or_none,
    prefix_alpha_scores,
)


class SessionError(RuntimeError):
    """The requested operation is not valid for the session's current state."""


class RevisionError(SessionError):
    """A local frame revision could not be replayed.

    Raised only *after* the whole replay has been attempted, before any session
    field is mutated, so a failed revision leaves the original session intact.
    The structured ``details`` payload is forwarded to the HTTP 409 response.
    """

    def __init__(self, message: str, details: dict) -> None:
        super().__init__(message)
        self.details = dict(details)


MAX_FRAMES = 2000
MAX_EXTRA_TOKENS = 8
MAX_REVISE_FRAMES = 16


@dataclass
class Confirmation:
    """Everything needed to undo one confirmation exactly."""

    seq: int
    prefix_before: str
    prefix_after: str
    frame: int
    # Beam at ``frame`` with only the *earlier* confirmations applied.  Undo
    # replays the frames consumed since from this state.
    beams_before: List[Beam]
    last_char: str
    # Exact forward probabilities of the prefix before / after confirmation,
    # split by blank-ending and label-ending paths.  Never renormalised.
    prefix_logp: float
    prefix_logp_blank: float
    prefix_logp_nonblank: float
    confirmed_logp: float
    confirmed_logp_blank: float
    confirmed_logp_nonblank: float


@dataclass
class Revision:
    """One accepted local frame revision, for the on-page revision history."""

    seq: int
    start_frame: int
    end_frame: int  # exclusive boundary, i.e. frames start_frame .. end_frame-1
    frame_count: int


@dataclass
class Session:
    id: str
    decoder: CTCBeamDecoder
    frames: np.ndarray  # (T, V) natural-log probabilities, -inf at zero
    probs: np.ndarray   # (T, V) original probabilities, for display
    beams: List[Beam] = field(default_factory=list)
    next_frame: int = 0
    prefix: str = ""
    confirmations: List[Confirmation] = field(default_factory=list)
    revisions: List[Revision] = field(default_factory=list)
    revision_seq: int = 0
    seq: int = 0
    revision: int = 0
    error: Optional[str] = None
    # Single worker => operations apply in submission order.
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="ctc"),
        repr=False,
    )
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if not self.beams:
            self.beams = self.decoder.initial_beams()

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def vocab(self) -> List[str]:
        return self.decoder.vocab

    @property
    def beam_width(self) -> int:
        return self.decoder.beam_width

    # -- derived views -----------------------------------------------------

    def best_text(self) -> str:
        return self.beams[0].text if self.beams else self.prefix

    def label_ids(self, text: str) -> List[int]:
        try:
            return [self.decoder.vocab.index(ch) for ch in text]
        except ValueError as exc:
            raise SessionError(f"文本包含词表外字符：{exc}") from exc

    def prefix_scores(self, text: str) -> tuple[float, float, float]:
        """Exact CTC forward score of ``text`` at the current frame.

        Returns ``(total, blank-ending, non-blank-ending)`` log probabilities.
        This is the prefix's own probability, not a renormalised beam score.
        """
        alpha = compute_prefix_alpha(self.frames[: self.next_frame], self.label_ids(text))
        return prefix_alpha_scores(alpha)

    def state(self) -> dict:
        """JSON-safe snapshot of the session."""
        return {
            "session_id": self.id,
            "vocab": list(self.vocab),
            "beam_width": self.beam_width,
            "frame_count": self.frame_count,
            "next_frame": self.next_frame,
            "progress": self.next_frame / self.frame_count if self.frame_count else 0.0,
            "prefix": self.prefix,
            "prefix_scores": self.prefix_score_view(),
            "revision": self.revision,
            "seq": self.seq,
            "confirmed_count": len(self.confirmations),
            "best_text": self.best_text(),
            "beams": [beam_to_dict(b) for b in self.beams],
            "confirmations": [confirmation_to_dict(c) for c in self.confirmations],
            "revisions": [revision_to_dict(r) for r in self.revisions],
            "revision_count": len(self.revisions),
            "error": self.error,
        }

    def prefix_score_view(self) -> Optional[dict]:
        """Retained blank / non-blank path mass of the confirmed last character.

        Captured at confirmation time from an exact CTC forward pass over the
        frames consumed so far, and then kept as-is: extending the prefix or
        advancing frames never rescales them.  Showing the retained value is
        what lets you compare confirmations on the same absolute scale.
        """
        if not self.prefix:
            return {
                "text": "",
                "last_char": "",
                "logp": 0.0,
                "logp_blank": 0.0,
                "logp_nonblank": None,
            }
        record = self.confirmations[-1]
        # Scores for the last character of the confirmed text, both ending
        # states, retained exactly as computed at confirmation time.
        return {
            "text": self.prefix,
            "last_char": self.prefix[-1],
            "confirmed_at": record.frame,
            "logp": finite_or_none(record.confirmed_logp),
            "logp_blank": finite_or_none(record.confirmed_logp_blank),
            "logp_nonblank": finite_or_none(record.confirmed_logp_nonblank),
            "previous_logp": finite_or_none(record.prefix_logp),
            "previous_logp_blank": finite_or_none(record.prefix_logp_blank),
            "previous_logp_nonblank": finite_or_none(record.prefix_logp_nonblank),
        }

    def beams_at(self, frame: int) -> List[Beam]:
        """Reproduce the beams displayed at boundary ``frame``.

        The replay applies confirmations at the boundaries where they were
        made, so the result matches what the stream actually showed, rather
        than "unrestricted decode plus a filter".
        """
        if frame < 0 or frame > self.next_frame:
            raise SessionError("只能查看已经推进过的帧")
        schedule = {record.frame: record.prefix_after for record in self.confirmations}
        prefix = ""
        beams = self.decoder.initial_beams()
        if 0 in schedule:
            prefix = schedule[0]
            beams = self.decoder.filter_compatible(beams, prefix)
        for index in range(frame):
            beams = self.decoder.step(beams, self.frames[index], prefix=prefix)
            boundary = index + 1
            if boundary in schedule:
                prefix = schedule[boundary]
                beams = self.decoder.filter_compatible(beams, prefix)
        return beams


def beam_to_dict(beam: Beam) -> dict:
    return {
        "text": beam.text,
        "logp": float(beam.logp),
        "logp_blank": finite_or_none(beam.logp_blank),
        "logp_nonblank": finite_or_none(beam.logp_nonblank),
        "prob": float(np.exp(beam.logp)) if np.isfinite(beam.logp) else 0.0,
    }


def confirmation_to_dict(record: Confirmation) -> dict:
    return {
        "seq": record.seq,
        "prefix_before": record.prefix_before,
        "prefix_after": record.prefix_after,
        "frame": record.frame,
        "last_char": record.last_char,
        "confirmed_logp": finite_or_none(record.confirmed_logp),
        "confirmed_logp_blank": finite_or_none(record.confirmed_logp_blank),
        "confirmed_logp_nonblank": finite_or_none(record.confirmed_logp_nonblank),
    }


def revision_to_dict(record: Revision) -> dict:
    return {
        "seq": record.seq,
        "start_frame": record.start_frame,
        "end_frame": record.end_frame,
        "frame_count": record.frame_count,
    }


def _copy_beams(beams: Sequence[Beam]) -> List[Beam]:
    return [copy.copy(b) for b in beams]


# ---------------------------------------------------------------------------
# Input validation


def validate_vocab(vocab) -> List[str]:
    """Normalise a vocabulary: index 0 blank, at most 8 single characters."""
    if not isinstance(vocab, list) or not vocab:
        raise DecodeError("词表必须是非空数组")
    if len(vocab) > MAX_EXTRA_TOKENS + 1:
        raise DecodeError(f"词表最多包含 blank 及 {MAX_EXTRA_TOKENS} 个单字")
    normalised: List[str] = []
    for index, token in enumerate(vocab):
        if not isinstance(token, str):
            raise DecodeError("词表项必须是字符串")
        if index == 0:
            if token.strip().lower() not in ("<blank>", "<b>", "blank", "_"):
                raise DecodeError("第 0 项必须是 blank（可写 <blank>、<b>、blank 或 _）")
            normalised.append("<blank>")
            continue
        if token.strip().lower() in ("<blank>", "<b>", "blank", "_") or token == "":
            raise DecodeError("blank 只能出现在第 0 项，其余项不能为空")
        if any(ch.isspace() for ch in token):
            raise DecodeError(f"词表项不能含空白字符：{token!r}")
        if len(token) != 1:
            raise DecodeError(f"词表第 {index} 项必须是单个字符：{token!r}")
        normalised.append(token)
    if len(normalised) < 2:
        raise DecodeError("词表至少需要一个非 blank 字符")
    if len(set(normalised)) != len(normalised):
        raise DecodeError("词表项不能重复")
    return normalised


def validate_frames(frames, vocab_size: int) -> np.ndarray:
    """Validate frame probabilities and return a (T, V) float64 array."""
    if not isinstance(frames, list) or not frames:
        raise DecodeError("帧概率必须是非空数组")
    if len(frames) > MAX_FRAMES:
        raise DecodeError(f"帧数不能超过 {MAX_FRAMES}")
    array = np.empty((len(frames), vocab_size), dtype=np.float64)
    for t, row in enumerate(frames):
        if not isinstance(row, (list, tuple)) or len(row) != vocab_size:
            raise DecodeError(f"第 {t} 帧应有 {vocab_size} 个概率")
        for k, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DecodeError(f"第 {t} 帧第 {k} 个值不是数字")
            number = float(value)
            if not np.isfinite(number):
                raise DecodeError(f"第 {t} 帧第 {k} 个值不是有限数")
            if number < 0.0 or number > 1.0:
                raise DecodeError(f"第 {t} 帧第 {k} 个值不在 [0,1] 内")
            array[t, k] = number
    return array


def validate_revision_rows(rows, vocab_size: int) -> np.ndarray:
    """Validate the replacement probability rows of a local revision.

    Rows follow the original vocabulary order (index 0 = blank) and use the same
    probability conventions as session creation: finite numbers in [0, 1]; a row
    may sum to less than 1 (the decoder works with unnormalised per-label
    probabilities), but an all-zero row is rejected because it would kill every
    beam without carrying acoustic information.
    """
    if not isinstance(rows, list) or not rows:
        raise DecodeError("修订概率行必须是非空数组")
    if len(rows) > MAX_REVISE_FRAMES:
        raise DecodeError(f"一次至多修订连续 {MAX_REVISE_FRAMES} 帧")
    array = np.empty((len(rows), vocab_size), dtype=np.float64)
    for t, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != vocab_size:
            raise DecodeError(f"修订第 {t} 行应有 {vocab_size} 个概率（按原词表顺序）")
        for k, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DecodeError(f"修订第 {t} 行第 {k} 个值不是数字")
            number = float(value)
            if not np.isfinite(number):
                raise DecodeError(f"修订第 {t} 行第 {k} 个值不是有限数")
            if number < 0.0 or number > 1.0:
                raise DecodeError(f"修订第 {t} 行第 {k} 个值不在 [0,1] 内")
            array[t, k] = number
        if not array[t].any():
            raise DecodeError(f"修订第 {t} 行全为 0，该帧没有任何可达路径")
    return array


# ---------------------------------------------------------------------------
# Session manager


class SessionManager:
    """In-memory store of decoding sessions."""

    def __init__(self, max_sessions: int = 64) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions

    def create(self, vocab, beam_width: int, frames) -> Session:
        normalised = validate_vocab(vocab)
        probs = validate_frames(frames, len(normalised))
        with np.errstate(divide="ignore"):
            logp = np.log(probs)
        decoder = CTCBeamDecoder(normalised, beam_width)
        session = Session(id=uuid.uuid4().hex, decoder=decoder, frames=logp, probs=probs)
        with self._lock:
            while len(self._sessions) >= self.max_sessions:
                oldest = next(iter(self._sessions))
                evicted = self._sessions.pop(oldest)
                evicted.executor.shutdown(wait=False, cancel_futures=True)
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionError("会话不存在或已过期")
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            # Acquire the lock so a request that already fetched this session
            # cannot start mutating it after it has been removed.
            with session.lock:
                session.executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, session: Session, func: Callable, *args) -> Future:
        """Queue an operation on the session's ordered worker.

        The session lock is taken inside the worker so that a concurrent read
        (``GET /api/sessions/{id}``) can never observe a half-applied change.
        """

        def run():
            with session.lock:
                return func(session, *args)

        return session.executor.submit(run)


# ---------------------------------------------------------------------------
# Mutating operations.  They run on the session's single worker, in order.


def advance(session: Session, to_frame: int) -> None:
    """Consume frames up to boundary ``to_frame`` without going backwards."""
    if isinstance(to_frame, bool) or not isinstance(to_frame, int):
        raise SessionError("目标帧必须是整数")
    if to_frame < session.next_frame:
        raise SessionError("不能后退到更早的帧；撤销确认请使用 undo")
    if to_frame > session.frame_count:
        raise SessionError("目标帧超出帧数")
    if to_frame == session.next_frame:
        return
    block = session.frames[session.next_frame : to_frame]
    session.beams = session.decoder.advance(session.beams, block, prefix=session.prefix)
    session.next_frame = to_frame
    session.revision += 1
    session.error = None


def confirm(session: Session, prefix: str) -> None:
    """Extend the confirmed prefix to ``prefix`` at the current frame.

    ``prefix`` must extend the current confirmed prefix and must be present in
    at least one live hypothesis.  Scores are not renormalised.
    """
    if not isinstance(prefix, str):
        raise SessionError("确认文本必须是字符串")
    if not prefix:
        raise SessionError("确认文本不能为空")
    if not prefix.startswith(session.prefix):
        raise SessionError("确认文本必须保留已确认前缀")
    if prefix == session.prefix:
        raise SessionError("确认文本没有延长前缀")
    if any(ch not in session.decoder._token_index for ch in prefix):
        raise SessionError("确认文本包含词表外字符")
    compatible = [b for b in session.beams if b.text.startswith(prefix)]
    if not compatible:
        raise SessionError("当前候选中没有以该文本开头的路径")

    before_total, before_blank, before_nonblank = session.prefix_scores(session.prefix)
    after_total, after_blank, after_nonblank = session.prefix_scores(prefix)

    session.seq += 1
    session.confirmations.append(
        Confirmation(
            seq=session.seq,
            prefix_before=session.prefix,
            prefix_after=prefix,
            frame=session.next_frame,
            beams_before=_copy_beams(session.beams),
            last_char=prefix[-1],
            prefix_logp=before_total,
            prefix_logp_blank=before_blank,
            prefix_logp_nonblank=before_nonblank,
            confirmed_logp=after_total,
            confirmed_logp_blank=after_blank,
            confirmed_logp_nonblank=after_nonblank,
        )
    )
    session.prefix = prefix
    session.beams = compatible
    session.revision += 1
    session.error = None


def undo(session: Session) -> None:
    """Undo the latest confirmation at the current frame.

    Restoring the beam is *not* just removing the text restriction: the
    restriction changed which hypotheses survived pruning, so the frames
    consumed since the confirmation are replayed from the stored
    pre-confirmation beam, under the earlier prefix restriction.  The result is
    exactly the beam that would exist now had the confirmation never happened.
    """
    if not session.confirmations:
        raise SessionError("没有可撤销的确认")
    record = session.confirmations[-1]
    if session.next_frame < record.frame:  # pragma: no cover - no rewind path
        raise SessionError("当前帧早于确认帧，无法撤销")

    beams = _copy_beams(record.beams_before)
    if session.next_frame > record.frame:
        block = session.frames[record.frame : session.next_frame]
        beams = session.decoder.advance(beams, block, prefix=record.prefix_before)

    session.confirmations.pop()
    session.prefix = record.prefix_before
    session.beams = beams
    session.revision += 1
    session.error = None


# ---------------------------------------------------------------------------
# Local frame revision
#
# Replacing already-consumed frames invalidates every derived quantity, so the
# operation is implemented as a full deterministic replay on the patched frame
# array rather than an incremental patch:
#
#   * the beam search is re-run from frame 0 one frame at a time, applying each
#     confirmation at its original boundary, exactly as ``beams_at`` does;
#   * each confirmation's exact CTC forward scores are recomputed on the new
#     frames and its stored ``beams_before`` (the undo state) is rebuilt;
#   * the confirmed texts themselves are preserved and used as the replay
#     schedule, but the moment one of them is no longer reachable (or the
#     restricted beam dies) the replay aborts and nothing is committed.
#
# All work happens on local copies; the session is only mutated after the replay
# finishes successfully, so a failed revision leaves the original session --
# frames, beams, confirmations, undo state and revision counter -- untouched.


def _exact_scores(session: Session, frames: np.ndarray, text: str) -> tuple[float, float, float]:
    """Exact CTC forward score (total, blank, non-blank) of ``text``."""
    alpha = compute_prefix_alpha(frames, session.label_ids(text))
    return prefix_alpha_scores(alpha)


def _replay_revision(
    session: Session, new_frames: np.ndarray
) -> tuple[List[Beam], List[Confirmation]]:
    """Replay the stream on patched frames; raise :class:`RevisionError` early.

    Returns the live beam at the current frame and the rebuilt confirmation
    records (same order and texts as the original ones).
    """
    decoder = session.decoder
    # Confirmations grouped by their boundary, preserving submission order at a
    # shared boundary (a second confirmation can be issued without advancing).
    schedule: Dict[int, List[Confirmation]] = {}
    for record in session.confirmations:
        schedule.setdefault(record.frame, []).append(record)

    rebuilt: List[Confirmation] = []
    beams = decoder.initial_beams()
    prefix = ""

    def fail(kind: str, frame: int, message: str, **extra) -> None:
        details = {"reason": kind, "frame": frame, **extra}
        raise RevisionError(message, details)

    boundary_records = schedule.get(0)
    if boundary_records:
        for record in boundary_records:
            target = record.prefix_after
            compatible = decoder.filter_compatible(beams, target)
            if not compatible:
                fail(
                    "confirmation_unreachable",
                    0,
                    f"第 1 次失败：帧 0 处的确认「{target}」在新数据上不再可达，原会话已完整保留。",
                    confirmation_seq=record.seq,
                    prefix=target,
                )
            before_total, before_blank, before_nonblank = _exact_scores(
                session, new_frames[:0], record.prefix_before
            )
            after_total, after_blank, after_nonblank = _exact_scores(
                session, new_frames[:0], target
            )
            rebuilt.append(
                Confirmation(
                    seq=record.seq,
                    prefix_before=record.prefix_before,
                    prefix_after=target,
                    frame=0,
                    beams_before=_copy_beams(beams),
                    last_char=target[-1],
                    prefix_logp=before_total,
                    prefix_logp_blank=before_blank,
                    prefix_logp_nonblank=before_nonblank,
                    confirmed_logp=after_total,
                    confirmed_logp_blank=after_blank,
                    confirmed_logp_nonblank=after_nonblank,
                )
            )
            prefix = target
            beams = compatible

    for index in range(session.next_frame):
        beams = decoder.step(beams, new_frames[index], prefix=prefix)
        boundary = index + 1
        if not beams:
            fail(
                "beam_lost",
                boundary,
                f"第 1 次失败：束在帧 {boundary} 之后消失（当前前缀「{prefix}」"
                "已无任何可达路径），原会话已完整保留。",
                prefix=prefix,
            )
        boundary_records = schedule.get(boundary)
        if boundary_records:
            for record in boundary_records:
                target = record.prefix_after
                compatible = [b for b in beams if b.text.startswith(target)]
                if not compatible:
                    exact, _, _ = _exact_scores(session, new_frames[:boundary], target)
                    fail(
                        "confirmation_unreachable",
                        boundary,
                        f"第 1 次失败：帧 {boundary} 处的确认「{target}」在新数据上不再可达"
                        "（已不在束候选中），原会话已完整保留。",
                        confirmation_seq=record.seq,
                        prefix=target,
                        exact_logp=finite_or_none(exact),
                    )
                before_total, before_blank, before_nonblank = _exact_scores(
                    session, new_frames[:boundary], record.prefix_before
                )
                after_total, after_blank, after_nonblank = _exact_scores(
                    session, new_frames[:boundary], target
                )
                rebuilt.append(
                    Confirmation(
                        seq=record.seq,
                        prefix_before=record.prefix_before,
                        prefix_after=target,
                        frame=boundary,
                        beams_before=_copy_beams(beams),
                        last_char=target[-1],
                        prefix_logp=before_total,
                        prefix_logp_blank=before_blank,
                        prefix_logp_nonblank=before_nonblank,
                        confirmed_logp=after_total,
                        confirmed_logp_blank=after_blank,
                        confirmed_logp_nonblank=after_nonblank,
                    )
                )
                prefix = target
                beams = compatible
                if not beams:
                    fail(
                        "beam_lost",
                        boundary,
                        f"第 1 次失败：束在帧 {boundary} 的确认后消失，原会话已完整保留。",
                        prefix=prefix,
                    )
    return beams, rebuilt


def revise_frames(session: Session, start_frame: int, rows) -> None:
    """Replace a short run of consumed frames and replay the whole session.

    See the module-level rules: at most :data:`MAX_REVISE_FRAMES` consecutive
    frames inside ``[0, next_frame)`` may be replaced, rows follow the original
    vocabulary order, and any replay failure raises before the session is
    touched.  On success the frame arrays, live beams, confirmation records
    (including their scores and undo snapshots) and the revision history are
    updated; the confirmed text and frame progress do not change.
    """
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise SessionError("起始帧必须是整数")
    if start_frame < 0:
        raise SessionError("起始帧不能为负")
    new_rows = validate_revision_rows(rows, session.decoder.vocab_size)
    end_frame = start_frame + new_rows.shape[0]
    if start_frame >= session.next_frame:
        raise SessionError("只能修订已经推进过的帧")
    if end_frame > session.next_frame:
        raise SessionError(
            f"修订区间 [{start_frame}, {end_frame}) 超出已推进帧（当前 {session.next_frame} 帧）"
        )

    # Patch copies first; nothing in the session changes until the replay wins.
    new_frames = session.frames.copy()
    new_probs = session.probs.copy()
    with np.errstate(divide="ignore"):
        new_frames[start_frame:end_frame] = np.log(new_rows)
    new_probs[start_frame:end_frame] = new_rows

    rebuilt_beams, rebuilt_confirmations = _replay_revision(session, new_frames)

    session.frames = new_frames
    session.probs = new_probs
    session.beams = rebuilt_beams
    session.confirmations = rebuilt_confirmations
    session.revision_seq += 1
    session.revisions.append(
        Revision(
            seq=session.revision_seq,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_count=int(new_rows.shape[0]),
        )
    )
    session.revision += 1
    session.error = None
