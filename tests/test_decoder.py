"""Unit tests for the numerical core (no HTTP)."""

import itertools
import math

import numpy as np
import pytest

from app.decoder import (
    CTCBeamDecoder,
    DecodeError,
    NEG_INF,
    compute_prefix_alpha,
    is_compatible,
    prefix_alpha_scores,
)
from app.state import (
    MAX_FRAMES,
    MAX_REVISE_FRAMES,
    RevisionError,
    SessionError,
    SessionManager,
    advance,
    confirm,
    revise_frames,
    undo,
    validate_frames,
    validate_revision_rows,
    validate_vocab,
)


def lse(*values):
    values = [v for v in values if v != NEG_INF]
    if not values:
        return NEG_INF
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def brute_force(frames, text, vocab):
    """Enumerate every label path and collapse it, for tiny cases only."""
    v_size = len(vocab)
    blank_mass = NEG_INF
    label_mass = NEG_INF
    for path in itertools.product(range(v_size), repeat=len(frames)):
        collapsed = []
        previous = None
        for k in path:
            if k != 0 and k != previous:
                collapsed.append(vocab[k])
            previous = k
        if "".join(collapsed) != text:
            continue
        logp = 0.0
        for t, k in enumerate(path):
            p = frames[t][k]
            if p <= 0.0:
                logp = NEG_INF
                break
            logp += math.log(p)
        if logp == NEG_INF:
            continue
        if path[-1] == 0:
            blank_mass = lse(blank_mass, logp)
        else:
            label_mass = lse(label_mass, logp)
    return lse(blank_mass, label_mass), blank_mass, label_mass


def log_probs(frames):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(frames, dtype=np.float64))


def close(a, b, tol=1e-9):
    if a == NEG_INF or b == NEG_INF:
        return a == b
    return abs(a - b) < tol


FRAMES = [
    [0.6, 0.4, 0.0],
    [0.1, 0.8, 0.1],
    [0.5, 0.0, 0.5],
    [0.2, 0.3, 0.5],
]
VOCAB = ["<blank>", "a", "b"]


# --------------------------------------------------------------------------- decoder


def test_beam_scores_match_brute_force():
    logf = log_probs(FRAMES)
    decoder = CTCBeamDecoder(VOCAB, 16)
    beams = {b.text: b for b in decoder.advance(decoder.initial_beams(), logf)}
    for text in ["", "a", "b", "ab", "ba", "aa", "bb"]:
        expected = brute_force(FRAMES, text, VOCAB)
        beam = beams.get(text)
        if beam is None:
            assert expected[0] == NEG_INF
            continue
        assert close(beam.logp, expected[0]), text
        assert close(beam.logp_blank, expected[1]), text
        assert close(beam.logp_nonblank, expected[2]), text


def test_repeated_label_collapses_without_blank():
    # "a" over three frames with zero blank probability cannot produce "aa"
    frames = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    decoder = CTCBeamDecoder(VOCAB, 4)
    beams = decoder.advance(decoder.initial_beams(), log_probs(frames))
    assert [b.text for b in beams] == ["a"]
    assert close(beams[0].logp, 0.0)

    # introduce a blank in the middle and "aa" becomes reachable
    frames = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    beams = decoder.advance(decoder.initial_beams(), log_probs(frames))
    texts = [b.text for b in beams]
    assert "aa" in texts
    assert close(beams[texts.index("aa")].logp, 0.0)


def test_zero_probability_produces_no_nan():
    frames = [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
    decoder = CTCBeamDecoder(VOCAB, 8)
    beams = decoder.advance(decoder.initial_beams(), log_probs(frames))
    assert beams
    for beam in beams:
        assert not math.isnan(beam.logp)
        assert not math.isnan(beam.logp_blank)
        assert not math.isnan(beam.logp_nonblank)


def test_chunked_equals_per_frame():
    rng = np.random.default_rng(7)
    probs = rng.random((23, 4))
    probs /= probs.sum(axis=1, keepdims=True)
    probs[5, :] = 0.0
    probs[5, 1] = 1.0
    with np.errstate(divide="ignore"):
        logf = np.log(probs)
    decoder = CTCBeamDecoder(["<blank>", "a", "b", "c"], 5)
    one_by_one = decoder.initial_beams()
    for row in logf:
        one_by_one = decoder.step(one_by_one, row)
    whole = decoder.advance(decoder.initial_beams(), logf)
    for size in (1, 3, 7, 22):
        chunked = decoder.initial_beams()
        for start in range(0, len(logf), size):
            chunked = decoder.advance(chunked, logf[start : start + size])
        assert [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in chunked] == [
            (b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in one_by_one
        ]
    assert [(b.text, b.logp) for b in whole] == [(b.text, b.logp) for b in one_by_one]


def test_ties_use_unicode_order():
    # a symmetric frame makes "a" and "b" tie exactly
    frames = log_probs([[0.0, 0.5, 0.5]])
    decoder = CTCBeamDecoder(VOCAB, 4)
    beams = decoder.step(decoder.initial_beams(), frames[0])
    tied = [b.text for b in beams if close(b.logp, math.log(0.5))]
    assert tied == ["a", "b"]


def test_same_text_merged_before_pruning():
    # Two frames of [blank 0.6, "a" 0.4]: three paths decode to "a"
    # (blank-a 0.24, a-blank 0.24, a-a 0.16) and must be summed into one beam.
    decoder = CTCBeamDecoder(["<blank>", "a"], 3)
    beams = decoder.advance(
        decoder.initial_beams(), log_probs([[0.6, 0.4], [0.6, 0.4]])
    )
    texts = [b.text for b in beams]
    assert texts.count("a") == 1
    beam = beams[texts.index("a")]
    assert close(beam.logp, math.log(0.64))
    assert close(beam.logp_blank, math.log(0.24))
    assert close(beam.logp_nonblank, math.log(0.40))


def test_beam_width_prunes_after_merging():
    # Width 2 keeps the two best texts; equal scores fall back to Unicode order.
    decoder = CTCBeamDecoder(["<blank>", "a"], 2)
    beams = decoder.advance(
        decoder.initial_beams(), log_probs([[0.5, 0.5], [0.5, 0.5]])
    )
    assert [b.text for b in beams] == ["a", ""]
    assert close(beams[1].logp, math.log(0.25))
    assert [b.text for b in beams].count("") == 1


# ------------------------------------------------------------------- prefix scores


def test_prefix_scores_match_brute_force():
    logf = log_probs(FRAMES)
    for text, ids in [("", []), ("a", [1]), ("b", [2]), ("ab", [1, 2]), ("ba", [2, 1]), ("aa", [1, 1])]:
        got = prefix_alpha_scores(compute_prefix_alpha(logf, ids))
        expected = brute_force(FRAMES, text, VOCAB)
        assert close(got[0], expected[0]), text
        assert close(got[1], expected[1]), text
        assert close(got[2], expected[2]), text


def test_compatibility_rule():
    assert is_compatible("ab", "")
    assert is_compatible("abc", "ab")
    assert not is_compatible("a", "ab")
    assert not is_compatible("ba", "ab")


# ------------------------------------------------------------------------ sessions


def make_session(frames=FRAMES, width=4):
    manager = SessionManager()
    return manager, manager.create(VOCAB, width, frames)


def test_advance_rejects_backwards_and_past_end():
    manager, session = make_session()
    advance(session, 2)
    with pytest.raises(SessionError):
        advance(session, 1)
    with pytest.raises(SessionError):
        advance(session, len(FRAMES) + 1)
    advance(session, 2)  # same frame is a no-op
    assert session.next_frame == 2


def test_confirm_restricts_without_renormalising():
    manager, session = make_session()
    advance(session, 3)
    unrestricted = {b.text: b.logp for b in session.beams}
    target = session.beams[0].text
    confirm(session, target)
    assert session.prefix == target
    for beam in session.beams:
        assert beam.text.startswith(target)
        assert close(beam.logp, unrestricted[beam.text])
    assert session.prefix_scores(target)[0] is not None


def test_confirm_requires_compatible_text():
    manager, session = make_session()
    advance(session, 2)
    with pytest.raises(SessionError):
        confirm(session, "zzz")
    with pytest.raises(SessionError):
        confirm(session, "")


def test_undo_restores_beam_as_if_confirmation_never_happened():
    frames = [
        [0.5, 0.5, 0.0],
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6],
        [0.6, 0.2, 0.2],
        [0.3, 0.3, 0.4],
    ]
    manager, session = make_session(frames=frames, width=4)
    advance(session, 1)
    # confirm something that is not the eventual winner, to force a real replay
    confirm(session, "a")
    advance(session, 5)
    restricted = [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams]
    undo(session)
    assert session.prefix == ""
    reference = manager.create(VOCAB, 4, frames)
    advance(reference, 5)
    assert [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams] == [
        (b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in reference.beams
    ]
    assert restricted != [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams]


def test_undo_at_same_frame_restores_immediately():
    manager, session = make_session()
    advance(session, 4)
    before = [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams]
    confirm(session, session.beams[0].text)
    undo(session)
    assert [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams] == before


def test_nested_confirmations_undo_in_order():
    frames = [
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
    ]
    manager, session = make_session(frames=frames, width=6)
    advance(session, 2)
    confirm(session, "a")
    advance(session, 4)
    deepest = [(b.text, b.logp) for b in session.beams]
    if any(b.text == "aa" for b in session.beams):
        confirm(session, "aa")
    undo(session)
    if session.prefix == "a":
        assert [(b.text, b.logp) for b in session.beams] == deepest
    undo(session)
    assert session.prefix == ""
    reference = manager.create(VOCAB, 6, frames)
    advance(reference, 4)
    assert [(b.text, b.logp) for b in session.beams] == [
        (b.text, b.logp) for b in reference.beams
    ]


def test_undo_restores_beam_lost_by_restriction():
    """The hard undo case: the confirmed prefix's competitors were pruned away.

    With width 1, confirming "a" leaves only "a"; the unrestricted beam would
    have been "b" by frame 3.  Undo must bring back "b" at the current frame,
    not merely strip the prefix and keep the restricted beam.
    """
    # After confirming "b" at frame 2 with width 2, the restricted beam is
    # ["ba", "b"]; the unrestricted beam at frame 6 is ["b", "ab"].  Filtering
    # the unrestricted beam would yield ["b"], which is NOT the difficult
    # correct answer for the restricted stream.  Undo must replay from the
    # stored frame-2 state and reproduce ["b", "ab"] exactly.
    frames = [
        [0.1, 0.41, 0.49],
        [0.03, 0.13, 0.84],
        [0.06, 0.11, 0.83],
        [0.41, 0.25, 0.34],
        [0.62, 0.26, 0.12],
        [0.40, 0.34, 0.26],
    ]
    manager, session = make_session(frames=frames, width=2)
    advance(session, 2)
    assert session.beams[0].text == "b"
    confirm(session, "b")
    advance(session, 6)
    restricted = [(b.text, b.logp) for b in session.beams]
    assert [t for t, _ in restricted] == ["ba", "b"]

    undo(session)
    reference = manager.create(VOCAB, 2, frames)
    advance(reference, 6)
    # The beams must be the full unrestricted beam, scores included.
    assert [(b.text, b.logp) for b in session.beams] == [
        (b.text, b.logp) for b in reference.beams
    ]
    assert [b.text for b in session.beams] == ["b", "ab"]
    # A filter-only undo would have produced ["b"] with the restricted score.
    assert restricted[0][1] != session.beams[0].logp


def test_undo_keeps_earlier_confirmation():
    frames = [
        [0.0, 0.8, 0.2],
        [0.8, 0.1, 0.1],
        [0.0, 0.8, 0.2],
        [0.8, 0.1, 0.1],
    ]
    manager, session = make_session(frames=frames, width=4)
    advance(session, 4)
    first = session.beams[0].text
    confirm(session, first)
    after_first = [(b.text, b.logp) for b in session.beams]
    second = session.beams[0].text
    if second != first:
        confirm(session, second)
        undo(session)
    assert session.prefix == first
    assert [(b.text, b.logp) for b in session.beams] == after_first


def test_failed_operation_keeps_state():
    manager, session = make_session()
    advance(session, 2)
    snapshot = (session.prefix, session.next_frame, list(session.beams), session.revision)
    with pytest.raises(SessionError):
        confirm(session, "zzz")
    with pytest.raises(SessionError):
        undo(session)
    with pytest.raises(SessionError):
        advance(session, 99)
    assert (session.prefix, session.next_frame, list(session.beams), session.revision) == snapshot


def test_advancing_with_prefix_uses_restriction_before_pruning():
    # With width 1 and a confirmed "a", the beam must not be emptied by a frame
    # where "a" is not the top hypothesis overall.
    frames = [[0.1, 0.9, 0.0], [0.0, 0.5, 0.5], [0.0, 0.45, 0.55], [0.05, 0.05, 0.9]]
    manager, session = make_session(frames=frames, width=1)
    advance(session, 1)
    confirm(session, "a")
    advance(session, 4)
    assert session.beams
    assert all(b.text.startswith("a") for b in session.beams)


# ---------------------------------------------------------------------- validation


def test_vocab_validation():
    assert validate_vocab(["<blank>", "a"]) == ["<blank>", "a"]
    assert validate_vocab(["blank", "a"]) == ["<blank>", "a"]
    for bad in ([], ["a"], ["<blank>"], ["<blank>", "ab"], ["<blank>", "a", "a"],
                ["<blank>", "<blank>"], ["<blank>", ""], ["<blank>", "a", "b", "c", "d", "e", "f", "g", "h", "i"],
                ["<blank>", 3]):
        with pytest.raises(Exception):
            validate_vocab(bad)


def test_frames_validation():
    validate_frames([[0.5, 0.5]], 2)
    for bad in ([], [[0.5]], [[0.5, 0.5, 0.0]], [[1.5, -0.5]], [[float("nan"), 1.0]],
                [[True, False]], [["0.5", "0.5"]]):
        with pytest.raises(Exception):
            validate_frames(bad, 2)
    too_many = [[0.5, 0.5]] * (MAX_FRAMES + 1)
    with pytest.raises(Exception):
        validate_frames(too_many, 2)


def test_beam_width_bounds():
    for width in (0, 17):
        with pytest.raises(Exception):
            CTCBeamDecoder(VOCAB, width)
    CTCBeamDecoder(VOCAB, 1)
    CTCBeamDecoder(VOCAB, 16)


# ----------------------------------------------------------------- frame revision


REVISE_FRAMES = [
    [0.6, 0.4, 0.0],
    [0.1, 0.8, 0.1],
    [0.5, 0.0, 0.5],
    [0.2, 0.3, 0.5],
    [0.7, 0.2, 0.1],
]


def _beam_signature(session):
    return [(b.text, b.logp, b.logp_blank, b.logp_nonblank) for b in session.beams]


def test_identity_revision_reproduces_state_bit_for_bit():
    manager, session = make_session(frames=REVISE_FRAMES, width=4)
    advance(session, 3)
    confirm(session, session.beams[0].text)
    advance(session, 5)
    history_before = {
        f: [(b.text, b.logp) for b in session.beams_at(f)] for f in range(6)
    }
    scores_before = [
        (c.confirmed_logp, c.confirmed_logp_blank, c.confirmed_logp_nonblank)
        for c in session.confirmations
    ]

    revise_frames(session, 2, REVISE_FRAMES[2:4])

    assert session.prefix == session.confirmations[0].prefix_after
    assert len(session.revisions) == 1
    assert (session.revisions[0].start_frame, session.revisions[0].end_frame) == (2, 4)
    # live beam identical
    reference = manager.create(VOCAB, 4, REVISE_FRAMES)
    advance(reference, 3)
    confirm(reference, reference.beams[0].text)
    advance(reference, 5)
    assert _beam_signature(session) == _beam_signature(reference)
    # historical replay identical at every boundary
    for f in range(6):
        assert [(b.text, b.logp) for b in session.beams_at(f)] == history_before[f]
    # confirmation scores unchanged
    scores_after = [
        (c.confirmed_logp, c.confirmed_logp_blank, c.confirmed_logp_nonblank)
        for c in session.confirmations
    ]
    assert scores_after == scores_before


def test_revision_matches_fresh_session_built_from_patched_frames():
    manager, session = make_session(frames=REVISE_FRAMES, width=4)
    advance(session, 2)
    confirm(session, session.beams[0].text)
    advance(session, 5)

    patched = [row[:] for row in REVISE_FRAMES]
    patched[1] = [0.05, 0.9, 0.05]
    patched[2] = [0.1, 0.8, 0.1]
    revise_frames(session, 1, patched[1:3])

    assert [list(row) for row in session.probs[1:3]] == patched[1:3]
    reference = manager.create(VOCAB, 4, patched)
    advance(reference, 2)
    target = reference.beams[0].text
    # The original confirmation text must be the same one reachable on new data.
    assert target == session.confirmations[0].prefix_after
    confirm(reference, target)
    advance(reference, 5)
    assert _beam_signature(session) == _beam_signature(reference)


def test_revision_rebuilds_undo_state_and_undo_stays_exact():
    manager, session = make_session(frames=REVISE_FRAMES, width=3)
    advance(session, 2)
    confirm(session, session.beams[0].text)
    advance(session, 5)

    patched = [row[:] for row in REVISE_FRAMES]
    patched[0] = [0.2, 0.7, 0.1]
    revise_frames(session, 0, [patched[0]])

    # Undo after revision must equal an unrestricted replay on patched frames.
    undo(session)
    assert session.prefix == ""
    reference = manager.create(VOCAB, 3, patched)
    advance(reference, 5)
    assert _beam_signature(session) == _beam_signature(reference)


def test_revision_recomputes_confirmation_scores_on_new_frames():
    manager, session = make_session(frames=REVISE_FRAMES, width=4)
    advance(session, 3)
    target = session.beams[0].text
    confirm(session, target)
    old_score = session.confirmations[0].confirmed_logp

    # Make the confirmed text much more likely on an early frame: its exact
    # forward score must improve (less negative).
    revised = [[0.05, 0.9, 0.05]]
    revise_frames(session, 1, revised)
    new_score = session.confirmations[0].confirmed_logp
    assert session.confirmations[0].prefix_after == target
    assert new_score != old_score
    # And it must equal a from-scratch exact forward computation.
    total, _, _ = session.prefix_scores(target)
    assert close(total, new_score)


def test_failed_revision_reports_earliest_failure_and_keeps_session():
    manager, session = make_session(frames=REVISE_FRAMES, width=4)
    advance(session, 3)
    target = session.beams[0].text
    assert "b" in target  # "ab" with these frames
    confirm(session, target)
    advance(session, 5)

    snap = (
        session.frames.copy(),
        session.probs.copy(),
        _beam_signature(session),
        session.revision,
        len(session.confirmations),
        len(session.revisions),
        session.prefix,
    )
    # All-"a" frames covering frame 2, the only early frame emitting "b", make
    # the confirmed "ab" unreachable at its original frame-3 boundary.
    bad_rows = [[0.05, 0.95, 0.0], [0.05, 0.95, 0.0]]
    with pytest.raises(RevisionError) as exc:
        revise_frames(session, 1, bad_rows)
    details = exc.value.details
    assert details["reason"] in {"beam_lost", "confirmation_unreachable"}
    assert details["frame"] >= 1
    assert details["prefix"] == target
    assert "第 1 次失败" in str(exc.value)

    frames, probs, beams, rev, n_conf, n_rev, prefix = snap
    assert (session.frames == frames).all()
    assert (session.probs == probs).all()
    assert _beam_signature(session) == beams
    assert session.revision == rev
    assert len(session.confirmations) == n_conf
    assert len(session.revisions) == n_rev
    assert session.prefix == prefix


def test_revision_fails_when_beam_dies_from_blank_only_frames():
    manager, session = make_session(frames=REVISE_FRAMES, width=4)
    advance(session, 2)
    confirm(session, session.beams[0].text)
    advance(session, 5)
    # Frames that only allow blank are valid inputs but kill a non-empty
    # confirmed prefix once its blank-ending mass is exhausted at pruning.
    rows = [[1.0, 0.0, 0.0]] * 3
    with pytest.raises(RevisionError) as exc:
        revise_frames(session, 0, rows)
    # Either the prefix itself becomes unreachable or the restricted beam dies.
    assert exc.value.details["reason"] in {"beam_lost", "confirmation_unreachable"}
    assert session.prefix  # original session retained


def test_revision_validation_rules():
    manager, session = make_session(frames=REVISE_FRAMES, width=4)
    advance(session, 3)
    with pytest.raises(SessionError):
        revise_frames(session, 3, [[0.5, 0.4, 0.1]])  # not consumed yet
    with pytest.raises(SessionError):
        revise_frames(session, -1, [[0.5, 0.4, 0.1]])
    with pytest.raises(DecodeError):
        revise_frames(session, 0, [[0.5, 0.4]])  # wrong vocab width
    with pytest.raises(DecodeError):
        revise_frames(session, 0, [[1.2, -0.2, 0.0]])
    with pytest.raises(DecodeError):
        revise_frames(session, 0, [[0.0, 0.0, 0.0]])  # all-zero row
    with pytest.raises(DecodeError):
        validate_revision_rows([[0.5, 0.4, 0.1]] * (MAX_REVISE_FRAMES + 1), 3)
    # boundary exactly at next_frame is accepted for the last consumed frame
    revise_frames(session, 2, REVISE_FRAMES[2:3])
    assert session.revisions[-1].end_frame == 3


def test_multiple_revisions_compose_and_history_follows_new_data():
    manager, session = make_session(frames=REVISE_FRAMES, width=4)
    advance(session, 4)
    revise_frames(session, 0, [[0.2, 0.7, 0.1]])
    first = _beam_signature(session)
    revise_frames(session, 3, [[0.1, 0.1, 0.8]])
    assert len(session.revisions) == 2
    assert session.revisions[0].seq == 1 and session.revisions[1].seq == 2

    patched = [row[:] for row in REVISE_FRAMES]
    patched[0] = [0.2, 0.7, 0.1]
    patched[3] = [0.1, 0.1, 0.8]
    reference = manager.create(VOCAB, 4, patched)
    advance(reference, 4)
    assert _beam_signature(session) == _beam_signature(reference)
    assert first != _beam_signature(session)
    # peek/history replay uses the patched frames too
    assert [(b.text, b.logp) for b in session.beams_at(2)] == [
        (b.text, b.logp) for b in reference.beams_at(2)
    ]


def test_revision_with_nested_confirmations_replays_all_of_them():
    frames = [
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
    ]
    manager, session = make_session(frames=frames, width=6)
    advance(session, 2)
    confirm(session, "a")
    advance(session, 4)
    if any(b.text == "aa" for b in session.beams):
        confirm(session, "aa")
    advance(session, 5)

    revise_frames(session, 0, [frames[0]])  # identity on frame 0
    texts = [c.prefix_after for c in session.confirmations]
    assert texts and all(t.startswith("a") for t in texts)
    reference = manager.create(VOCAB, 6, frames)
    advance(reference, 2)
    confirm(reference, "a")
    advance(reference, 4)
    if any(b.text == "aa" for b in reference.beams):
        confirm(reference, "aa")
    advance(reference, 5)
    assert _beam_signature(session) == _beam_signature(reference)
    # undo chain still works after the replay
    while session.confirmations:
        undo(session)
    assert session.prefix == ""


def test_validate_revision_rows_helper():
    rows = validate_revision_rows([[0.2, 0.8], [1.0, 0.0]], 2)
    assert rows.shape == (2, 2)
    for bad in ([], [[0.5]], [[float("nan"), 1.0]], [["1", "0"]], [[True, False]]):
        with pytest.raises(DecodeError):
            validate_revision_rows(bad, 2)
