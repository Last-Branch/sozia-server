"""Tests for GlossAccumulator — RED phase (implementation does not exist yet)."""

from __future__ import annotations

import pytest

from sozia.server.fusion.gloss_accumulator import GlossAccumulator, GlossEntry

# ---------------------------------------------------------------------------
# Session ID fixtures
# ---------------------------------------------------------------------------

_SESSION = "cccccccc-0000-4000-c000-000000000001"
_SESSION_B = "dddddddd-0000-4000-d000-000000000002"

# ---------------------------------------------------------------------------
# Entry factory
# ---------------------------------------------------------------------------


def _entry(text: str, confidence: float = 0.8, timestamp_ms: int = 0, duration_ms: int = 100) -> GlossEntry:
    return GlossEntry(
        text=text,
        confidence=confidence,
        timestamp_ms=timestamp_ms,
        duration_ms=duration_ms,
    )


# ===========================================================================
# 1. push new word
# ===========================================================================


class TestPushNewWord:
    def test_push_single_word_peek_returns_it(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        assert acc.peek(_SESSION) == "YEMEK"

    def test_is_empty_false_after_push(self):
        acc = GlossAccumulator()
        assert acc.is_empty(_SESSION)
        acc.push(_SESSION, _entry("YEMEK"))
        assert not acc.is_empty(_SESSION)

    def test_push_returns_true_for_new_word(self):
        acc = GlossAccumulator()
        result = acc.push(_SESSION, _entry("YEMEK"))
        assert result is True


# ===========================================================================
# 2. push second different word
# ===========================================================================


class TestPushSecondWord:
    def test_peek_returns_space_joined_words(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION, _entry("OKUL"))
        assert acc.peek(_SESSION) == "YEMEK OKUL"

    def test_entries_count_after_two_different_words(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION, _entry("OKUL"))
        assert len(acc.entries(_SESSION)) == 2

    def test_entries_order_is_insertion_order(self):
        acc = GlossAccumulator()
        words = ["YEMEK", "OKUL", "GIT"]
        for w in words:
            acc.push(_SESSION, _entry(w))
        texts = [e.text for e in acc.entries(_SESSION)]
        assert texts == words


# ===========================================================================
# 3. dedup — lower-confidence repeat is dropped
# ===========================================================================


class TestDedupLowerConfidence:
    def test_lower_conf_duplicate_returns_false(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK", confidence=0.7))
        result = acc.push(_SESSION, _entry("YEMEK", confidence=0.6))
        assert result is False

    def test_lower_conf_duplicate_does_not_change_entry(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK", confidence=0.7))
        acc.push(_SESSION, _entry("YEMEK", confidence=0.6))
        entries = acc.entries(_SESSION)
        assert len(entries) == 1
        assert entries[0].confidence == pytest.approx(0.7)

    def test_lower_conf_duplicate_buffer_length_unchanged(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK", confidence=0.7))
        acc.push(_SESSION, _entry("YEMEK", confidence=0.6))
        assert len(acc.entries(_SESSION)) == 1


# ===========================================================================
# 4. dedup — higher-confidence repeat replaces existing
# ===========================================================================


class TestDedupHigherConfidence:
    def test_higher_conf_duplicate_returns_true(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK", confidence=0.6))
        result = acc.push(_SESSION, _entry("YEMEK", confidence=0.8))
        assert result is True

    def test_higher_conf_duplicate_updates_confidence(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK", confidence=0.6))
        acc.push(_SESSION, _entry("YEMEK", confidence=0.8))
        entries = acc.entries(_SESSION)
        assert len(entries) == 1
        assert entries[0].confidence == pytest.approx(0.8)

    def test_higher_conf_replaces_preserves_buffer_length(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK", confidence=0.6))
        acc.push(_SESSION, _entry("YEMEK", confidence=0.8))
        assert len(acc.entries(_SESSION)) == 1

    def test_dedup_only_applies_to_last_entry(self):
        """Dedup rule compares against the *last* entry, not any earlier one."""
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK", confidence=0.7))
        acc.push(_SESSION, _entry("OKUL", confidence=0.8))
        # "YEMEK" is not the last word — a new "YEMEK" is a different (new) word.
        result = acc.push(_SESSION, _entry("YEMEK", confidence=0.5))
        # Not a dedup of the last entry; buffer should grow (if under max).
        assert len(acc.entries(_SESSION)) == 3
        assert result is True


# ===========================================================================
# 5. max_words cap
# ===========================================================================


class TestMaxWordsCap:
    def test_fourth_word_returns_false_when_cap_is_three(self):
        acc = GlossAccumulator(max_words=3)
        words = ["YEMEK", "OKUL", "GIT", "EV"]
        for i, w in enumerate(words):
            result = acc.push(_SESSION, _entry(w))
            if i < 3:
                assert result is True, f"word {i+1} should be accepted"
            else:
                assert result is False, "fourth word should be rejected"

    def test_peek_contains_only_first_three_words(self):
        acc = GlossAccumulator(max_words=3)
        for w in ["YEMEK", "OKUL", "GIT", "EV"]:
            acc.push(_SESSION, _entry(w))
        assert acc.peek(_SESSION) == "YEMEK OKUL GIT"

    def test_buffer_length_does_not_exceed_max_words(self):
        acc = GlossAccumulator(max_words=2)
        for w in ["A", "B", "C", "D", "E"]:
            acc.push(_SESSION, _entry(w))
        assert len(acc.entries(_SESSION)) == 2

    def test_max_words_one_allows_only_first_entry(self):
        acc = GlossAccumulator(max_words=1)
        acc.push(_SESSION, _entry("YEMEK"))
        result = acc.push(_SESSION, _entry("OKUL"))
        assert result is False
        assert acc.peek(_SESSION) == "YEMEK"


# ===========================================================================
# 6. flush clears buffer
# ===========================================================================


class TestFlushClearsBuffer:
    def test_flush_returns_all_entries(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION, _entry("OKUL"))
        flushed = acc.flush(_SESSION)
        assert len(flushed) == 2
        assert flushed[0].text == "YEMEK"
        assert flushed[1].text == "OKUL"

    def test_is_empty_after_flush(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.flush(_SESSION)
        assert acc.is_empty(_SESSION)

    def test_peek_empty_string_after_flush(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.flush(_SESSION)
        assert acc.peek(_SESSION) == ""

    def test_push_works_again_after_flush(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.flush(_SESSION)
        acc.push(_SESSION, _entry("OKUL"))
        assert acc.peek(_SESSION) == "OKUL"


# ===========================================================================
# 7. flush on empty session
# ===========================================================================


class TestFlushOnEmpty:
    def test_flush_empty_returns_empty_list(self):
        acc = GlossAccumulator()
        result = acc.flush(_SESSION)
        assert result == []

    def test_flush_empty_does_not_raise(self):
        acc = GlossAccumulator()
        acc.flush(_SESSION)  # must not raise

    def test_is_empty_true_for_new_session(self):
        acc = GlossAccumulator()
        assert acc.is_empty(_SESSION)


# ===========================================================================
# 8. partial_id round-trip
# ===========================================================================


class TestPartialIdRoundTrip:
    def test_pop_returns_set_id(self):
        acc = GlossAccumulator()
        seg_id = "seg-uuid-1234"
        acc.set_partial_id(_SESSION, seg_id)
        result = acc.pop_partial_id(_SESSION)
        assert result == seg_id

    def test_second_pop_returns_none(self):
        acc = GlossAccumulator()
        acc.set_partial_id(_SESSION, "seg-uuid-1234")
        acc.pop_partial_id(_SESSION)
        assert acc.pop_partial_id(_SESSION) is None

    def test_pop_before_set_returns_none(self):
        acc = GlossAccumulator()
        assert acc.pop_partial_id(_SESSION) is None

    def test_set_partial_id_overwrites_previous(self):
        acc = GlossAccumulator()
        acc.set_partial_id(_SESSION, "first-id")
        acc.set_partial_id(_SESSION, "second-id")
        assert acc.pop_partial_id(_SESSION) == "second-id"


# ===========================================================================
# 9. partial_id cleared on flush
# ===========================================================================


class TestPartialIdClearedOnFlush:
    def test_pop_partial_id_none_after_flush(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.set_partial_id(_SESSION, "seg-abc")
        acc.flush(_SESSION)
        assert acc.pop_partial_id(_SESSION) is None

    def test_flush_clears_partial_id_even_when_buffer_empty(self):
        acc = GlossAccumulator()
        acc.set_partial_id(_SESSION, "seg-abc")
        acc.flush(_SESSION)
        assert acc.pop_partial_id(_SESSION) is None


# ===========================================================================
# 10. reset clears everything
# ===========================================================================


class TestReset:
    def test_reset_clears_buffer(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION, _entry("OKUL"))
        acc.reset(_SESSION)
        assert acc.is_empty(_SESSION)

    def test_reset_clears_partial_id(self):
        acc = GlossAccumulator()
        acc.set_partial_id(_SESSION, "seg-xyz")
        acc.reset(_SESSION)
        assert acc.pop_partial_id(_SESSION) is None

    def test_reset_clears_both_buffer_and_partial_id(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.set_partial_id(_SESSION, "seg-xyz")
        acc.reset(_SESSION)
        assert acc.is_empty(_SESSION)
        assert acc.pop_partial_id(_SESSION) is None

    def test_reset_unknown_session_is_noop(self):
        acc = GlossAccumulator()
        acc.reset("does-not-exist")  # must not raise

    def test_push_works_after_reset(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.reset(_SESSION)
        acc.push(_SESSION, _entry("OKUL"))
        assert acc.peek(_SESSION) == "OKUL"


# ===========================================================================
# 11. Multi-session isolation
# ===========================================================================


class TestMultiSessionIsolation:
    def test_push_to_a_does_not_affect_b(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        assert acc.is_empty(_SESSION_B)

    def test_flush_a_does_not_affect_b(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION_B, _entry("OKUL"))
        acc.flush(_SESSION)
        assert not acc.is_empty(_SESSION_B)
        assert acc.peek(_SESSION_B) == "OKUL"

    def test_reset_a_does_not_affect_b(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION_B, _entry("OKUL"))
        acc.reset(_SESSION)
        assert acc.peek(_SESSION_B) == "OKUL"

    def test_partial_id_isolated_per_session(self):
        acc = GlossAccumulator()
        acc.set_partial_id(_SESSION, "id-a")
        acc.set_partial_id(_SESSION_B, "id-b")
        assert acc.pop_partial_id(_SESSION) == "id-a"
        assert acc.pop_partial_id(_SESSION_B) == "id-b"

    def test_max_words_applies_per_session(self):
        acc = GlossAccumulator(max_words=2)
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION, _entry("OKUL"))
        # Session A is full — session B is independent.
        result = acc.push(_SESSION_B, _entry("GIT"))
        assert result is True
        assert acc.peek(_SESSION_B) == "GIT"


# ===========================================================================
# 12. entries() returns a copy
# ===========================================================================


class TestEntriesReturnsCopy:
    def test_mutating_returned_list_does_not_change_internal_state(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        acc.push(_SESSION, _entry("OKUL"))

        snapshot = acc.entries(_SESSION)
        snapshot.clear()  # mutate the returned list

        # Internal buffer must be untouched.
        assert len(acc.entries(_SESSION)) == 2

    def test_entries_returns_correct_values_after_mutation_of_snapshot(self):
        acc = GlossAccumulator()
        acc.push(_SESSION, _entry("YEMEK"))
        snapshot = acc.entries(_SESSION)
        snapshot.append(_entry("MUTATED"))

        # Buffer must still have only the original entry.
        assert acc.peek(_SESSION) == "YEMEK"

    def test_entries_empty_session_returns_empty_list(self):
        acc = GlossAccumulator()
        assert acc.entries(_SESSION) == []
