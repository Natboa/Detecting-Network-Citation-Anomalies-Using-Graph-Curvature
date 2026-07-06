"""S1.1: abstract_inverted_index reconstruction (OpenAlex ships abstracts only inverted)."""

from citecurv.data.abstracts import invert_abstract


def test_inverts_positions_to_text():
    assert invert_abstract({"Deep": [0], "learning": [1], "wins": [2]}) == "Deep learning wins"


def test_repeated_words_keep_all_positions():
    assert invert_abstract({"the": [0, 2], "cat": [1], "sat": [3]}) == "the cat the sat"


def test_none_and_empty_return_none():
    assert invert_abstract(None) is None
    assert invert_abstract({}) is None
