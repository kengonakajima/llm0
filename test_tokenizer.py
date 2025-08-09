import pathlib

from char_tokenizer import CharTokenizer, _demo_round_trip


def test_round_trip_small():
    text = "hello, world!\n"
    _demo_round_trip(text)


def test_vocab_from_corpus():
    corpus_path = pathlib.Path("tiny_shakespeare.txt")
    assert corpus_path.exists(), "tiny_shakespeare.txt must exist in repo root"
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    # Expect some reasonable vocab size for ascii-heavy text
    assert 60 <= tokenizer.vocab_size <= 200


def test_raises_on_oov():
    tokenizer = CharTokenizer.from_text("abc")
    try:
        tokenizer.encode("abd")
        assert False, "Expected ValueError for OOV character"
    except ValueError:
        pass
