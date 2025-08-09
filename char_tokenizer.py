from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CharTokenizer:
    """A minimal character-level tokenizer.

    Builds a vocabulary from unique characters present in provided text and
    offers encode/decode between text and integer token ids.
    """

    char_to_id: Dict[str, int]
    id_to_char: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Create a tokenizer from text by extracting unique characters.

        The vocabulary is deterministically ordered by Unicode codepoint.
        """
        unique_chars = sorted(set(text))
        id_to_char: Dict[int, str] = {index: character for index, character in enumerate(unique_chars)}
        char_to_id: Dict[str, int] = {character: index for index, character in id_to_char.items()}
        return cls(char_to_id=char_to_id, id_to_char=id_to_char)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token ids.

        Raises ValueError if an out-of-vocabulary character is encountered.
        """
        token_ids: List[int] = []
        for character in text:
            if character not in self.char_to_id:
                raise ValueError(f"Character not in vocabulary: {repr(character)}")
            token_ids.append(self.char_to_id[character])
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token ids back to text.

        Raises ValueError if an id is not part of the vocabulary.
        """
        characters: List[str] = []
        for token_id in token_ids:
            if token_id not in self.id_to_char:
                raise ValueError(f"Token id not in vocabulary: {token_id}")
            characters.append(self.id_to_char[token_id])
        return "".join(characters)


def _demo_round_trip(sample_text: str) -> None:
    """Internal quick check used by the test script.

    Builds a tokenizer from sample_text then verifies encode->decode round-trip.
    """
    tokenizer = CharTokenizer.from_text(sample_text)
    token_ids = tokenizer.encode(sample_text)
    recovered = tokenizer.decode(token_ids)
    assert recovered == sample_text


if __name__ == "__main__":
    # Minimal manual run: build from tiny_shakespeare.txt and print key info
    import pathlib

    corpus_path = pathlib.Path("tiny_shakespeare.txt")
    if not corpus_path.exists():
        raise SystemExit("tiny_shakespeare.txt not found. Please place the corpus in repo root.")

    corpus_text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(corpus_text)

    # Summary
    print("[CharTokenizer]")
    print(f"Corpus length (chars): {len(corpus_text):,}")
    print(f"Vocab size (unique chars): {tokenizer.vocab_size}")

    # Show few vocab items (by character order)
    preview = list(sorted(tokenizer.char_to_id.items(), key=lambda kv: kv[0]))[:12]
    print("Sample vocab entries (char -> id):", preview)

    # Round-trip example
    sample = "To be, or not to be:"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print("Round-trip example text:", sample)
    print("Encoded ids:", encoded)
    print("Decoded back:", decoded)
