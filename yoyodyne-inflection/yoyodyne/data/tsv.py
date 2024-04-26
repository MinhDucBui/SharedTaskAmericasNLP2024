"""TSV parsing.

The TsvParser yields data from TSV files using 1-based indexing and custom
separators.
"""

import csv
import dataclasses
from typing import Iterator, List, Tuple, Union
import re
from .. import defaults

num_extra_tokens = 100
SPECIAL_TOKENS = [f"<extra_id_{i}>" for i in range(num_extra_tokens)]


class Error(Exception):
    """Module-specific exception."""

    pass


@dataclasses.dataclass
class TsvParser:
    """Streams data from a TSV file.

    Args:
        source_col (int, optional): 1-indexed column in TSV containing
            source strings.
        features_col (int, optional): 1-indexed column in TSV containing
            features strings.
        target_col (int, optional): 1-indexed column in TSV containing
            target strings.
        source_sep (str, optional): string used to split source string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
        features_sep (str, optional): string used to split features string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
        target_sep (str, optional): string used to split target string into
            symbols; an empty string indicates that each Unicode codepoint is
            its own symbol.
    """

    source_col: int = defaults.SOURCE_COL
    features_col: int = defaults.FEATURES_COL
    target_col: int = defaults.TARGET_COL
    source_sep: str = defaults.SOURCE_SEP
    features_sep: str = defaults.FEATURES_SEP
    target_sep: str = defaults.TARGET_SEP

    def __post_init__(self) -> None:
        # This is automatically called after initialization.
        if self.source_col < 1:
            raise Error(f"Out of range source column: {self.source_col}")
        if self.features_col < 0:
            raise Error(f"Out of range features column: {self.features_col}")
        if self.target_col < 0:
            raise Error(f"Out of range target column: {self.target_col}")

    @staticmethod
    def _tsv_reader(path: str) -> Iterator[str]:
        with open(path, "r") as tsv:
            #next(tsv) # HACK: Skip first row
            yield from csv.reader(tsv, delimiter="\t")

    @staticmethod
    def _get_string(row: List[str], col: int, lower_casing: bool = False) -> str:
        """Returns a string from a row by index.

        Args:
           row (List[str]): the split row.
           col (int): the column index.
        Returns:
           str: symbol from that string.
        """
        if lower_casing:
            return row[col - 1].lower()  # -1 because we're using one-based indexing.
        else:
            return row[col - 1]

    @property
    def has_features(self) -> bool:
        return self.features_col != 0

    @property
    def has_target(self) -> bool:
        return self.target_col != 0

    def samples(self, path: str, lower_casing: bool) -> Iterator[
        Union[
            List[str],
            Tuple[List[str], List[str]],
            Tuple[List[str], List[str], List[str]],
        ]
    ]:
        """Yields source, and features and/or target if available."""
        for row in self._tsv_reader(path):
            source = self.source_symbols(
                self._get_string(row, self.source_col, lower_casing)
            )
            if self.has_features:
                features = self.features_symbols(
                    self._get_string(row, self.features_col)
                )
                if self.has_target:
                    target = self.target_symbols(
                        self._get_string(row, self.target_col, lower_casing)
                    )
                    yield source, features, target
                else:
                    yield source, features
            elif self.has_target:
                target = self.target_symbols(
                    self._get_string(row, self.target_col, lower_casing)
                )
                yield source, target
            else:
                yield source

    # String parsing methods.

    @staticmethod
    def _get_symbols(string: str, sep: str) -> List[str]:
        # For pretraining tokens
        pretraining_string = split_pretraining_tokens(string)
        if pretraining_string:
            return pretraining_string
        return list(string) if not sep else string.split(sep)

    def source_symbols(self, string: str) -> List[str]:
        return self._get_symbols(string, self.source_sep)

    def features_symbols(self, string: str) -> List[str]:
        # We deliberately obfuscate these to avoid overlap with source.
        return [
            f"[{symbol}]"
            for symbol in self._get_symbols(string, self.features_sep)
        ]

    def target_symbols(self, string: str) -> List[str]:
        return self._get_symbols(string, self.target_sep)

    # Deserialization methods.

    def source_string(self, symbols: List[str]) -> str:
        return self.source_sep.join(symbols)

    def features_string(self, symbols: List[str]) -> str:
        return self.features_sep.join(
            # This indexing strips off the obfuscation.
            [symbol[1:-1] for symbol in symbols],
        )

    def target_string(self, symbols: List[str]) -> str:
        return self.target_sep.join(symbols)

    @staticmethod
    def extract_ids(string_extract):
        # Use regular expression to extract all occurrences of "<extra_id_0>"
        matches = re.findall(r'<extra_id_\d+>', string_extract)
        found_ids = []
        if matches:
            for extracted_text in matches:
                found_ids.append(extracted_text)
        modified_text = re.sub(r'<extra_id_\d+>', '', string_extract)
        return modified_text, found_ids

def split_pretraining_tokens(string):
    # For pretraining tokens
    split_strings = re.split(r'(<extra_id_\d+>)', string)
    split_strings = [part for part in split_strings if part]
    if len(split_strings) >= 2:
        elements = []
        for element in split_strings:
            if "<extra_id_" in element:
                elements.append(element)
            else:
                elements += list(element)
        return elements
    return None
