"""Data modules."""

from typing import Optional, Set
import re
import pytorch_lightning as pl
from torch.utils import data

from .. import defaults, util
from . import collators, datasets, indexes, tsv

FEATURE_CODED = ['[ABSNUM:NI]', '[ABSNUM:PL]', '[ADVTENSE:A]', '[ADVTENSE:NA]', '[ASPECT:-]', '[ASPECT:BEG]', '[ASPECT:COM]', '[ASPECT:CUS]', '[ASPECT:DES]', '[ASPECT:HAB]', '[ASPECT:IMM]', '[ASPECT:INC]', '[ASPECT:INM]', '[ASPECT:INS]', '[ASPECT:IPFV]', '[ASPECT:NA]', '[ASPECT:OBL]', '[ASPECT:PFV]', '[ASPECT:PRG]', '[ASPECT:TER]', '[BRIBRI]', '[GUARANI]', '[MARK:NI]', '[MARK:S]', '[MARK:T]', '[MAYA]', '[MODE:ADVERS]', '[MODE:DES]', '[MODE:EXH]', '[MODE:IMP]', '[MODE:POT]', '[MODE:SUB]', '[PERSON:1_PL]', '[PERSON:1_PL_EXC]', '[PERSON:1_PL_INC]', '[PERSON:1_SI]', '[PERSON:1_SI_EXC]', '[PERSON:1_SI_INC]', '[PERSON:2_PL]', '[PERSON:2_SI]', '[PERSON:3_PL]', '[PERSON:3_SI]', '[PERSON:NA]', '[STATUS:CMP]', '[SUBTYPE:DEC]', '[SUBTYPE:INT]', '[TENSE:FUT_CER]', '[TENSE:FUT_POT]', '[TENSE:FUT_SIM]', '[TENSE:IPFV_HAB]', '[TENSE:IPFV_PROG]', '[TENSE:IPFV_REC]', '[TENSE:PAS_IMP]', '[TENSE:PAS_PLU]', '[TENSE:PAS_REC]', '[TENSE:PAS_SIM]', '[TENSE:PRE_SIM]', '[TENSE:PRF_PROG]', '[TENSE:PRF_REC]', '[TENSE:PRF_REM]', '[TRANSITIV:ITR]', '[TYPE:AFF]', '[TYPE:NEG]', '[VOICE:ACT]', '[VOICE:MID]']
SOURCE_CODED = [' ', '!', "'", '(', ')', ',', '-', '.', '1', '7', '8', 'A', 'B', 'C', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'U', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'À', 'Ñ', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'ñ', 'ò', 'ó', 'õ', 'ö', 'ù', 'ú', 'û', 'ý', 'ĩ', 'ũ', '̀', '́', '̠', 'ẽ', 'ỹ', '’']

class DataModule(pl.LightningDataModule):
    """Parses, indexes, collates and loads data."""

    parser: tsv.TsvParser
    index: indexes.Index
    batch_size: int
    collator: collators.Collator

    def __init__(
        self,
        # Paths.
        *,
        train: Optional[str] = None,
        val: Optional[str] = None,
        predict: Optional[str] = None,
        test: Optional[str] = None,
        index_path: Optional[str] = None,
        tie_embedding: bool = False,
        lower_casing: bool = False,
        # TSV parsing arguments.
        source_col: int = defaults.SOURCE_COL,
        features_col: int = defaults.FEATURES_COL,
        target_col: int = defaults.TARGET_COL,
        # String parsing arguments.
        source_sep: str = defaults.SOURCE_SEP,
        features_sep: str = defaults.FEATURES_SEP,
        target_sep: str = defaults.TARGET_SEP,
        # Collator options.
        batch_size=defaults.BATCH_SIZE,
        separate_features: bool = False,
        all_features: bool = False, # HACK
        max_source_length: int = defaults.MAX_SOURCE_LENGTH,
        max_target_length: int = defaults.MAX_TARGET_LENGTH,
        # Indexing.
        index: Optional[indexes.Index] = None,
    ):
        super().__init__()
        self.parser = tsv.TsvParser(
            source_col=source_col,
            features_col=features_col,
            target_col=target_col,
            source_sep=source_sep,
            features_sep=features_sep,
            target_sep=target_sep,
        )
        self.train = train
        self.val = val
        self.predict = predict
        self.test = test
        self.batch_size = batch_size
        self.separate_features = separate_features
        self.all_features = all_features
        self.tie_embedding = tie_embedding
        self.lower_casing = lower_casing
        self.index = index if index is not None else self._make_index()
        self.collator = collators.Collator(
            pad_idx=self.index.pad_idx,
            has_features=self.has_features,
            has_target=self.has_target,
            separate_features=separate_features,
            features_offset=(
                self.index.source_vocab_size if self.has_features else 0
            ),
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

    def _make_index(self) -> indexes.Index:
        # Computes index.
        source_vocabulary: Set[str] = set()
        features_vocabulary: Set[str] = set()
        target_vocabulary: Set[str] = set()
        if self.has_features:
            if self.has_target:
                for source, features, target in self.parser.samples(
                    self.train, self.lower_casing
                ):
                    # Add pretraining tokens
                    source, pretraining_tokens_src = self.parser.extract_ids("".join(source))
                    target, pretraining_tokens_trg = self.parser.extract_ids("".join(target))
                    source_vocabulary.update(source)
                    source_vocabulary.update(pretraining_tokens_src)
                    features_vocabulary.update(features)
                    if self.all_features:
                        features_vocabulary.update(FEATURE_CODED)
                        source_vocabulary.update(SOURCE_CODED)
                    target_vocabulary.update(target)
                    target_vocabulary.update(pretraining_tokens_trg)
            else:
                for source, features in self.parser.samples(self.train, self.lower_casing):
                    # Add pretraining tokens
                    source, pretraining_tokens_src = self.parser.extract_ids(source)
                    source_vocabulary.update(source)
                    source_vocabulary.update(pretraining_tokens_src)
                    features_vocabulary.update(features)
                    if self.all_features:
                        features_vocabulary.update(FEATURE_CODED)
                        source_vocabulary.update(SOURCE_CODED)
        elif self.has_target:
            for source, target in self.parser.samples(self.train, self.lower_casing):
                # Add pretraining tokens
                source, pretraining_tokens_src = self.parser.extract_ids(source)
                target, pretraining_tokens_trg = self.parser.extract_ids(target)
                source_vocabulary.update(source)
                source_vocabulary.update(pretraining_tokens_src)
                target_vocabulary.update(target)
                target_vocabulary.update(pretraining_tokens_trg)

        else:
            for source in self.parser.samples(self.train, self.lower_casing):
                # Add pretraining tokens
                source, pretraining_tokens_src = self.parser.extract_ids(source)
                source_vocabulary.update(source)

        if self.tie_embedding:
            target_vocabulary = source_vocabulary | target_vocabulary
            source_vocabulary = target_vocabulary

        return indexes.Index(
            source_vocabulary=sorted(source_vocabulary),
            features_vocabulary=(
                sorted(features_vocabulary) if features_vocabulary else None
            ),
            target_vocabulary=(
                sorted(target_vocabulary) if target_vocabulary else None
            ),
        )

    def log_vocabularies(self) -> None:
        """Logs this module's vocabularies."""
        util.log_info(f"Source vocabulary: {self.index.source_map.pprint()}")
        if self.has_features:
            util.log_info(
                f"Features vocabulary: {self.index.features_map.pprint()}"
            )
        if self.has_target:
            util.log_info(
                f"Target vocabulary: {self.index.target_map.pprint()}"
            )

    def write_index(self, model_dir: str, experiment: str) -> None:
        """Writes the index."""
        self.index.write(model_dir, experiment)

    @property
    def has_features(self) -> int:
        return self.parser.has_features

    @property
    def has_target(self) -> int:
        return self.parser.has_target

    @property
    def source_vocab_size(self) -> int:
        if self.separate_features:
            return self.index.source_vocab_size
        else:
            return (
                self.index.source_vocab_size + self.index.features_vocab_size
            )

    def _dataset(self, path: str, lower_casing: bool = False) -> datasets.Dataset:
        return datasets.Dataset(
            list(self.parser.samples(path, lower_casing))[1:],  # HARDCODED: Skip row
            self.index,
            self.parser
        )

    # Required API.

    def train_dataloader(self) -> data.DataLoader:
        assert self.train is not None, "no train path"
        return data.DataLoader(
            self._dataset(self.train, lower_casing=self.lower_casing),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self) -> data.DataLoader:
        assert self.val is not None, "no val path"
        return data.DataLoader(
            self._dataset(self.val),
            collate_fn=self.collator,
            batch_size=2 * self.batch_size,  # Because no gradients.
            num_workers=1,
        )

    def predict_dataloader(self) -> data.DataLoader:
        assert self.predict is not None, "no predict path"
        return data.DataLoader(
            self._dataset(self.predict),
            collate_fn=self.collator,
            batch_size=2 * self.batch_size,  # Because no gradients.
            num_workers=1,
        )

    def test_dataloader(self) -> data.DataLoader:
        assert self.test is not None, "no test path"
        return data.DataLoader(
            self._dataset(self.test),
            collate_fn=self.collator,
            batch_size=2 * self.batch_size,  # Because no gradients.
            num_workers=1,
        )
