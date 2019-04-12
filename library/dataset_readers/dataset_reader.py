from typing import Iterator, Dict
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register('nameLang-dataset')
class NameLangDatasetReader(DatasetReader):
    """
    DatasetReader that takes in a dataset file containing name-language pairs,
    and helps us to create tokenized instances under hood.
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer]=None) -> None:
        super().__init__(lazy=False)
        # Index all the character tokens
        self.token_indexers = token_indexers or {
            "characters": SingleIdTokenIndexer()}

    def text_to_instance(self, name: str, label: str=None) -> Instance:
        # Create a list of tokens with characters
        tokens = [Token(ch) for ch in name]
        # Creare a text field with character tokens
        char_field = TextField(tokens, self.token_indexers)
        fields = {"name": char_field}
        if label is None:
            return Instance(fields)

        label_field = LabelField(label=label)
        fields["label"] = label_field
        # Create instance with given fields: Token array and label
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                line = line.strip().split()
                # Last item is always the label, i.e langauge 
                language = line[-1]
                # Previous tokens should be concatenated 
                # since one simply can have more than one name
                name = ' '.join(line[:-1])
                # Create instance by using the name and the label
                yield self.text_to_instance(name, language)
