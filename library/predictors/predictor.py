from overrides import overrides
import numpy as np
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("name-predictor")
class NamePredictor(Predictor):
    """
    Predictor that takes in a word, i.e name, and returns
    the language of that specific name.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self.model = model

    def predict(self, name: str) -> JsonDict:
        # Calculate logit scores
        # Its length is equal to the number of all available labels
        tag_logits = self.predict_json({"name": name})['tag_logits']
        # Take the index of maximum score, that is, the index of the label.
        max_id = np.argmax(tag_logits, axis=-1)
        return self.model.vocab.get_token_from_index(max_id, 'labels')

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"word": "..."}``.
        Runs the underlying model, and adds the ``"characters"`` to the output.
        """
        name = json_dict["name"]
        return self._dataset_reader.text_to_instance(name)
