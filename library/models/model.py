from typing import Dict
import torch
from overrides import overrides
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Average
from torch.nn.modules import NLLLoss
from torch.nn import LogSoftmax
torch.manual_seed(1)


@Model.register('lstm-labeller')
class NamesClassifier(Model):
    """
    NamesClassifier that takes in a name and label. 
    It performs forward passes and uptades the metrics such as loss and accuracy.
    """

    def __init__(self,
                 char_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        # Initialize embedding vector.
        self.char_embeddings = char_embeddings
        # Initialize encode
        self.encoder = encoder
        # Initialize hidden-tag layer.
        # It outputs score  for wach label.
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        # Initialize the average metric.
        self.accuracy = Average()
        # it’s faster and has better numerical properties compared to Softmax
        self.m = LogSoftmax()
        # The negative log likelihood loss. It is useful to train a
        # classification problem with `C` classes
        self.loss = NLLLoss()

    @overrides
    def forward(self,
                name: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # To ignore the some sepcific indices, create a mask
        mask = get_text_field_mask(name)
        # Craete embeddings given a name
        embeddings = self.char_embeddings(name)
        # Encode the embeddings with mask
        encoder_out = self.encoder(embeddings, mask)
        # Calculate the logit scores
        tag_logits = self.hidden2tag(encoder_out)
        # Update the metrics and return output
        output = {"tag_logits": tag_logits}
        if label is not None:
            output["loss"] = self.loss(self.m(tag_logits), label)
            prediction = tag_logits.max(1)[1]
            self.accuracy(prediction.eq(label).double().mean())
        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Simply return accuracy after each pass
        return {"accuracy": float(self.accuracy.get_metric(reset))}
