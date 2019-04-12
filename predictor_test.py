# pylint: disable=no-self-use,invalid-name,unused-import
import sys
from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

# required so that our custom model + predictor + dataset reader
# will be registered by name
import library

class NameLanguageClassifierPredictor:
    def test_name(self,name):
        # Load pre-trained model
        archive = load_archive('./pre_trained/model.tar.gz')
        # Load predictor and predict the language of the name
        predictor = Predictor.from_archive(archive, 'name-predictor')
        result = predictor.predict(name)
        print(result)

if __name__ == "__main__":
    name = sys.argv[1]
    test = NameLanguageClassifierPredictor()
    test.test_name(name)