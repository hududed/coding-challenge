from sklearn.feature_extraction.text import TfidfVectorizer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import joblib

### ONNX conversion

pipeline = joblib.load("output/model.pkl")

seps = {
    TfidfVectorizer: {
        "separators": [
            ' ', '.', '\\?', ',', ';', ':', '!',
            '\\(', '\\)', '\n', '"', "'",
            "-", "\\[", "\\]", "@"
        ]
    }
}
model_onnx = convert_sklearn(
    pipeline, "tfidf",
    initial_types=[("input", StringTensorType([None, 2]))],
    options=seps, target_opset=12)

with open("output/model.onnx", "wb") as f:
    f.write(model_onnx.SerializeToString())

