# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

clr = joblib.load("output/model.pkl")
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("output/model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
