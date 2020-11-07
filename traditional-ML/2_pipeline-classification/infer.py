import onnxruntime as rt
import joblib
from numpy import load
import sklearn.pipeline


sess = rt.InferenceSession("output/model.onnx")
train_data = load("train_data.npy",allow_pickle=True)

print('---', train_data[0])
inputs = {'input': train_data[:1]}
pred_onx = sess.run(None, inputs)

print("onnx predict_proba")
print("predict", pred_onx[0])
print("predict_proba", pred_onx[1])
print("skl predict_proba")
print("predict", pipeline.predict(train_data[:1]))
print("predict_proba", pipeline.predict_proba(train_data[:1]))

