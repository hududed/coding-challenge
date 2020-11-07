import onnxruntime as rt
import numpy as np

data=np.array([[5.4,3.9,1.7,0.4],[6.1,2.6,5.6,1.4],[5.2,2.7,3.9,1.4]])

sess = rt.InferenceSession("output/model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
print(pred_onx)
