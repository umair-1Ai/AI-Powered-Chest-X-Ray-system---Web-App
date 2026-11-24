# ================= CELL TO ADD TO NOTEBOOK =================
def export_gradcam_tflite(model, path='efficientnetb3_gradcam.tflite'):
    base = next(l for l in model.layers if isinstance(l, keras.Model))   # nested EfficientNetB3
    gap, bn = model.get_layer('gap'), model.get_layer('bn_head')
    d512, d256 = model.get_layer('dense_512'), model.get_layer('dense_256')
    pred_layer = model.get_layer('predictions')
    H, W = model.input_shape[1:3]

    class GradCamModule(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, H, W, 3], tf.float32, name='image')])
        def __call__(self, image):
            feats  = base(image, training=False)             # (1,h,w,C) last conv feature map
            pooled = gap(feats)                               # (1,C)
            zb  = bn(pooled, training=False)
            a1  = d512(zb)                                    # relu'd activations (dropout inactive)
            a2  = d256(a1)
            probs = pred_layer(a2)                            # (1,K) softmax
            c = tf.argmax(probs[0]); oh = tf.one_hot(c, probs.shape[-1])[None, :]
            # Hand-written backprop of probs[c] through the head (avoids tf.ReluGrad, which TFLite lacks)
            d_logit = probs * (oh - tf.reduce_sum(probs * oh, axis=-1, keepdims=True))
            d_a2 = tf.matmul(d_logit, pred_layer.kernel, transpose_b=True) * tf.cast(a2 > 0, tf.float32)
            d_a1 = tf.matmul(d_a2,   d256.kernel,        transpose_b=True) * tf.cast(a1 > 0, tf.float32)
            d_zb = tf.matmul(d_a1,   d512.kernel,        transpose_b=True)
            w = d_zb * (bn.gamma / tf.sqrt(bn.moving_variance + bn.epsilon))   # d score / d pooled
            # pooled-grads of Grad-CAM = (1/HW) * w  (same at every position); the constant cancels on normalising
            cam = tf.nn.relu(tf.reduce_sum(feats * w[:, None, None, :], axis=-1))   # (1,h,w)
            cam = cam / tf.maximum(tf.reduce_max(cam), 1e-12)
            return {'probs': probs, 'cam': cam}

    m = GradCamModule()
    conc = m.__call__.get_concrete_function()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([conc], m)
    open(path, 'wb').write(conv.convert())
    return path

gradcam_path = export_gradcam_tflite(model)
print("Saved:", gradcam_path, f"{os.path.getsize(gradcam_path)/1e6:.1f} MB")

# --- sanity check: probabilities must match the Keras model ---
it = tf.lite.Interpreter(model_path=gradcam_path); it.allocate_tensors()
x = preprocess_input(np.expand_dims(cv2.resize(cv2.cvtColor(cv2.imread(X_test[0]), cv2.COLOR_BGR2RGB), IMG_SIZE), 0)).astype(np.float32)
it.set_tensor(it.get_input_details()[0]['index'], x); it.invoke()
outs = {len(d['shape']): it.get_tensor(d['index']) for d in it.get_output_details()}
print("max prob diff vs Keras:", np.abs(outs[2] - model.predict(x, verbose=0)).max())
print("cam shape:", outs[3].shape)
