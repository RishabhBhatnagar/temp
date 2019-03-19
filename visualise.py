from keras.models import load_model, Model
from ml_nn import *


def generate_encapsulate_model_with_output_layer_names(model, output_layer_names):
    enc_model = Model(
        inputs=model.input,
        outputs=list(map(
            lambda oln: model.get_layer(oln).output,
            output_layer_names
        ))
    )
    return enc_model






lstm_model_name = Constants.lstm_model_name
gensim_model_name = Constants.gensim_model_name
print()
lstm_model = load_model(lstm_model_name)
output_layer_names = [layer.name for layer in lstm_model.layers]
enc_model = generate_encapsulate_model_with_output_layer_names(
    lstm_model, 
    output_layer_names
)
# enc_model.save("./enc_keras_model.h5")


print(predict_single_essay("hello i am rishabh bhatnagar, this is computer engineering course", Constants.gensim_model_name, Constants.lstm_model_name, lstm_model=enc_model))

