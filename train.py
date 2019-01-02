from model_mobile import *
from generator import *
from keras.losses import *
from keras.optimizers import SGD


img_paths=path_generator(root='./data')
eval_data = image_generator(img_paths,10)

model = CSRNet_M()
model = import_weights(model)
model.summary()

sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9)
model.compile(optimizer=sgd, loss=mean_squared_error)
#model.load_weights("weights/premiere.finetuned.h5")
model.fit_generator(eval_data,epochs=2, steps_per_epoch=1000, verbose=1)
save_mod(model, "weights/premier.finetuned.1000iter.h5", "models/Model_mobile.json")