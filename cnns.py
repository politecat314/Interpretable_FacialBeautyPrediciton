import tensorflow as tf
from tensorflow.keras import layers
from keras_vggface.vggface import VGGFace
from keras_vggface.models import ResNet50_V3
from deepface.basemodels.Facenet import loadModel
from vit_keras.vit import vit_b16

class ResNet50():
    def __init__(self, input_shape=(224, 224, 3), weights="vggface"):
        self.input_shape = input_shape
        self.weights = weights
    
    def construct(self):
        if self.weights == "vggface":
            resnet = VGGFace(
                model="resnet50",
                include_top=False,
                input_shape=self.input_shape,
                pooling="avg",
                weights="vggface"
            )
            train = tf.keras.models.Model(
                resnet.get_layer("conv5_1_1x1_reduce").input,
                resnet.output
            )

        elif self.weights == "imagenet":
            resnet = tf.keras.applications.ResNet50(
                include_top=False,
                input_shape=self.input_shape,
                pooling="avg",
                weights="imagenet"
            )

            train = tf.keras.models.Model(
                resnet.get_layer("conv5_block1_1_conv").input,
                resnet.output
            )

        resnet._name = "resnet50" + "_" + self.weights

        base = tf.keras.models.Model(
            resnet.input,
            resnet.layers[-34].output
        )
        base._name = "resnet50" + "_" + self.weights
        for layer in base.layers:
            layer.trainable = False

        #v3 = ResNet50_V3()
        train._name = "resnet50_stage5" + "_" + self.weights

        self.name = resnet._name
        self.base = base
        self.train = train
        self.full = resnet
        
class VGG16():
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape

    def construct(self):
        vgg = VGGFace(
            model="vgg16",
            include_top=False,
            input_shape=self.input_shape,
            weights="vggface"
        )
        vgg._name = "vgg16"

        base = tf.keras.models.Model(
            vgg.input,
            vgg.get_layer("pool4").output
        )
        base._name = "vgg16"

        train = tf.keras.models.Model(
            vgg.get_layer("conv5_1").input,
            vgg.output
        )
        train._name = "vgg16_stage5"

        self.name = vgg._name
        self.base = base
        self.train = train
        self.full = vgg

class SENet50():
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape

    def construct(self):
        senet = VGGFace(
            model="senet50",
            include_top=False,
            input_shape=self.input_shape,
            weights="vggface"
        )
        senet._name = "senet50"

        base = tf.keras.models.Model(
            senet.input,
            senet.layers[-55].output
        )
        base._name = "senet50"

        train = tf.keras.models.Model(
            senet.layers[-54].input,
            senet.output
        )
        train._name = "senet50_stage5"

        self.name = senet._name
        self.base = base
        self.train = train
        self.full = senet

class ViT():
    def __init__(self, input_shape=(112, 112, 3)):
        self.input_shape = input_shape

    def construct(self):
        vit = vit_b16(
            image_size = (224,224),
            pretrained=True,
            include_top=False,
            pretrained_top = False,
        )
        vit._name = "vit"

        base = tf.keras.models.Model(vit.input, vit.layers[-5].output)
        train = tf.keras.models.Model(vit.layers[-4].input, vit.output)

        self.name = vit._name
        self.base = base
        self.train = train
        self.full = vit







def pool(model, size):
    input = tf.keras.layers.Input(model.input_shape[1:])
    x = model(input)
    x = tf.keras.layers.AveragePooling2D(size, name="avg_pool")(x)
    return tf.keras.models.Model(input, x)

class EfficientNetV2S():
    def __init__(self, input_shape=(112, 112, 3)):
        self.input_shape = input_shape

    def construct(self):
        efficientnet = tf.keras.models.load_model("./efficientnet.h5")
        efficientnet = tf.keras.models.Model(
            efficientnet.input,
            efficientnet.layers[-5].output
        )
        efficientnet._name = "efficientnetv2s"

        base = tf.keras.models.Model(
            efficientnet.input,
            efficientnet.layers[-245].output
        )
        base._name = "efficientnetv2s"

        train = tf.keras.models.Model(
            efficientnet.layers[-244].input,
            efficientnet.output
        )
        train._name = "efficientnetv2s_stage5"

        self.name = efficientnet._name
        self.base = base
        self.train = pool(train, (7,7))
        self.full = pool(efficientnet, (7,7))

class EfficientNetV2S2():
    def __init__(self, input_shape=(112, 112, 3)):
        self.input_shape = input_shape

    def construct(self):
        efficientnet = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=(224,224,3),
            pooling="avg",
        )
        efficientnet._name = "efficientnetv2s2"

        base = tf.keras.models.Model(
            efficientnet.input,
            efficientnet.get_layer("block5i_add").output
        )
        base._name = "efficientnetv2s2"

        train = tf.keras.models.Model(
            efficientnet.get_layer("block6a_expand_conv").input,
            efficientnet.output
        )
        train._name = "efficientnetv2s2_stage5"

        self.name = efficientnet._name
        self.base = base
        self.train = train
        self.full = efficientnet

class ConvNeXt():
    def __init__(self, input_shape=(112, 112, 3)):
        self.input_shape = input_shape

    def construct(self):
        convnext = tf.keras.applications.convnext.ConvNeXtSmall(
            model_name="convnext_small",
            include_top=False,
            weights="imagenet",
            input_shape=(224,224,3),
            pooling="avg",
        )

        base = tf.keras.models.Model(
            convnext.input,
            convnext.layers[-36].output
        )
        base._name = convnext._name

        train = tf.keras.models.Model(
            convnext.layers[-35].input,
            convnext.output
        )
        train._name = convnext._name+"_stage3"

        self.name = convnext._name
        self.base = base
        self.train = train
        self.full = convnext

class FaceNet():
    def __init__(self, input_shape=(160, 160, 3)):
        self.input_shape = input_shape

    def construct(self):
        facenet = loadModel()
        facenet = tf.keras.models.Model(
            facenet.input,
            facenet.layers[-5].output
        )
        facenet._name = "facenet"

        base = tf.keras.models.Model(
            facenet.input,
            facenet.layers[-127].output
        )
        base._name = "facenet"

        train = tf.keras.models.Model(
            facenet.layers[-126].input,
            facenet.output
        )
        train._name = "facenet_stage5"

        self.name = facenet._name
        self.base = base
        self.train = train
        self.full = facenet

class DenseNet():
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape

    def construct(self):
        densenet = tf.keras.applications.DenseNet201(include_top=False,
            input_shape=self.input_shape,
            weights="imagenet",
            pooling="avg"
        )
        densenet._name = "densenet201"

        base = tf.keras.models.Model(
            densenet.input,
            densenet.get_layer("pool4_pool").output
        )
        base._name = "densenet201"

        train = tf.keras.models.Model(
            densenet.get_layer("conv5_block1_0_bn").input,
            densenet.output
        )
        train._name = "densenet201_stage5"

        self.name = densenet._name
        self.base = base
        self.train = train
        self.full = densenet


if __name__ == "__main__":
    model = ResNet50()
    model.construct()
    print(model.train.summary())