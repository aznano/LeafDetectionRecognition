from recognition import InceptionV3_t
import ModelConfig

def train_inception():
    inception_model = InceptionV3_t.InceptionV3Leaf(ModelConfig.InceptionConfig_t.nb_categories)
    InceptionV3_t.train_t(inception_model)

if __name__ == "__main__":
    train_inception()