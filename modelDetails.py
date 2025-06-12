from utils.model_summary import model_summary
from models.complex_resnet import complex_resnet18
from data.dataloader import load_config
from models.resnet import AFResNet

# Setup model
config = load_config()
model = AFResNet(config)

# Generate summary
model_summary(model, input_size=(3, 128, 128))
