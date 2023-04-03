from .model_template import ModelTemplate
from .demo_model import DemoModel
from .ctf_model import CTFModel
from .kt_NEXT_model import kt_NEXT_Model
__all__ = {
    'ModelTemplate': ModelTemplate,
    'DemoModel': DemoModel,
    'CTFModel': CTFModel,
    'kt_NEXT_Model': kt_NEXT_Model
}

def build_model(model_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, dataset=dataset
    )

    return model
