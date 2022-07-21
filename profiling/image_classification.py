from .proflier import Profiler

# continue from https://www.kaggle.com/code/jhoward/which-image-models-are-best
# and https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning
# add loading of Triton server here as all models use that

class Profile(Profiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Profile"
        self.description = "Profile"
