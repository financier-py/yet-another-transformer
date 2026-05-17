from pathlib import Path
import torch


# Всякие пути
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
CSV_PATH = DATA_DIR / 'poems.csv'
TOKENIZER_PATH = DATA_DIR / "bpe_tokenizer.json"
MODEL_SAVE_PATH = DATA_DIR / "transformer.pth"

# Токенизатор
VOCAB_SIZE = 4000

# Гиперпараметры модельки
MAX_SEQ_LEN = 32
D_MODEL = 256
NUM_LAYERS = 3
NUM_HEADS = 4
D_FF = 512
DROPOUT = 0.1

# Параметры для обучения
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")