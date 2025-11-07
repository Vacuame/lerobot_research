class HistoryLSTMConfig:
    input_size=6
    hidden_size=64
    num_layers=1
class HistoryObsConfig:
    embedding_type: str = 'conv1d'

class HistoryConv1dConfig:
    history_segment_num: int = 4
    history_segment_alpha: float = 0.5
    history_segment_decay: str = 'exponential'
    
