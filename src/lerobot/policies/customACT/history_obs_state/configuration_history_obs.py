from dataclasses import dataclass, field
import abc
from lerobot.utils.hub import HubMixin
import draccus

@dataclass
class HistoryObsConfig():
    type: str = None

@dataclass
class HistoryLSTMConfig(HistoryObsConfig):
    type: str = "lstm"
    input_size: int = 6
    hidden_size: int = 64
    num_layers: int = 1
    
@dataclass
class HistoryConv1dConfig(HistoryObsConfig):
    #这里的参数都是默认值，可以在config_customACT里改
    type: str = "conv1d"
    history_segment_num: int = 3 
    history_segment_alpha: float = 0.5
    history_segment_decay: str = 'linear'  # 'exponential' or 'linear' or None
    event_prior_weight: float = 1.0
    



# # 暂时不用
# @dataclass
# class HistoryObsConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):
#     @property
#     def type(self) -> str:
#         choice_name = self.get_choice_name(self.__class__)
#         if not isinstance(choice_name, str):
#             raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
#         return choice_name

# @dataclass
# class HistoryLSTMConfig(HistoryObsConfig):
#     input_size: int = 6
#     hidden_size: int = 64
#     num_layers: int = 1

# @dataclass
# class HistoryConv1dConfig(HistoryObsConfig):
#     history_segment_num: int = 3
#     history_segment_alpha: float = 0.5
#     history_segment_decay: str = 'lienar'  # 'exponential' or 'linear' or None
    
