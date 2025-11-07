import torch.nn as nn
from lerobot.policies.customACT.configuration_customACT import ACTConfig
from lerobot.policies.customACT.history_obs_state.configuration_history_obs import HistoryObsConfig,HistoryConv1dConfig
from lerobot.policies.customACT.history_obs_state.embedding_conv1d_history_obs import HistoryConv1dEmbedding

class HistoryObsStateEmbedding(nn.Module):
    def __init__(self, act_config: ACTConfig):
        super().__init__()
        history_obs_config = HistoryObsConfig() #TODO 参数暂时直接从类生成，之后要改
        if(history_obs_config.embedding_type == "conv1d"):
            self.historyobs_embedding = HistoryConv1dEmbedding(act_config, HistoryConv1dConfig())
        else:
            pass # 其他模型
    def forward(self, x):
        out = self.historyobs_embedding(x)
        return out