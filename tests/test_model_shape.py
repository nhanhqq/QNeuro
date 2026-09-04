import torch
from src.models.model import TinyBiLSTM,count_trainable_parameters
def test_parameter_count_and_cpu_free_model_contract():
 m=TinyBiLSTM(4,latent='classical');assert count_trainable_parameters(m)==912;assert count_trainable_parameters(m)<1000;assert m(torch.randn(2,7,8)).shape==(2,4)

def test_scaled_quantum_parameter_counts():
 assert count_trainable_parameters(TinyBiLSTM(2,latent='quantum'))==894
 assert count_trainable_parameters(TinyBiLSTM(4,latent='quantum'))==920
 assert count_trainable_parameters(TinyBiLSTM(9,latent='quantum'))==985
