from attack import Cifar10ModelForGetFeature, ACBI_PGD
import torch


model = Cifar10ModelForGetFeature(256)
x = torch.rand(size=(32, 3, 32, 32))
y = torch.randint(0, 10, size=(32, ))
attacker = ACBI_PGD(
    model,
    eps=8.0/255,
    eps_step_for_attack=2.0/255,
    init_num_for_v=2,
)
x_adv = attacker(x, y)
print(x_adv)
