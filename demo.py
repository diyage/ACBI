from attack import Cifar10ModelForGetFeature, ACBI_PGD
import torch


model = Cifar10ModelForGetFeature(256)
"""
please make sure this model has been already trained well.
"""
x = torch.rand(size=(32, 3, 32, 32))
y = torch.randint(0, 10, size=(32, ))
"""
y is the real target
"""

clean_predict: torch.Tensor = model(x)
clean_acc = (clean_predict.argmax(dim=-1) == y).float().mean()

attacker = ACBI_PGD(
    model,
    eps=8.0/255,
    eps_step_for_attack=2.0/255,
    init_num_for_v=2,
)
x_adv = attacker(x, y)
adv_predict: torch.Tensor = model(x_adv)
adv_acc = (adv_predict.argmax(dim=-1) == y).float().mean()

print('Clean Accuracy: {:.3%}'.format(clean_acc.item()))
print('Adversarial Accuracy: {:.3%}'.format(adv_acc.item()))
