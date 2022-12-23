import torch
import torch.nn as nn
import numpy as np
from typing import *
import torch.nn.functional as F
import math
from torchattacks.attacks.fab import projection_linf
from torchattacks.attacks.fab import FAB
import time


class ForGetFeatrueModel(nn.Module):
    def __init__(self, output_multiply_when_eval=1):
        super().__init__()
        self.block1 = None  # type: nn.Sequential

        self.block2 = None  # type: nn.Sequential

        self.features = None  # type: nn.Linear

        self.output_multiply_when_eval = output_multiply_when_eval

    def init_para(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)

                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):

                if m.bias is not None:
                    m.bias.data.zero_()

    def post_processing(self, out_put):
        if isinstance(self.output_multiply_when_eval, (float, int)):
            return self.output_multiply_when_eval * out_put
        elif isinstance(self.output_multiply_when_eval, nn.Module):
            return self.output_multiply_when_eval(out_put)
        else:
            print('Have no post_processing!')

    def forward(self, x: torch.Tensor):
        x = self.block1(x)  # type:torch.Tensor
        x = x.view(size=(x.shape[0], -1))
        x = self.block2(x)  # type:torch.Tensor
        x = self.features(x)
        if self.training:
            return x
        else:
            return self.post_processing(x)

    def forward_(self, x: torch.Tensor):
        v = self.forward_head(x)
        out = self.forward_tail(v)
        return out, v

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)  # type:torch.Tensor
        x = x.view(size=(x.shape[0], -1))

        v = self.block2(x)  # type:torch.Tensor
        return v

    def forward_tail(self, v: torch.Tensor) -> torch.Tensor:
        out = self.features(v)

        if self.training:
            return out
        else:
            return self.post_processing(out)


class Cifar10ModelForGetFeature(ForGetFeatrueModel):
    def __init__(self, feature_num: int, kind_num: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(

            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

        )

        self.block2 = nn.Sequential(
            nn.Linear(4 * 4 * 256, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, feature_num, bias=True),
            nn.ReLU(),
        )

        self.features = nn.Linear(feature_num, kind_num, bias=True)
        self.init_para()



def get_decison_bound_split_w_b(
        weight: Union[torch.Tensor, np.ndarray, nn.Parameter],
        bias: Union[torch.Tensor, np.ndarray, nn.Parameter]
) -> tuple:
    def get_vector_delta(x: torch.Tensor):
        shape_length = len(x.shape)

        k = x.shape[0]
        x = x.view(k, -1)

        need_pos = []
        for i in range(k):
            for j in range(k):
                if j > i:
                    need_pos.append(True)
                else:
                    need_pos.append(False)

        a = x.unsqueeze(dim=0).expand(size=(k, *x.shape))
        b = x.unsqueeze(dim=1).expand(size=(k, *x.shape))

        c = b - a
        c = c.view(k * k, -1)
        res = c[need_pos]

        if shape_length == 1:
            return res.view(k * (k - 1) // 2, )
        else:
            return res
    if bias is None:
        if isinstance(bias, np.ndarray):
            bias = np.zeros(shape=(weight.shape[0],))
        else:
            bias = torch.zeros(size=(weight.shape[0],)).to(device=weight.device)

    if isinstance(weight, np.ndarray):
        weight = torch.from_numpy(weight)
        bias = torch.from_numpy(bias)

    weight_delta = get_vector_delta(weight)
    bias_delta = get_vector_delta(bias)

    if isinstance(weight, np.ndarray):
        return weight_delta.numpy(), bias_delta.numpy()
    else:
        return weight_delta, bias_delta


class DireactionDFD(nn.Module):
    def __init__(self, distance_type: str = 'mean', kinds_num: int = 10):
        super().__init__()

        self.__distance_type = distance_type
        self.kinds_num = kinds_num

        total_index = 0
        self.map_index = []
        for i in range(self.kinds_num):
            tmp = []
            for j in range(self.kinds_num):

                if i == j:
                    tmp.append(-1)

                elif i < j:
                    tmp.append(total_index)
                    total_index += 1
                else:

                    tmp.append(self.map_index[j][i])

            self.map_index.append(tmp)

        self.map_index = torch.tensor(self.map_index)
        self.map_index = (self.map_index[self.map_index != -1]).view(kinds_num, kinds_num - 1)  # target to line index

    def forward(self,
                feature_v: torch.Tensor,
                liner_layer: nn.Linear,
                logits: torch.Tensor,
                target: torch.Tensor,
                ):
        if feature_v.shape[0] == 0:
            return torch.tensor(0, dtype=torch.float32)

        else:

            weight, bias = liner_layer.weight, liner_layer.bias

            weight_delta, bias_delta = get_decison_bound_split_w_b(weight, bias)

            B = logits.shape[0]
            K = logits.shape[1]
            R = K - 1
            target_pos = F.one_hot(target, K).bool()  # B * K

            tmp1 = torch.matmul(feature_v, weight_delta.transpose(1, 0)) + bias_delta  # B * N
            tmp2 = torch.sum(weight_delta ** 2, dim=1)  # N,

            distances = tmp1 / tmp2.sqrt()  # B * N

            imgs_related_line_index = self.map_index[target].to(target.device)  # B * R

            distances_related = distances.gather(dim=1, index=imgs_related_line_index)  # B * R

            if self.__distance_type == 'mean':
                return distances_related.mean(dim=1).mean()
            elif self.__distance_type == 'sum':
                return distances_related.sum(dim=1).sum()
            elif self.__distance_type == 'max':
                return distances_related.max(dim=1)[0]
            elif self.__distance_type == 'min':
                return distances_related.min(dim=1)[0]
            else:
                return distances_related


class InitOnV(nn.Module):
    def __init__(self, kinds_num, *args, **kwargs):
        super().__init__()
        self.dfd = DireactionDFD('sum', kinds_num=kinds_num)
        self.loss = nn.CrossEntropyLoss()

    def __init_single_run_normal(self,
                                 adv_images: torch.Tensor,
                                 original_images: torch.Tensor
                                 ) -> torch.Tensor:

        adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            -self.eps,
            self.eps
        ).to(adv_images.device)

        tmp_delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(original_images + tmp_delta, min=0, max=1).detach()
        return adv_images

    def __init_single_run_on_v(self,
                               adv_images: torch.Tensor,
                               original_images: torch.Tensor,
                               labels: torch.Tensor,
                               ) -> torch.Tensor:
        adv_images.requires_grad_(True)

        v_adv = self.model.forward_head(adv_images)

        if isinstance(self.dfd, DireactionDFD):
            logits_adv = self.model.forward_tail(v_adv)
            distance = self.dfd(v_adv, self.model.features, logits_adv, labels)  # type: torch.Tensor
        else:
            distance = self.dfd(v_adv, self.model.features, labels)  # type: torch.Tensor

        cost = - 1.0 * np.random.randint(1, 10) * distance

        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + self.eps_step_for_init * grad.sign()
        delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(original_images + delta, min=0, max=1).detach()

        return adv_images

    def forward(self, *args, **kwargs):
        pass


class ACBI_PGD(InitOnV):
    def __init__(self,
                 model: ForGetFeatrueModel,
                 eps: float = 8.0 / 255,
                 eps_step_for_attack: float = 2.0 / 255,
                 eps_step_for_init: float = 8.0 / 255,
                 restart_num: int = 1,
                 init_num_for_v: int = 2,
                 iteration_num: int = 20,
                 kinds_num: int = 10,
                 adaptive: bool = False,
                 seed: int = 0,
                 distanc_type: str = 'min',
                 ):
        super().__init__(kinds_num)

        self.model = model
        self.eps = eps
        self.eps_step_for_attack = eps_step_for_attack
        self.eps_step_for_init = eps_step_for_init
        self.restart_num = restart_num
        self.init_num_for_v = init_num_for_v
        self.iteration_num = iteration_num
        self.kinds_num = kinds_num
        self.adaptive = adaptive

        self.seed = seed
        # self.dfd = DecisionFunctionDistanceLoss()

        assert distanc_type in ['min', 'max', 'mean', 'orig']

        self.dfd = DireactionDFD(distanc_type, kinds_num=kinds_num)

        self.loss = nn.CrossEntropyLoss()

    def __get_eps_step_for_attack(self, t, now_attack_num):
        if self.adaptive:
            return 0.5 * self.eps * (1 + math.cos((t % now_attack_num)/now_attack_num * math.pi))
        else:
            return self.eps_step_for_attack

    def __get_attack_num(self, r):
        # if you use discarding policy, this method need changed
        return self.iteration_num

    def __attack_single_run(self,
                            adv_images: torch.Tensor,
                            original_images: torch.Tensor,
                            labels: torch.Tensor,
                            t: int,
                            attack_num: int,
                            ) -> torch.Tensor:

        adv_images.requires_grad = True

        outputs = self.model(adv_images)

        cost = self.loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        eps_step_for_attack = self.__get_eps_step_for_attack(t, attack_num)

        adv_images = adv_images.detach() + eps_step_for_attack * grad.sign()
        delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(original_images + delta, min=0, max=1).detach()

        return adv_images

    def __init_single_run_normal(self,
                                 adv_images: torch.Tensor,
                                 original_images: torch.Tensor
                                 ) -> torch.Tensor:

        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps,
                                                                        self.eps).to(
            adv_images.device)

        tmp_delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(original_images + tmp_delta, min=0, max=1).detach()
        return adv_images

    def __init_single_run_on_v(self,
                               adv_images: torch.Tensor,
                               original_images: torch.Tensor,
                               labels: torch.Tensor,
                               ) -> torch.Tensor:
        adv_images.requires_grad_(True)

        v_adv = self.model.forward_head(adv_images)

        if isinstance(self.dfd, DireactionDFD):
            logits_adv = self.model.forward_tail(v_adv)
            distance = self.dfd(v_adv, self.model.features, logits_adv, labels)  # type: torch.Tensor
        else:
            distance = self.dfd(v_adv, self.model.features, labels)  # type: torch.Tensor

        # cost = - 1.0 * np.random.randint(1, 10) * distance.sum()
        cost = - 1.0 * distance.sum()

        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + self.eps_step_for_init * grad.sign()
        delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(original_images + delta, min=0, max=1).detach()

        return adv_images

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # easy version
        # just need compare the accuracy after attack, then return the lowest one(and x_adv)

        # most powerful version
        self.model.eval()
        x_adv_best = x.clone()
        original_images = x.clone()

        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)

        need_attack_pos = torch.ones(size=(x.shape[0],)).bool().to(x.device)

        for r in range(self.restart_num):

            # init on the input space(it's important)
            adv_images = self.__init_single_run_normal(x.clone(), original_images)

            # init on the v space
            for _ in range(self.init_num_for_v):
                adv_images = self.__init_single_run_on_v(adv_images, original_images, y)

            attack_num = self.__get_attack_num(r)

            for t in range(attack_num):

                adv_images = self.__attack_single_run(adv_images, original_images, y, t, attack_num)

                right_pos_curr = (self.model(adv_images).argmax(dim=-1) == y).float().bool()

                attack_success = ~right_pos_curr

                need_attack_pos[attack_success] = False

                x_adv_best[attack_success] = adv_images[attack_success]

        return x_adv_best


class ACBI_FAB(FAB):
    def __init__(self,
                 model,
                 eps=None,
                 restart_num: int = 1,
                 init_num_for_v: int = 1,
                 iteration_num: int = 20,
                 kinds_num: int = 10,
                 alpha_max=0.1,
                 eta=1.05,
                 beta=0.9,
                 seed=0,
                 distanc_type: str = 'min',
                 ):

        super().__init__(
            model, 'Linf', eps, iteration_num,
            restart_num, alpha_max, eta, beta, False, seed,
            False, kinds_num
        )

        self.eps_step_for_init = eps
        self.init_num_for_v = init_num_for_v

        assert distanc_type in ['min', 'max', 'mean', 'orig']

        self.dfd = DireactionDFD(distanc_type, kinds_num=kinds_num)

        self.loss = nn.CrossEntropyLoss()

    def __init_single_run_normal(self,
                                 adv_images: torch.Tensor,
                                 original_images: torch.Tensor
                                 ) -> torch.Tensor:

        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps,
                                                                        self.eps).to(
            adv_images.device)

        tmp_delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(original_images + tmp_delta, min=0, max=1).detach()
        return adv_images

    def __init_single_run_on_v(self,
                               adv_images: torch.Tensor,
                               original_images: torch.Tensor,
                               labels: torch.Tensor,
                               ) -> torch.Tensor:
        adv_images.requires_grad_(True)

        v_adv = self.model.forward_head(adv_images)

        if isinstance(self.dfd, DireactionDFD):
            logits_adv = self.model.forward_tail(v_adv)
            distance = self.dfd(v_adv, self.model.features, logits_adv, labels)  # type: torch.Tensor
        else:
            distance = self.dfd(v_adv, self.model.features, labels)  # type: torch.Tensor

        cost = - 1.0 * distance.sum()
        # cost = - 1.0 * np.random.randint(1, 10) * distance.sum()
        # print(-cost.item())
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + self.eps_step_for_init * grad.sign()
        delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(original_images + delta, min=0, max=1).detach()

        return adv_images

    def perturb(self, x, y):
        adv = x.clone()
        with torch.no_grad():
            acc = self.model(x).max(1)[1] == y

            startt = time.time()

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not self.targeted:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()

                        x_adv = x_to_fool.clone()

                        adv_curr = self.attack_single_run(x_adv, y_to_fool, use_rand_start=(counter > 0))

                        acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool

                        if self.norm == 'Linf':
                            res = (x_to_fool - adv_curr).abs().view(x_to_fool.shape[0], -1).max(1)[0]
                        elif self.norm == 'L2':
                            res = ((x_to_fool - adv_curr) ** 2).view(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()
                        acc_curr = torch.max(acc_curr, res > self.eps)

                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), self.eps, time.time() - startt))

            else:
                print('has no targeted attack!')

        return adv

    def attack_single_run(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """
        # self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)
        # assert next(self.model.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:

            if use_rand_start:
                if self.norm == 'Linf':
                    if self.init_num_for_v == 0:
                        t = 2 * torch.rand(x1.shape).to(self.device) - 1
                        x1 = im2 + (torch.min(res2,
                                              self.eps * torch.ones(res2.shape)
                                              .to(self.device)
                                              ).reshape([-1, *[1] * self.ndims])
                                    ) * t / (t.reshape([t.shape[0], -1]).abs()
                                             .max(dim=1, keepdim=True)[0]
                                             .reshape([-1, *[1] * self.ndims])) * .5
                        x1 = x1.clamp(0.0, 1.0)
                    else:

                        x1 = self.__init_single_run_normal(x1.clone(), im2)
                        for _ in range(self.init_num_for_v):
                            with torch.enable_grad():
                                x1 = self.__init_single_run_on_v(x1.clone(), im2, la2)

            counter_iter = 0
            while counter_iter < self.steps:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch(x1, la2)
                    if self.norm == 'Linf':
                        dist1 = df.abs() / (1e-12 +
                                            dg.abs()
                                            .view(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1))
                    else:
                        raise ValueError('norm not supported')

                    ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1).sum(dim=-1))
                    w = dg2.reshape([bs, -1])

                    if self.norm == 'Linf':
                        d3 = projection_linf(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))

                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                    if self.norm == 'Linf':
                        a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                            .view(-1, *[1]*self.ndims)
                    a0 = torch.max(a0, 1e-8 * torch.ones(
                        a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(torch.max(a1 / (a1 + a2),
                                                torch.zeros(a1.shape)
                                                .to(self.device)),
                                      self.alpha_max * torch.ones(a1.shape)
                                      .to(self.device))
                    x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        if self.norm == 'Linf':
                            t = (x1[ind_adv] - im2[ind_adv]).reshape(
                                [ind_adv.shape[0], -1]).abs().max(dim=1)[0]

                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                            float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.float().sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c


if __name__ == '__main__':
    m = Cifar10ModelForGetFeature(256)
    x = torch.rand(size=(8, 3, 32, 32))
    y = torch.randint(0, 10, size=(8, ))
    attacker = ACBI_FAB(m, eps=8.0/255)
    x_adv = attacker(x, y)
    print(x_adv.shape)