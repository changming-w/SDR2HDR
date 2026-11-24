import torch
from collections import OrderedDict
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.losses import build_loss
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SR6D3GANModel(SRModel):
    """优化版：增强健壮性+可读性+框架兼容性"""

    def __init__(self, opt):
        super().__init__(opt)  # 简化Python3+的super调用
        self.logger = get_root_logger()  # 全局logger，避免重复创建
        self.opt = opt  # 保存配置，方便后续复用

        # 1. 显式打印生成器日志（增强可读性）
        self._log_generator_info()

        # 2. 初始化核心属性（带默认值，避免AttributeError）
        self.log_dict = OrderedDict()
        self.net_d = None
        self.optimizer_d = None
        self.cri_gan = None
        self.gan_weight = 0.01
        self.d_train_interval = self.opt['train'].get('d_train_interval', 1)

        # 3. 训练模式下初始化GAN相关组件（加容错判断）
        if self.is_train:
            self._init_gan_components()

        self.logger.info(f"Init: All components initialized. D train interval: {self.d_train_interval} steps")

    def _log_generator_info(self):
        """单独封装生成器日志逻辑，代码更整洁"""
        try:
            gen_type = self.opt['network_g']['type']
            self.logger.info(f"Init: Generator ({gen_type}) ready (Device: {self.device})")
        except KeyError:
            self.logger.warning("Init: Generator type not found in config, but parent class has initialized it")

    def _init_gan_components(self):
        """单独封装GAN组件初始化，逻辑解耦"""
        # 初始化判别器
        try:
            self.net_d = build_network(self.opt['network_d']).to(self.device)
            self.net_d.train()
            self.logger.info(f"Init: Discriminator ({self.opt['network_d']['type']}) ready (Device: {self.device})")
        except KeyError as e:
            raise RuntimeError(f"Init: Missing 'network_d' config! {str(e)}") from e

        # 初始化GAN损失
        try:
            gan_opt = self.opt['train']['gan_opt']
            self.cri_gan = build_loss(gan_opt).to(self.device)
            self.gan_weight = gan_opt.get('loss_weight', 0.01)
            self.logger.info(f"Init: GAN loss ready (Type: {gan_opt['gan_type']}, Weight: {self.gan_weight})")
        except KeyError as e:
            raise RuntimeError(f"Init: Missing 'gan_opt' config! {str(e)}") from e

        # 初始化判别器优化器
        try:
            self.optimizer_d = self._build_d_optimizer(self.opt['train']['optim_d'])
            self.optimizers.append(self.optimizer_d)
            self.logger.info(f"Init: D optimizer ready (Type: {self.opt['train']['optim_d']['type']})")
        except KeyError as e:
            raise RuntimeError(f"Init: Missing 'optim_d' config! {str(e)}") from e

    def _build_d_optimizer(self, optim_cfg):
        """优化：参数校验+默认值完善，增强容错性"""
        if not isinstance(optim_cfg, dict):
            raise ValueError(f"Optimizer config must be dict, got {type(optim_cfg)}")

        d_params = [p for p in self.net_d.parameters() if p.requires_grad]
        optim_type = optim_cfg.get('type', 'adam').lower()  # 默认为Adam，增强容错

        # 通用参数（带合理默认值）
        lr = optim_cfg.get('lr', 1e-4)
        weight_decay = optim_cfg.get('weight_decay', 1e-5)  # 默认加少量权重衰减，防止过拟合

        # 按优化器类型创建（参数补全默认值）
        if optim_type == 'adam':
            return torch.optim.Adam(
                d_params,
                lr=lr,
                betas=optim_cfg.get('betas', (0.9, 0.999)),
                eps=optim_cfg.get('eps', 1e-8),
                weight_decay=weight_decay,
                amsgrad=optim_cfg.get('amsgrad', False)
            )
        elif optim_type == 'sgd':
            return torch.optim.SGD(
                d_params,
                lr=lr,
                momentum=optim_cfg.get('momentum', 0.9),  # SGD默认加动量
                dampening=optim_cfg.get('dampening', 0.0),
                weight_decay=weight_decay,
                nesterov=optim_cfg.get('nesterov', False)
            )
        elif optim_type == 'rmsprop':
            return torch.optim.RMSprop(
                d_params,
                lr=lr,
                alpha=optim_cfg.get('alpha', 0.99),
                eps=optim_cfg.get('eps', 1e-8),
                weight_decay=weight_decay,
                momentum=optim_cfg.get('momentum', 0.0),
                centered=optim_cfg.get('centered', False)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optim_cfg.get('type')}. Support: adam/sgd/rmsprop")

    def optimize_parameters(self, current_iter):
        self.log_dict.clear()  # 重置日志字典

        # 训练生成器（加判空，避免未启用时报错）
        if self.optimizer_g is None:
            self.logger.error("Optimize: Generator optimizer not initialized!")
            return

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        l_total = self._calc_total_loss_G()

        # GAN损失（判空保护）
        if self.cri_gan is not None and self.net_d is not None:
            fake_pred = self.net_d(self.output)
            l_gan = self.cri_gan(fake_pred, target_is_real=True)
            l_total += l_gan * self.gan_weight
            self.log_dict['l_gan'] = round(l_gan.item(), 6)  # 保留6位小数，日志更整洁

        l_total.backward()
        self.optimizer_g.step()

        # 训练判别器（多重判空，增强健壮性）
        if self.is_train and self.optimizer_d is not None and self.net_d is not None and self.cri_gan is not None:
            if current_iter % self.d_train_interval == 0:
                self._train_discriminator()
            else:
                self.log_dict.update({k: 0.0 for k in ['l_d_real', 'l_d_fake', 'l_d_total']})

        # EMA更新（判空保护）
        if hasattr(self, 'ema_decay') and self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def _train_discriminator(self):
        """单独封装判别器训练逻辑，代码解耦"""
        self.optimizer_d.zero_grad()

        # 真实样本损失
        real_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_pred, target_is_real=True, is_disc=True)

        # 生成样本损失
        fake_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_pred, target_is_real=False, is_disc=True)

        # 总损失（L2正则化，增强稳定性）
        l_d_total = (l_d_real + l_d_fake) / 2
        l_d_total.backward()
        self.optimizer_d.step()

        # 日志保留6位小数，避免科学计数法
        self.log_dict.update({
            'l_d_real': round(l_d_real.item(), 6),
            'l_d_fake': round(l_d_fake.item(), 6),
            'l_d_total': round(l_d_total.item(), 6)
        })

    def _calc_total_loss_G(self):
        """优化：日志格式化+判空保护"""
        l_total = 0.0

        # 像素损失
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix * self.opt['train']['pixel_opt'].get('loss_weight', 1.0)
            self.log_dict['l_pix'] = round(l_pix.item(), 6)

        # 感知损失
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            perceptual_weight = self.opt['train']['perceptual_opt'].get('perceptual_weight', 0.1)
            style_weight = self.opt['train']['perceptual_opt'].get('style_weight', 0.0)

            if l_percep is not None:
                l_total += l_percep * perceptual_weight
                self.log_dict['l_percep'] = round(l_percep.item(), 6)
            if l_style is not None and style_weight > 0:
                l_total += l_style * style_weight
                self.log_dict['l_style'] = round(l_style.item(), 6)

        return l_total

    def save(self, epoch, current_iter):
        """优化：保存前判空，避免报错"""
        if self.is_train and self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
            self.logger.info(f"Save: Discriminator saved (Epoch: {epoch}, Iter: {current_iter})")
        super().save(epoch, current_iter)