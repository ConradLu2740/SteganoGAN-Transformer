import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RobustnessConfig:
    enable: bool = True
    jpeg_quality_min: int = 50
    jpeg_quality_max: int = 90
    gaussian_noise_std_min: float = 0.01
    gaussian_noise_std_max: float = 0.05
    crop_ratio_min: float = 0.05
    crop_ratio_max: float = 0.10


@dataclass
class SteganoGANConfig:
    image_size: int = 128
    data_depth: int = 3
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 4.0
    window_size: int = 8
    patch_size: int = 4
    dropout: float = 0.1

    learning_rate_encoder: float = 1e-4
    learning_rate_decoder: float = 1e-4
    learning_rate_discriminator: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    batch_size: int = 1
    num_epochs: int = 50
    num_workers: int = 4

    lambda_image: float = 1.0
    lambda_adv: float = 0.01
    lambda_bit: float = 1.0

    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    data_path: str = "./data/datasets"
    checkpoint_path: str = "./checkpoints"
    log_dir: str = "./runs"
    seed: int = 42


def load_config(yaml_path: str) -> SteganoGANConfig:
    """从YAML文件加载配置，返回SteganoGANConfig实例"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_dict = yaml.safe_load(f)

    if yaml_dict is None:
        yaml_dict = {}

    robustness_dict = yaml_dict.pop("robustness", {})
    robustness_cfg = RobustnessConfig(**robustness_dict)

    config = SteganoGANConfig(**yaml_dict, robustness=robustness_cfg)
    return config


def save_config(config: SteganoGANConfig, yaml_path: str) -> None:
    """将SteganoGANConfig实例保存到YAML文件"""
    config_dict = asdict(config)
    os.makedirs(os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else ".", exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def update_config_from_args(config: SteganoGANConfig, args: dict) -> SteganoGANConfig:
    """用命令行参数字典覆盖配置项，返回更新后的配置"""
    for key, value in args.items():
        if value is None:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.robustness, key):
            setattr(config.robustness, key, value)
    return config
