from __future__ import annotations

from ptr.config import load_config


def test_base_config_has_mtp_and_e6():
    cfg = load_config("configs/base.yaml")
    assert cfg.model.mtp_weight == 0.3
    assert cfg.model.router_init_std == 0.02
    assert "e6" in cfg.data
    assert cfg.data["e6"]["steps"] == 300
