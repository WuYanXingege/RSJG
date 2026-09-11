from src.data_src.scene_src.scene_eth5 import Scene_eth5


def test_eth_scene_uses_its_native_six_frame_stride(monkeypatch):
    monkeypatch.setattr(Scene_eth5, "load_scene_all", lambda self, verbose: None)

    assert Scene_eth5("eth").delta_frame == 6
    assert Scene_eth5("hotel").delta_frame == 10
    assert Scene_eth5("univ").delta_frame == 10
