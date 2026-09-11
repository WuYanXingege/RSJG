from src.data_src.scene_src.scene_sdd import Scene_sdd


def test_sdd_geometry_only_scene_skips_full_resolution_visual_arrays():
    scene = Scene_sdd('bookstore_0', load_visual_data=False)

    assert scene.H is not None
    assert scene.H_inv is not None
    assert scene.RGB_image is None
    assert scene.semantic_map_gt is None
    assert scene.semantic_map_pred is None
