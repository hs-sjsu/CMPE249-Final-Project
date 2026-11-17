import numpy as np
def unit_checks():
    from types import SimpleNamespace
    from .metrics import evaluate, summarize
    from .data.kitti import project_lidar
    truth = np.array([[0, 10, 20], [40, 80, 81]], dtype=float)
    prediction = np.array([[123, 10, 25], [20, 80, 0]], dtype=float)
    scores = evaluate(prediction, truth)
    assert scores['all']['pixels'] == 4
    assert scores['all']['absrel'] == 0.1875
    assert scores['all']['delta1'] == 0.5
    assert np.isclose(scores['all']['rmse_m'], np.sqrt(425 / 4))
    assert [scores[k]['pixels'] for k in ['0-20 m', '20-40 m', '40-80 m']] == [1, 1, 2]
    frame = SimpleNamespace(image=np.zeros((3, 3, 3)), lidar=np.array([[0, 0, 5, 0.1], [0, 0, 2, 0.9], [0, 0, -1, 0.5]]), calibration={'P2': np.column_stack((np.eye(3), np.zeros(3))), 'R0_rect': np.eye(3), 'Tr_velo_to_cam': np.column_stack((np.eye(3), np.zeros(3)))})
    projected = project_lidar(frame)
    assert projected[0, 0] == 2 and np.count_nonzero(projected) == 1
    print('PASS: analytic metrics, threshold boundaries, projection and nearest-point collision')
