import numpy as np

from core.density import DensityEstimator


def test_density_percent_is_capped() -> None:
    estimator = DensityEstimator(max_capacity=2)

    assert estimator.calculate_density_percent(3) == 100.0


def test_count_vehicles_in_roi_filters_classes() -> None:
    estimator = DensityEstimator(max_capacity=10)
    roi = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=np.int32)
    objects = [
        {"class_name": "car", "center_point": (50, 50)},
        {"class_name": "person", "center_point": (50, 50)},
        {"class_name": "truck", "center_point": (150, 50)},
    ]

    count, vehicles = estimator.count_vehicles_in_roi(objects, roi)

    assert count == 1
    assert vehicles[0]["class_name"] == "car"


def test_density_status_thresholds() -> None:
    estimator = DensityEstimator(max_capacity=10, normal_threshold=40, crowded_threshold=70)

    assert estimator.get_traffic_status(39.9) == "Binh thuong"
    assert estimator.get_traffic_status(40) == "Dong"
    assert estimator.get_traffic_status(69.9) == "Dong"
    assert estimator.get_traffic_status(70) == "Un tac"


def test_pcu_calculation_and_metrics() -> None:
    estimator = DensityEstimator(max_capacity=10)
    vehicles = [
        {"class_name": "car"},
        {"class_name": "motorcycle"},
        {"class_name": "motorcycle"},
        {"class_name": "bus"},
    ]
    # car: 1.0, motorcycle: 0.3 * 2 = 0.6, bus: 2.5 => total = 4.1 PCU
    metrics = estimator.analyze_pcu_metrics(vehicles)
    assert metrics["pcu_total"] == 4.1
    assert metrics["pcu_density_percent"] == 41.0
    assert metrics["motorcycle_count"] == 2
    assert metrics["motorcycle_ratio_percent"] == 50.0

