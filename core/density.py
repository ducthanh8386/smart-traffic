from __future__ import annotations

from core.roi import point_in_roi


class DensityEstimator:
    """Estimate traffic density inside an ROI."""

    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    PCU_WEIGHTS = {
        "motorcycle": 0.3,
        "car": 1.0,
        "bus": 2.5,
        "truck": 2.0,
    }

    def __init__(
        self,
        max_capacity: int = 30,
        normal_threshold: float = 40,
        crowded_threshold: float = 70,
        pcu_weights: dict[str, float] | None = None,
    ):
        self.max_capacity = max(int(max_capacity), 1)
        self.normal_threshold = float(normal_threshold)
        self.crowded_threshold = float(crowded_threshold)
        self.pcu_weights = pcu_weights or dict(self.PCU_WEIGHTS)

    def count_vehicles_in_roi(self, tracked_objects: list[dict], roi) -> tuple[int, list[dict]]:
        vehicles = [
            obj
            for obj in tracked_objects
            if obj["class_name"] in self.VEHICLE_CLASSES and point_in_roi(obj["center_point"], roi)
        ]
        return len(vehicles), vehicles

    def calculate_density_percent(self, current_vehicle_count: int) -> float:
        return min((current_vehicle_count / self.max_capacity) * 100.0, 100.0)

    def calculate_pcu(self, vehicles: list[dict]) -> float:
        """Calculate Passenger Car Unit (PCU) total for vehicles in ROI."""
        total_pcu = 0.0
        for obj in vehicles:
            cls_name = obj.get("class_name", "car")
            weight = self.pcu_weights.get(cls_name, 1.0)
            total_pcu += weight
        return round(total_pcu, 2)

    def calculate_pcu_density_percent(self, pcu_total: float) -> float:
        """Calculate density percentage based on PCU relative to max capacity."""
        return min((pcu_total / self.max_capacity) * 100.0, 100.0)

    def get_traffic_status(self, density_percent: float) -> str:
        if density_percent < self.normal_threshold:
            return "Binh thuong"
        if density_percent < self.crowded_threshold:
            return "Dong"
        return "Un tac"

    def get_recommendation(self, status: str) -> str:
        if status == "Un tac":
            return "De xuat keo dai den xanh them 20 giay."
        if status == "Dong":
            return "Theo doi them va chuan bi dieu chinh chu ky den."
        return "Luu luong on dinh."

    def analyze_pcu_metrics(self, vehicles: list[dict]) -> dict[str, float]:
        """Return structured PCU and vehicle composition metrics."""
        pcu_total = self.calculate_pcu(vehicles)
        pcu_density = self.calculate_pcu_density_percent(pcu_total)
        motorcycle_count = sum(1 for v in vehicles if v.get("class_name") == "motorcycle")
        motorcycle_ratio = (motorcycle_count / len(vehicles) * 100.0) if vehicles else 0.0
        return {
            "pcu_total": pcu_total,
            "pcu_density_percent": round(pcu_density, 2),
            "motorcycle_count": motorcycle_count,
            "motorcycle_ratio_percent": round(motorcycle_ratio, 2),
        }

