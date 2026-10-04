"""Frigate AXEngine detector; filename avoids shadowing the axengine runtime."""
from typing import Literal
from pydantic import ConfigDict, Field
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from frigate.util.model import post_process_yolo
from radxa_frigate.runtime import ModelSession
from radxa_frigate.yolo26 import decode

DETECTOR_KEY = "axengine"


class AxengineDetectorConfig(BaseDetectorConfig):
    """AX-M1 PCIe inference with validated local model metadata."""
    model_config = ConfigDict(title="AX-M1 AXCL NPU")
    type: Literal[DETECTOR_KEY]
    manifest: str = Field(default="/models/manifest.json", title="AXCL model manifest")
    axcl_model: Literal["yolov9", "yolo26n", "yolo26s"] = Field(default="yolov9", title="AXCL model profile")


class Axengine(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, config):
        super().__init__(config)
        self.width, self.height = config.model.width, config.model.height
        size = 320 if config.axcl_model == "yolov9" else 640
        if (self.width, self.height) != (size, size):
            raise ValueError("Frigate model dimensions must match the selected AXCL profile")
        self.model_id = config.axcl_model
        self.session = ModelSession(config.manifest, config.axcl_model, kind="object", timeout=5.0)

    def detect_raw(self, tensor_input):
        outputs = self.session.run(tensor_input)
        if self.model_id == "yolov9":
            return post_process_yolo(outputs, self.width, self.height)
        return decode(outputs, self.width, self.height)
