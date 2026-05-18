import logging
from pathlib import Path

from config import CROPS_DIR, DESC_PATH, FRAMES_DIR, MASKS_DIR, TEMP_GT_JSON
from pipeline.base import PipelineStage
from support_objects.select_support_object import select_support_objects
from utils.cropper import save_crop
from utils.data_loader import load_descriptions, load_frame_and_mask
from utils.gt_builder import GTBuilder


class TempGtStage(PipelineStage):
    def __init__(
        self,
        desc_path: Path = DESC_PATH,
        frames_dir: Path = FRAMES_DIR,
        masks_dir: Path = MASKS_DIR,
        crops_dir: Path = CROPS_DIR,
        output_path: Path = TEMP_GT_JSON,
    ):
        self.desc_path = desc_path
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.crops_dir = crops_dir
        self._output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    def run(self) -> None:
        logger = logging.getLogger(__name__)
        self.crops_dir.mkdir(exist_ok=True)

        logger.info("Loading object descriptions...")
        descriptions = load_descriptions(self.desc_path)

        frame_names = sorted(
            f.name for f in self.frames_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")
        )
        logger.info("Building temp_gt from %d frames...", len(frame_names))

        gt_builder = GTBuilder(descriptions)
        for frame_name in frame_names:
            logger.info("Processing %s...", frame_name)
            rgb, mask = load_frame_and_mask(frame_name, self.frames_dir, self.masks_dir)
            if mask is None:
                continue

            supports = select_support_objects(mask, descriptions)
            support_ids = [obj["id"] for obj in supports]
            frame_id = frame_name.split(".")[0]

            for obj in supports:
                save_crop(rgb, mask, obj["bbox"], obj["id"], support_ids, frame_id, self.crops_dir)

            gt_builder.process_frame(mask, supports)

        temp_gt = gt_builder.build_gt()
        self.save(temp_gt)
        logger.info("temp_gt saved to %s", self.output_path)
