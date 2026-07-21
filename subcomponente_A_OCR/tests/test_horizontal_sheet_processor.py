import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from horizontal_sheet_processor import (
    classify_horizontal_page,
    detect_orientation,
    match_catalog_option,
)


class HorizontalSheetProcessorTests(unittest.TestCase):
    def test_catalog_match_handles_education_variants(self) -> None:
        catalog = [
            "primaria",
            "primaria truncada",
            "secundaria",
            "preparatoria",
            "bachillerato",
            "licenciatura",
        ]
        self.assertEqual(match_catalog_option("secundaria truncada", catalog), "secundaria")
        self.assertEqual(match_catalog_option("bachillerato", catalog), "bachillerato")
        self.assertEqual(match_catalog_option("licenciatura truncada", catalog), "licenciatura")

    def test_catalog_match_handles_boolean_and_frequency(self) -> None:
        self.assertEqual(match_catalog_option("si", ["si", "no"]), "si")
        self.assertEqual(match_catalog_option("mensual", ["anual", "mensual"]), "mensual")

    def test_detect_orientation_prefers_content_in_upper_region(self) -> None:
        image = np.zeros((120, 240), dtype=np.uint8)
        image[10:60, 20:200] = 255
        orientation = detect_orientation(image, (0, 0, 240, 120))
        self.assertEqual(orientation, 0)

    def test_classify_horizontal_page_on_wide_layout(self) -> None:
        image = np.zeros((180, 320), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (300, 140), 255, -1)
        self.assertEqual(classify_horizontal_page(image, (0, 0, 320, 180)), "horizontal")


if __name__ == "__main__":
    unittest.main()
