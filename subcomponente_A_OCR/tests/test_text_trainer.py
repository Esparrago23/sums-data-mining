import unittest

from src.text_trainer import _clean_ocr_text, _score_ocr_candidate


class TextTrainerOCRTests(unittest.TestCase):
    def test_clean_ocr_text_removes_common_noise(self) -> None:
        sample = "  || Calle 123 __  "
        self.assertEqual(_clean_ocr_text(sample), "Calle 123")

    def test_score_ocr_candidate_prefers_words_over_noise(self) -> None:
        noisy = "= = | _"
        readable = "Calle los Pinos"
        self.assertGreater(_score_ocr_candidate(readable, [90, 88, 86]), _score_ocr_candidate(noisy, [0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
