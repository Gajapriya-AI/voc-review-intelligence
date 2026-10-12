import unittest
from src.analysis import analyze_review
from src.model import load_model

class ReviewAnalysisTests(unittest.TestCase):
    def test_empty_review_needs_review(self):
        self.assertEqual(analyze_review(" ")["sentiment"], "Needs review")
    def test_positive_review(self):
        r=analyze_review("The battery lasts all day and the camera takes sharp photos.")
        self.assertEqual(r["sentiment"], "Positive")
    def test_negative_review(self):
        r=analyze_review("The battery drains fast and the camera quality is poor.")
        self.assertEqual(r["sentiment"], "Negative")
    def test_mixed_review_not_positive(self):
        r=analyze_review("Phone looks beautiful, but battery life is poor. Camera quality is disappointing. It gets very hot while charging.")
        self.assertIn(r["sentiment"], ["Mixed", "Negative"])
        self.assertNotEqual(r["sentiment"], "Positive")
        self.assertTrue(any(a["aspect"]=="Battery" for a in r["aspects"]))
        self.assertTrue(any(a["aspect"]=="Heating & Noise" and a["sentiment"]=="Negative" for a in r["aspects"]))
    def test_aspect_sentiments(self):
        r=analyze_review("The laptop performance is excellent but battery drains fast.")
        self.assertIn(r["sentiment"], ["Mixed", "Negative"])
        self.assertTrue(r["aspects"])
    def test_model_loads_after_training(self):
        model=load_model()
        self.assertIsNotNone(model)
        pred,conf=model.predict_sentiment("This is a great product.")
        self.assertIn(pred,["Positive","Negative","Neutral"])
        self.assertGreaterEqual(conf,0)
        self.assertLessEqual(conf,1)

if __name__=='__main__': unittest.main()
