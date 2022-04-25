import unittest

import SentimentAnalysis

class SentimentAnalysis_Test(unittest.TestCase):
    def setUp(self):
        self.sa = SentimentAnalysis.SentimentAnalysis()

    def test(self):
        params = [
            'All too much of the man-made is ugly, inefficient, depressing chaos.'
        ]
        response = self.sa.predict(params)
        self.assertIsNotNone(response)


if __name__ == '__main__':
    unittest.main()
