import unittest
from fastapi.testclient import TestClient
from task1 import app, load_model  # Ensure task1.py is in the same directory

class TestPredictAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class method to load the model before any tests are executed."""
        super(TestPredictAPI, cls).setUpClass()
        # Set the model path here
        cls.model_path = 'mnist_model.h5'
        # Load the model using the function defined in task1.py
        load_model(cls.model_path)

    def setUp(self):
        """Set up the TestClient for the FastAPI app."""
        self.client = TestClient(app)

    def test_predict_valid_image(self):
        """Test the '/predict/' endpoint with a simulated valid image upload."""
        # use 'valid_28x28.jpg' as a correct file in the 'test_samples' folder
        with open("test_samples/valid_28x28.jpg", "rb") as img:
            response = self.client.post("/predict/", files={"file": ("valid_28x28.jpg", img, "image/jpeg")})
        # Check that the API returns a 200 OK response, indicating successful processing.
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")
        # Verify that the response body includes a digit as expected.
        self.assertIn("digit", response.json(), "API should return a digit.")
        self.assertIsInstance(response.json().get('digit'), str, "The 'digit' should be a string.")

    def test_predict_invalid_image(self):
        """Test the '/predict/' endpoint with an invalid file to check error handling."""
        # use 'invalid_image.png' as a file designed to fail the image processing
        with open("test_samples/invalid_image.png", "rb") as img:
            response = self.client.post("/predict/", files={"file": ("invalid_image.png", img, "image/png")})
        # Ensure the API correctly returns a 400 Bad Request response for invalid inputs.
        self.assertEqual(response.status_code, 400, "Expected 400 Bad Request due to invalid image.")

if __name__ == '__main__':
    unittest.main()
