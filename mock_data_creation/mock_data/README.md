# Mock Data Details
Using the ImageNet-R benchmark, we created an error-relevant data attributes mock predictions mock data. See `mock_data_create/create_mock_data_manual.ipynb` for creating more splits.
* `split_0.txt`: base noise = prediction corruption at corruption probability = 0.25 (no attribute; corrupt uniformly across other classes) 
* `split_1.txt`: corrupt goose images with sketch attribute at corruption probability = 0.6 (corrupt uniformly across other classes) with base noise (see above)
* `split_2.txt`: corrupt orangutan images with sketch attribute at corruption probability = 0.6 (weighted corruption to chimpanzee, ~25%) with base noise (see `split_0`)