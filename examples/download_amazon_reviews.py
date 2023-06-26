from datasets import load_dataset

"""
The data can be directly downloaded as follows
curl https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/train/dataset_en_train.json > dataset_en_train
curl https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/test/dataset_en_test.json > dataset_en_test
curl https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/dev/dataset_en_dev.json > dataset_en_dev

This downloads an array of JSON objects.

The HuggingFace interface will convert this to the more efficient Arrow format.
"""

dataset_name = "amazon_reviews_multi"
language = "en"

# On Windows the default cache dir is:
# C:\Users\USERNAME\.cache\huggingface\datasets\amazon_reviews_multi\en\1.0.0\CHECKSUM
# On Linux it is:
## ~/.cache/huggingface/datasets/amazon_reviews_multi/en/1.0.0/CHECKSUM
cache_dir = "datasets" 

data = load_dataset(dataset_name, language, cache_dir=cache_dir)

# The raw data under datasets/download can be deleted

