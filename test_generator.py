import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Sample generator test case
generator_test_cases = [
    LLMTestCase(
        query="What is Pinecone used for?",
        retrieved_context="Pinecone is used for vector embeddings in machine learning.",
        expected_output="Pinecone helps manage vector embeddings in machine learning models."
    ),
]

# Function to test generator responses
@pytest.mark.parametrize("test_case", generator_test_cases)
def test_generator(test_case: LLMTestCase):
    metrics = [AnswerRelevancyMetric()]
    assert_test(test_case, metrics)

