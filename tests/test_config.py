from shopsage.config import TEST_END, TEST_START, VAL_START, Settings


def test_split_dates_are_ordered():
    assert VAL_START < TEST_START < TEST_END


def test_base_url_gets_v1_suffix():
    s = Settings(llm_base_url="http://localhost:11434")
    assert s.openai_base_url == "http://localhost:11434/v1"
    s = Settings(llm_base_url="https://api.example.com/v1/")
    assert s.openai_base_url == "https://api.example.com/v1"
