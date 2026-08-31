from cruciblex.domain.case import CaseSpec
from cruciblex.generation.filtering import case_dimensions, filter_cases


def _case(name: str, dtype: str, tags: list[str]) -> CaseSpec:
    return CaseSpec.model_validate({
        "id": 1,
        "operator": {"name": name},
        "invocation": {"api": name, "api_type": "function", "executor": "torch"},
        "parameters": [{"name": "input", "kind": "tensor", "dtypes": [dtype], "metadata": {}}],
        "metadata": {"tags": tags},
    })


def test_filter_cases_supports_operator_dtype_and_tag_selectors():
    relu = _case("torch.relu", "fp32", ["hardware", "unary"])
    matmul = _case("torch.matmul", "fp16", ["hardware", "matrix"])
    assert case_dimensions(relu)["operator"] == {"torch.relu"}
    assert filter_cases([relu, matmul], include={"dtype": {"fp16"}}) == [matmul]
    assert filter_cases([relu, matmul], exclude={"tag": {"matrix"}}) == [relu]
