import numpy as np

from cruciblex.domain.case import (
    CaseSpec,
    GenerationSpec,
    InvocationSpec,
    OperatorSpec,
    OracleSpec,
    ParameterSpec,
    ShapeSpec,
)
from cruciblex.domain.enums import ParameterKind
from cruciblex.generation.expand import expand_cases
from cruciblex.plugins.generators.default import DefaultInputGenerator


def _parameter(name: str, dims: list[int], relationship: dict[str, object] | None = None) -> ParameterSpec:
    metadata = {} if relationship is None else {"shape_relationship": relationship}
    return ParameterSpec(name=name, kind=ParameterKind.TENSOR, dtypes=["fp32"], shape=ShapeSpec(dims=dims), metadata=metadata)


def test_shape_relationships_resolve_declaratively_and_deterministically():
    case = CaseSpec(
        id=701,
        operator=OperatorSpec(name="relationship-test"),
        invocation=InvocationSpec(api="numpy.add", api_type="function"),
        oracle=OracleSpec(),
        generation=GenerationSpec(seed=17, constraints=["shape_relationships"]),
        parameters=[
            _parameter("input", [2, 3, 4]),
            _parameter("broadcast", [9, 3, 7], {"kind": "broadcastable_with", "source": "input"}),
            _parameter("same_numel", [2, 6], {"kind": "same_numel", "source": "input"}),
            _parameter("dim_match", [5, 6], {"kind": "dim_equal", "source": "input", "dimension": 0, "source_dimension": 1}),
            _parameter("aligned", [5, 7], {"kind": "divisible_by", "divisor": 4, "dimension": 1}),
            _parameter("ranked", [2], {"kind": "rank_range", "min_rank": 3, "max_rank": 4}),
            _parameter("aliased", [7, 8], {"kind": "dimension_alias", "source": "input", "source_dimension": 0, "dimension": 1}),
            _parameter("transposed", [9, 9, 9], {"kind": "transpose_of", "source": "input", "axes": [2, 0, 1]}),
        ],
    )
    first, second = expand_cases([case])[0], expand_cases([case])[0]
    shapes = {parameter.name: parameter.shape.dims for parameter in first.parameters}
    assert shapes == {"input": [2, 3, 4], "broadcast": [1, 3, 1], "same_numel": [2, 12], "dim_match": [3, 6], "aligned": [5, 8], "ranked": [2, 1, 1], "aliased": [7, 2], "transposed": [4, 2, 3]}
    assert first.model_dump() == second.model_dump()


def test_transpose_relationship_rejects_invalid_axes_without_mutating_shape():
    case = CaseSpec(
        id=704,
        operator=OperatorSpec(name="invalid-transpose"),
        invocation=InvocationSpec(api="numpy.add", api_type="function"),
        oracle=OracleSpec(),
        generation=GenerationSpec(seed=3, constraints=["shape_relationships"]),
        parameters=[
            _parameter("input", [2, 3, 4]),
            _parameter("transposed", [5, 6, 7], {"kind": "transpose_of", "source": "input", "axes": [0, 0, 1]}),
        ],
    )

    expanded = expand_cases([case])[0]

    assert expanded.parameters[1].shape.dims == [5, 6, 7]


def test_dtype_value_enum_and_optional_policies():
    generator = DefaultInputGenerator()
    tensor = _parameter("input", [2]).model_copy(update={"metadata": {"value_policy": {"kind": "one"}}})
    enum = ParameterSpec(name="mode", kind=ParameterKind.ATTRIBUTE, metadata={"enum_values": ["nearest", "linear"]})
    optional = ParameterSpec(name="scale", kind=ParameterKind.ATTRIBUTE, required=False, metadata={"optional_values": [None, 1.0]})
    assert generator._generate_parameter(tensor).tolist() == [1.0, 1.0]
    assert generator._generate_parameter(enum) == "nearest"
    assert generator._generate_parameter(optional) is None


def test_dtype_policy_selects_allowed_group_member():
    parameter = _parameter("input", [2]).model_copy(update={"metadata": {"dtype_policy": {"group": "floating", "groups": {"floating": ["fp16", "fp32"]}, "allowed": ["fp32"]}}})
    case = CaseSpec(id=702, operator=OperatorSpec(name="dtype-policy"), invocation=InvocationSpec(api="numpy.add", api_type="function"), oracle=OracleSpec(), generation=GenerationSpec(seed=0, constraints=["dtype_policy"]), parameters=[parameter])
    assert expand_cases([case])[0].parameters[0].dtypes == ["fp32"]


def test_product_limits_bound_multiple_parameter_shapes():
    case = CaseSpec(
        id=703,
        operator=OperatorSpec(name="product-limit"),
        invocation=InvocationSpec(api="numpy.add", api_type="function"),
        oracle=OracleSpec(),
        generation=GenerationSpec(
            constraints=["product_limits"],
            metadata={"product_limits": [{"parameters": ["left", "right"], "max_elements": 64}]},
        ),
        parameters=[_parameter("left", [8, 8]), _parameter("right", [4, 4])],
    )

    generated = expand_cases([case])[0]
    shapes = [parameter.shape.dims for parameter in generated.parameters]
    assert shapes[0][0] * shapes[0][1] * shapes[1][0] * shapes[1][1] <= 64
    assert generated.parameters[0].metadata["product_limit"] == 64


def test_backend_dtype_filter_and_reference_promotion():
    filtered = _parameter("filtered", [2]).model_copy(update={
        "dtypes": ["fp16", "fp32"],
        "metadata": {"dtype_policy": {"backend": "gpu", "backend_allowed": {"gpu": ["fp16"]}}},
    })
    promoted = _parameter("output", [2], {"kind": "same_rank", "source": "left"}).model_copy(update={
        "dtypes": ["fp16"],
        "metadata": {"dtype_promotion": {"sources": ["left", "right"]}},
    })
    case = CaseSpec(
        id=704,
        operator=OperatorSpec(name="dtype-promotion"),
        invocation=InvocationSpec(api="numpy.add", api_type="function"),
        oracle=OracleSpec(),
        generation=GenerationSpec(constraints=["dtype_policy", "dtype_promotion"]),
        parameters=[filtered, _parameter("left", [2]).model_copy(update={"dtypes": ["fp16"]}), _parameter("right", [2]).model_copy(update={"dtypes": ["fp32"]}), promoted],
    )
    generated = expand_cases([case])[0]
    assert generated.parameters[0].dtypes == ["fp16"]
    assert generated.parameters[-1].dtypes == ["fp32"]
    assert generated.parameters[-1].metadata["resolved_dtype_promotion"] == ["fp16", "fp32"]


def test_value_distributions_are_seeded_and_sparsity_is_applied():
    parameter = _parameter("sample", [4, 4]).model_copy(update={
        "metadata": {"value_policy": {"kind": "uniform", "low": -2, "high": 2}},
    })
    sparse = _parameter("sparse", [8]).model_copy(update={
        "metadata": {"value_policy": {"kind": "sparsity", "ratio": 1.0}},
    })
    case = CaseSpec(
        id=705,
        operator=OperatorSpec(name="value-policy"),
        invocation=InvocationSpec(api="numpy.abs", api_type="function"),
        oracle=OracleSpec(),
        generation=GenerationSpec(seed=9, constraints=["value_policy"]),
        parameters=[parameter, sparse],
    )
    first = expand_cases([case])[0]
    second = expand_cases([case])[0]
    generator = DefaultInputGenerator()
    first_values = generator.generate(type("Request", (), {"case": first})())
    second_values = generator.generate(type("Request", (), {"case": second})())
    assert np.array_equal(first_values[0], second_values[0])
    assert np.all(first_values[1] == 0)


def test_dtype_aware_numeric_boundary_policies():
    generator = DefaultInputGenerator()
    integer = _parameter("integer", [4]).model_copy(update={
        "dtypes": ["int8"],
        "metadata": {"value_policy": {"kind": "integer_bounds"}},
    })
    floating = _parameter("floating", [3]).model_copy(update={
        "dtypes": ["fp32"],
        "metadata": {"value_policy": {"kind": "float_bounds", "scale": 0.25}},
    })
    integer_values = generator._generate_parameter(integer)
    float_values = generator._generate_parameter(floating)
    assert integer_values.tolist() == [-128, 127, -128, 127]
    assert float_values[0] < 0 < float_values[2]
    assert float_values[1] == 0
