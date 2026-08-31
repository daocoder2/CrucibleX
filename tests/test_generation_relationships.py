import numpy as np
import pytest

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


def test_exact_values_are_generated_and_shape_mismatch_is_rejected():
    parameter = _parameter("input", [2, 2]).model_copy(update={"values": [[1, 2], [3, 4]]})
    generator = DefaultInputGenerator()

    values = generator._generate_parameter(parameter)

    assert values.dtype == np.float32
    np.testing.assert_array_equal(values, np.asarray([[1, 2], [3, 4]], dtype=np.float32))

    invalid = parameter.model_copy(update={"values": [[1, 2, 3]]})
    with pytest.raises(ValueError, match="exact values shape"):
        generator._generate_parameter(invalid)


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



def test_matrix_profiles_are_deterministic_and_preserve_declared_structure():
    generator = DefaultInputGenerator()
    well_conditioned = _parameter("well", [6, 4]).model_copy(update={
        "metadata": {"value_policy": {"kind": "matrix_profile", "profile": "well_conditioned", "condition_number": 3.0}, "value_policy_seed": 19},
    })
    rank_deficient = _parameter("rank_deficient", [6, 4]).model_copy(update={
        "metadata": {"value_policy": {"kind": "matrix_profile", "profile": "rank_deficient", "rank": 2}, "value_policy_seed": 19},
    })

    first = generator._generate_parameter(well_conditioned)
    second = generator._generate_parameter(well_conditioned)
    deficient = generator._generate_parameter(rank_deficient)

    assert np.array_equal(first, second)
    assert np.linalg.cond(first) <= 3.01
    assert np.linalg.matrix_rank(deficient) <= 2


def test_matrix_profile_rejects_non_matrix_or_invalid_rank():
    generator = DefaultInputGenerator()
    vector = _parameter("vector", [4]).model_copy(update={
        "metadata": {"value_policy": {"kind": "matrix_profile", "profile": "well_conditioned"}},
    })
    invalid_rank = _parameter("invalid_rank", [4, 4]).model_copy(update={
        "metadata": {"value_policy": {"kind": "matrix_profile", "profile": "rank_deficient", "rank": 4}},
    })
    invalid_condition = _parameter("invalid_condition", [4, 4]).model_copy(update={
        "metadata": {"value_policy": {"kind": "matrix_profile", "profile": "well_conditioned", "condition_number": 0.5}},
    })

    with pytest.raises(ValueError, match="rank-2"):
        generator._generate_parameter(vector)
    with pytest.raises(ValueError, match="rank must satisfy"):
        generator._generate_parameter(invalid_rank)
    with pytest.raises(ValueError, match="condition_number"):
        generator._generate_parameter(invalid_condition)


def test_operator_facts_drive_dtype_backend_deny_and_broadcast_relationships():
    case = CaseSpec(
        id=720,
        operator=OperatorSpec(name="facts-test"),
        invocation=InvocationSpec(api="numpy.add", api_type="function"),
        generation=GenerationSpec(metadata={"operator_facts": {"schema_version": 1, "parameters": {
            "left": {"dtypes": ["fp32", "bf16"], "dtype_policy": {"backend": "npu", "backend_denied": {"npu": ["fp32"]}}, "shape_policy": {"broadcast_group": "inputs"}},
            "right": {"dtypes": ["fp32", "bf16"], "shape_policy": {"broadcast_group": "inputs"}},
        }}}),
        parameters=[_parameter("left", [2, 3]).model_copy(update={"dtypes": []}), _parameter("right", [7, 3]).model_copy(update={"dtypes": []})],
    )

    expanded = expand_cases([case])[0]

    assert expanded.parameters[0].dtypes == ["bf16"]
    assert expanded.parameters[1].shape.dims == [1, 3]
    assert all(parameter.metadata["resolved_operator_facts"] for parameter in expanded.parameters)


def test_extended_value_dtype_and_layout_policies_generate_declared_boundaries():
    generator = DefaultInputGenerator()
    complex_parameter = _parameter("complex", [4]).model_copy(update={"dtypes": ["complex64"], "metadata": {"value_policy": {"kind": "complex_normal"}}})
    subnormal_parameter = _parameter("subnormal", [3]).model_copy(update={"metadata": {"value_policy": {"kind": "subnormal"}}})
    layout_parameter = _parameter("layout", [2, 2]).model_copy(update={"metadata": {"shape_policy": {"storage_shape": [4, 4], "slice": [[1, 3], [0, 2]], "non_contiguous": True}}})

    complex_values = generator._generate_parameter(complex_parameter)
    subnormal_values = generator._generate_parameter(subnormal_parameter)
    layout_values = generator._generate_parameter(layout_parameter)

    assert complex_values.dtype == np.complex64
    assert np.any(complex_values.imag != 0)
    assert np.any(subnormal_values != 0)
    assert np.all(np.abs(subnormal_values[np.nonzero(subnormal_values)]) < np.finfo(np.float32).tiny)
    assert layout_values.shape == (2, 2)
    assert not layout_values.flags.c_contiguous


def test_policy_libraries_resolve_builtin_operator_facts_and_numeric_edges():
    case = CaseSpec(
        id=730,
        operator=OperatorSpec(name="torch.add"),
        invocation=InvocationSpec(api="torch.add", api_type="function"),
        parameters=[_parameter("input", [3, 3]), _parameter("other", [5, 3])],
    )
    expanded = expand_cases([case])[0]
    generator = DefaultInputGenerator()

    values = generator.generate(type("Request", (), {"case": expanded})())

    assert [parameter.dtypes for parameter in expanded.parameters] == [["fp32"], ["fp32"]]
    assert expanded.parameters[1].shape.dims == [1, 3]
    assert np.isinf(values[0]).any()
    assert np.isnan(values[0]).any()

    boolean = _parameter("boolean", [2]).model_copy(update={"dtypes": ["bool"], "metadata": {"value_policy": {"kind": "boundary_set", "values": [False, True]}}})
    integer = _parameter("integer", [5]).model_copy(update={"dtypes": ["int8"], "metadata": {"value_policy": {"kind": "boundary_set", "values": ["min", "-one", "zero", "one", "max"]}}})
    complex_values = _parameter("complex", [4]).model_copy(update={"dtypes": ["complex64"], "metadata": {"value_policy": {"kind": "boundary_set", "values": ["zero", "one", "inf", "nan"]}}})

    assert generator._generate_parameter(boolean).tolist() == [False, True]
    assert generator._generate_parameter(integer).tolist() == [-128, -1, 0, 1, 127]
    assert np.isinf(generator._generate_parameter(complex_values)).any()


def test_layout_policy_supports_storage_offset_and_element_strides():
    parameter = _parameter("strided", [2, 2]).model_copy(update={"metadata": {"shape_policy": {"storage_shape": [4, 4], "storage_offset": 1, "strides": [4, 2]}}})

    values = DefaultInputGenerator()._generate_parameter(parameter)

    assert values.shape == (2, 2)
    assert values.strides == (16, 8)
    assert not values.flags.c_contiguous



def test_builtin_matmul_facts_alias_inner_dimensions():
    case = CaseSpec(
        id=731,
        operator=OperatorSpec(name="torch.matmul"),
        invocation=InvocationSpec(api="torch.matmul", api_type="function"),
        parameters=[_parameter("input", [2, 3]), _parameter("other", [9, 4])],
    )

    expanded = expand_cases([case])[0]

    assert expanded.parameters[0].dtypes == ["fp32"]
    assert expanded.parameters[1].shape.dims == [3, 4]
    assert expanded.parameters[1].metadata["resolved_shape_relationship"] == "dimension_alias"


def test_bf16_reference_quantizes_and_special_policy_records_rejection():
    generator = DefaultInputGenerator()
    bf16 = _parameter("bf16", [2]).model_copy(update={"dtypes": ["bf16"], "values": [1.00390625, 1.01171875]})
    value = generator._generate_parameter(bf16)

    assert value.dtype == np.float32
    assert value.tolist() == [1.0, 1.015625]

    invalid = _parameter("invalid", [1]).model_copy(update={"dtypes": ["int32"], "metadata": {"value_policy": {"kind": "boundary_set", "values": ["nan"]}}})
    case = CaseSpec(
        id=732,
        operator=OperatorSpec(name="invalid-special"),
        invocation=InvocationSpec(api="numpy.add", api_type="function"),
        generation=GenerationSpec(constraints=["value_policy"]),
        parameters=[invalid],
    )
    expanded = expand_cases([case])[0]

    assert expanded.parameters[0].metadata["value_policy_validation"]["rejected"] == "unsupported_policy_for_dtype"
    with pytest.raises(ValueError, match="unsupported_policy_for_dtype"):
        generator._generate_parameter(expanded.parameters[0])


def test_collection_relationships_link_length_dtype_and_shape():
    source = ParameterSpec(
        name="left",
        kind=ParameterKind.TENSOR_LIST,
        dtypes=["fp16"],
        shape=ShapeSpec(dims=[2, 3]),
        metadata={"length": 3},
    )
    target = ParameterSpec(
        name="right",
        kind=ParameterKind.TENSOR_LIST,
        dtypes=["fp32"],
        shape=ShapeSpec(dims=[1]),
        metadata={"collection_relationship": {"kind": "same_length_as", "source": "left"}},
    )
    dtype_target = target.model_copy(update={"name": "dtype_right", "metadata": {"collection_relationship": {"kind": "same_item_dtype_as", "source": "left"}}})
    shape_target = target.model_copy(update={"name": "shape_right", "metadata": {"collection_relationship": {"kind": "same_item_shape_as", "source": "left"}}})
    case = CaseSpec(
        id=733,
        operator=OperatorSpec(name="collection-link"),
        invocation=InvocationSpec(api="numpy.add", api_type="function"),
        generation=GenerationSpec(constraints=["linked_parameters"]),
        parameters=[source, target, dtype_target, shape_target],
    )

    expanded = expand_cases([case])[0]
    generator = DefaultInputGenerator()
    values = generator._generate_parameter(expanded.parameters[1])
    shape_values = generator._generate_parameter(expanded.parameters[3])

    assert len(values) == 3
    assert expanded.parameters[2].metadata["item_dtypes"] == ["fp16"]
    assert expanded.parameters[3].metadata["item_shapes"] == [[2, 3]]
    assert all(value.shape == (2, 3) for value in shape_values)


def test_collection_broadcast_zip_cartesian_and_nested_items():
    left = ParameterSpec(name="left", kind=ParameterKind.TENSOR_LIST, dtypes=["fp32"], metadata={"item_shapes": [[2, 1], [1, 3]], "length": 2})
    right = ParameterSpec(name="right", kind=ParameterKind.TENSOR_LIST, dtypes=["fp32"], metadata={"item_shapes": [[1, 3], [2, 1]], "length": 2, "collection_relationship": {"kind": "broadcast_items_with", "source": "left"}})
    zipped = ParameterSpec(name="zipped", kind=ParameterKind.TENSOR_LIST, dtypes=["fp32"], metadata={"length": 5, "collection_relationship": {"kind": "zip_with", "source": "left"}})
    cartesian = ParameterSpec(name="cartesian", kind=ParameterKind.TENSOR_LIST, dtypes=["fp32"], metadata={"length": 4, "collection_relationship": {"kind": "cartesian_with", "source": "left"}})
    nested = ParameterSpec(name="nested", kind=ParameterKind.TENSOR_LIST, dtypes=["fp32"], metadata={"items": [{"kind": "tensor_list", "metadata": {"length": 2, "item_shapes": [[2], [3]]}}]})
    case = CaseSpec(id=734, operator=OperatorSpec(name="collection-composition"), invocation=InvocationSpec(api="numpy.add", api_type="function"), generation=GenerationSpec(constraints=["linked_parameters"]), parameters=[left, right, zipped, cartesian, nested])

    expanded = expand_cases([case])[0]
    generated = DefaultInputGenerator()
    broadcast_values = generated._generate_parameter(expanded.parameters[1])
    nested_values = generated._generate_parameter(expanded.parameters[4])

    assert expanded.parameters[1].metadata["item_shapes"] == [[2, 3], [2, 3]]
    assert all(value.shape == shape for value, shape in zip(broadcast_values, [(2, 3), (2, 3)], strict=True))
    assert expanded.parameters[2].metadata["length"] == 2
    assert expanded.parameters[2].metadata["collection_pairing"] == "zip"
    assert expanded.parameters[3].metadata["length"] == 8
    assert expanded.parameters[3].metadata["collection_pairing"] == "cartesian"
    assert len(nested_values) == 1
    assert [value.shape for value in nested_values[0]] == [(2,), (3,)]


@pytest.mark.parametrize("operator", ["torch.sum", "torch.mean", "torch.norm", "torch.sort", "torch.topk"])
def test_extended_operator_facts_apply_floating_rank_and_seeded_values(operator):
    case = CaseSpec(id=735, operator=OperatorSpec(name=operator), invocation=InvocationSpec(api=operator, api_type="function"), parameters=[_parameter("input", [2, 3])])

    expanded = expand_cases([case])[0]
    parameter = expanded.parameters[0]

    assert parameter.dtypes == ["fp32"]
    assert parameter.metadata["resolved_operator_facts"] is True
    assert parameter.metadata["shape_relationship"]["kind"] == "rank_range"
    assert parameter.metadata["value_policy"]["kind"] == "normal"


def test_index_select_facts_select_int64_index():
    case = CaseSpec(id=736, operator=OperatorSpec(name="torch.index_select"), invocation=InvocationSpec(api="torch.index_select", api_type="function"), parameters=[_parameter("input", [2, 3]), _parameter("index", [2]).model_copy(update={"dtypes": []})])

    expanded = expand_cases([case])[0]

    assert expanded.parameters[0].dtypes == ["fp32"]
    assert expanded.parameters[1].dtypes == ["int64"]
