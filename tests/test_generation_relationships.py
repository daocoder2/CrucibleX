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


def test_integer_attribute_collections_preserve_integer_items_and_tuple_abi():
    generator = DefaultInputGenerator()
    shape_list = ParameterSpec(name="shape", kind=ParameterKind.ATTRIBUTE_LIST, dtypes=["int64"], values=[3, 2])
    shape_tuple = shape_list.model_copy(update={"kind": ParameterKind.ATTRIBUTE_TUPLE})

    assert generator._generate_parameter(shape_list) == [3, 2]
    assert generator._generate_parameter(shape_tuple) == (3, 2)


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


def test_operator_contract_resolves_reduce_topk_index_and_matmul_evidence():
    reduce = CaseSpec(id=740, operator=OperatorSpec(name="torch.mean"), invocation=InvocationSpec(api="torch.mean", api_type="function"), parameters=[_parameter("input", [2, 3, 4]), _parameter("dim", [1]).model_copy(update={"values": 1}), _parameter("keepdim", [1]).model_copy(update={"values": True})])
    topk = CaseSpec(id=741, operator=OperatorSpec(name="torch.topk"), invocation=InvocationSpec(api="torch.topk", api_type="function"), parameters=[_parameter("input", [2, 5]), _parameter("k", [1]).model_copy(update={"values": 3}), _parameter("dim", [1]).model_copy(update={"values": 1})])
    indexed = CaseSpec(id=742, operator=OperatorSpec(name="torch.index_select"), invocation=InvocationSpec(api="torch.index_select", api_type="function"), parameters=[_parameter("input", [2, 5]), _parameter("index", [2]), _parameter("dim", [1]).model_copy(update={"values": 1})])
    matmul = CaseSpec(id=743, operator=OperatorSpec(name="torch.matmul"), invocation=InvocationSpec(api="torch.matmul", api_type="function"), parameters=[_parameter("input", [2, 1, 3, 4]), _parameter("other", [1, 5, 4, 6])])

    expanded = expand_cases([reduce, topk, indexed, matmul])

    assert expanded[0].metadata["resolved_operator_contract"]["output_shape"] == [2, 1, 4]
    assert expanded[0].metadata["resolved_operator_contract"]["output_dtype"] == "fp32"
    assert expanded[1].metadata["resolved_operator_contract"]["output_shape"] == [2, 3]
    assert expanded[1].metadata["resolved_operator_contract"]["indices_dtype"] == "int64"
    assert expanded[2].metadata["resolved_operator_contract"]["index_range"] == [0, 4]
    assert expanded[3].metadata["resolved_operator_contract"]["batch_shape"] == [2, 5]
    assert expanded[3].metadata["resolved_operator_contract"]["inner_dimension"] == 4


def test_declarative_conv_norm_attention_and_aclnn_contracts_remain_capability_only():
    case = CaseSpec(id=744, operator=OperatorSpec(name="contract-declarations"), invocation=InvocationSpec(api="torch.conv2d", api_type="function"), generation=GenerationSpec(metadata={"operator_fact_library": ["torch.conv2d", "torch.layer_norm", "torch.scaled_dot_product_attention"], "operator_contract": {"aclnn": {"format": "ND", "storage_shape": "declared", "workspace": "runtime_managed", "dynamic_output": False}}}), parameters=[])

    expanded = expand_cases([case])[0]
    contract = expanded.metadata["resolved_operator_contract"]

    assert contract["family"] == "attention"
    assert contract["runtime_supported"] is False
    assert contract["aclnn"]["workspace"] == "runtime_managed"
    assert contract["aclnn"]["dynamic_output"] is False


def test_operator_contract_generates_invalid_dim_k_index_and_reshape_variants():
    topk = CaseSpec(
        id=750,
        operator=OperatorSpec(name="torch.topk"),
        invocation=InvocationSpec(api="torch.topk", api_type="function"),
        generation=GenerationSpec(invalid_count=2),
        parameters=[
            _parameter("input", [2, 5]),
            ParameterSpec(name="k", kind=ParameterKind.ATTRIBUTE, values=3),
            ParameterSpec(name="dim", kind=ParameterKind.ATTRIBUTE, values=1),
        ],
    )
    indexed = CaseSpec(
        id=751,
        operator=OperatorSpec(name="torch.index_select"),
        invocation=InvocationSpec(api="torch.index_select", api_type="function"),
        generation=GenerationSpec(invalid_count=2),
        parameters=[
            _parameter("input", [2, 5]),
            _parameter("index", [2]),
            ParameterSpec(name="dim", kind=ParameterKind.ATTRIBUTE, values=1),
        ],
    )
    reshaped = CaseSpec(
        id=752,
        operator=OperatorSpec(name="torch.reshape"),
        invocation=InvocationSpec(api="torch.reshape", api_type="function"),
        generation=GenerationSpec(invalid_count=1),
        parameters=[_parameter("input", [2, 3]), ParameterSpec(name="shape", kind=ParameterKind.ATTRIBUTE_LIST, values=[3, 2])],
    )

    expanded = expand_cases([topk, indexed, reshaped])

    assert [case.metadata["contract_invalid_reason"] for case in expanded[1:3]] == ["dim_out_of_range", "k_exceeds_axis"]
    assert expanded[2].parameters[1].values == 6
    assert [case.metadata["contract_invalid_reason"] for case in expanded[4:6]] == ["dim_out_of_range", "index_out_of_range"]
    assert expanded[5].parameters[1].metadata["selected_invalid_value"] == 5
    assert expanded[7].parameters[1].values == [3, 3]


def test_where_reshape_and_transpose_contracts_resolve_output_shapes():
    where = CaseSpec(
        id=753,
        operator=OperatorSpec(name="torch.where"),
        invocation=InvocationSpec(api="torch.where", api_type="function"),
        parameters=[
            ParameterSpec(name="condition", kind=ParameterKind.TENSOR, shape=ShapeSpec(dims=[2, 3])),
            _parameter("input", [2, 3]),
            _parameter("other", [1, 3]),
        ],
    )
    transpose = CaseSpec(
        id=754,
        operator=OperatorSpec(name="torch.transpose"),
        invocation=InvocationSpec(api="torch.transpose", api_type="function"),
        parameters=[_parameter("input", [2, 3, 4]), ParameterSpec(name="dim0", kind=ParameterKind.ATTRIBUTE, values=0), ParameterSpec(name="dim1", kind=ParameterKind.ATTRIBUTE, values=2)],
    )

    expanded = expand_cases([where, transpose])

    assert expanded[0].parameters[0].dtypes == ["bool"]
    assert expanded[0].metadata["resolved_operator_contract"]["output_shape"] == [2, 3]
    assert expanded[1].metadata["resolved_operator_contract"]["output_shape"] == [4, 3, 2]


def test_conv_norm_attention_facts_generate_legal_shapes_and_contracts():
    conv = CaseSpec(
        id=760,
        operator=OperatorSpec(name="torch.conv2d"),
        invocation=InvocationSpec(api="torch.conv2d", api_type="function"),
        parameters=[
            _parameter("input", [2, 3, 8, 8]),
            _parameter("weight", [4, 9, 3, 3]),
            _parameter("bias", [9]),
        ],
    )
    norm = CaseSpec(
        id=761,
        operator=OperatorSpec(name="torch.layer_norm"),
        invocation=InvocationSpec(api="torch.layer_norm", api_type="function"),
        parameters=[
            _parameter("input", [2, 3, 4]),
            _parameter("normalized_shape", [9]),
            _parameter("weight", [9]),
            _parameter("bias", [9]),
        ],
    )
    attention = CaseSpec(
        id=762,
        operator=OperatorSpec(name="torch.scaled_dot_product_attention"),
        invocation=InvocationSpec(api="torch.scaled_dot_product_attention", api_type="function"),
        parameters=[
            _parameter("query", [2, 4, 3, 8]),
            _parameter("key", [2, 4, 5, 8]),
            _parameter("value", [2, 4, 5, 6]),
        ],
    )

    generated = expand_cases([conv, norm, attention])

    assert generated[0].parameters[1].shape.dims == [4, 3, 3, 3]
    assert generated[0].parameters[2].shape.dims == [4]
    assert generated[0].metadata["resolved_operator_contract"]["output_shape"] == [2, 4, 6, 6]
    assert generated[1].parameters[1].shape.dims == [4]
    assert generated[1].parameters[2].shape.dims == [4]
    assert generated[1].metadata["resolved_operator_contract"]["output_shape"] == [2, 3, 4]
    assert generated[2].metadata["resolved_operator_contract"]["output_shape"] == [2, 4, 3, 6]


def test_layer_norm_multidimensional_suffix_is_generated_automatically():
    case = CaseSpec(id=777, operator=OperatorSpec(name="torch.layer_norm"), invocation=InvocationSpec(api="torch.layer_norm", api_type="function"), parameters=[_parameter("input", [2, 3, 4, 5]), ParameterSpec(name="normalized_shape", kind=ParameterKind.ATTRIBUTE_TUPLE, dtypes=["int64"], values=[9, 9], metadata={"shape_relationship": {"kind": "last_k_dimensions_as", "source": "input", "k": 2}})])
    generated = expand_cases([case])[0]
    assert generated.parameters[1].shape.dims == [4, 5]


def test_conv_groups_and_attention_mask_are_generated_from_shape_contracts():
    conv = CaseSpec(id=775, operator=OperatorSpec(name="torch.conv2d"), invocation=InvocationSpec(api="torch.conv2d", api_type="function"), parameters=[_parameter("input", [1, 4, 7, 7]), _parameter("weight", [6, 9, 3, 3]), ParameterSpec(name="groups", kind=ParameterKind.ATTRIBUTE, values=2)])
    generated_conv = expand_cases([conv])[0]
    assert generated_conv.parameters[1].shape.dims == [6, 2, 3, 3]

    attention = CaseSpec(id=776, operator=OperatorSpec(name="torch.scaled_dot_product_attention"), invocation=InvocationSpec(api="torch.scaled_dot_product_attention", api_type="function"), parameters=[_parameter("query", [2, 3, 4, 5]), _parameter("key", [2, 3, 6, 5]), _parameter("value", [2, 3, 6, 7]), ParameterSpec(name="attn_mask", kind=ParameterKind.TENSOR, dtypes=["bool"], shape=ShapeSpec(dims=[1, 1, 1, 1]))])
    generated_attention = expand_cases([attention])[0]
    assert generated_attention.parameters[3].shape.dims == [2, 3, 4, 6]
    assert generated_attention.parameters[3].dtypes == ["bool"]


def test_attention_dimension_aliases_and_norm_attribute_invalid_generation():
    attention = CaseSpec(id=773, operator=OperatorSpec(name="torch.scaled_dot_product_attention"), invocation=InvocationSpec(api="torch.scaled_dot_product_attention", api_type="function"), generation=GenerationSpec(), parameters=[_parameter("query", [2, 3, 4, 5]), _parameter("key", [9, 8, 7, 6]), _parameter("value", [9, 8, 7, 6])])
    generated = expand_cases([attention])[0]
    shapes = {parameter.name: parameter.shape.dims for parameter in generated.parameters}
    assert shapes["key"] == [2, 3, 7, 5]
    assert shapes["value"] == [2, 3, 7, 6]

    norm = CaseSpec(id=774, operator=OperatorSpec(name="torch.layer_norm"), invocation=InvocationSpec(api="torch.layer_norm", api_type="function"), generation=GenerationSpec(invalid_count=1), parameters=[_parameter("input", [2, 3, 4]), ParameterSpec(name="normalized_shape", kind=ParameterKind.ATTRIBUTE_TUPLE, dtypes=["int64"], values=[4]), _parameter("weight", [4]), _parameter("bias", [4])])
    invalid = expand_cases([norm])[1]
    assert invalid.metadata["contract_invalid_reason"] == "normalized_shape_mismatch"
    assert invalid.parameters[1].values == [5]


def test_gather_scatter_contract_validates_complex_index_and_src_shapes():
    def parameter(name: str, dims: list[int], dtype: str = "fp32") -> ParameterSpec:
        return _parameter(name, dims).model_copy(update={"dtypes": [dtype]})

    def case(case_id: int, mode: str, index_dims: list[int], src_dims: list[int] | None = None) -> CaseSpec:
        parameters = [parameter("input", [2, 3, 4]), ParameterSpec(name="dim", kind=ParameterKind.ATTRIBUTE, dtypes=["int64"], values=1), parameter("index", index_dims, "int64")]
        if src_dims is not None:
            parameters.append(parameter("src", src_dims))
        return CaseSpec(
            id=case_id,
            operator=OperatorSpec(name=f"custom.{mode}"),
            invocation=InvocationSpec(api=f"torch.{mode}", api_type="function"),
            generation=GenerationSpec(metadata={"operator_facts": {"contract": {"family": "index", "mode": mode, "input": "input", "index": "index", "dim_parameter": "dim"}}}),
            parameters=parameters,
        )

    gather = expand_cases([case(788, "gather", [2, 5, 4])])[0].metadata["resolved_operator_contract"]
    gather_bad = expand_cases([case(789, "gather", [3, 5, 4])])[0].metadata["resolved_operator_contract"]
    scatter = expand_cases([case(790, "scatter", [2, 5, 4], [2, 5, 4])])[0].metadata["resolved_operator_contract"]
    scatter_bad = expand_cases([case(791, "scatter", [2, 5, 4], [2, 4, 4])])[0].metadata["resolved_operator_contract"]

    assert gather["valid_index_contract"] is True
    assert gather["output_shape"] == [2, 5, 4]
    assert gather_bad["valid_index_shape"] is False
    assert gather_bad["index_failure_reason"] == "index_non_dim_extent_exceeds_input"
    assert "output_shape" not in gather_bad
    assert scatter["valid_index_contract"] is True
    assert scatter["output_shape"] == [2, 3, 4]
    assert scatter_bad["valid_src_shape"] is False
    assert scatter_bad["index_failure_reason"] == "scatter_src_shape_mismatch"
    assert "output_shape" not in scatter_bad


def test_matmul_contract_distinguishes_broadcast_and_bmm_batch_rules():
    def case(case_id: int, left: list[int], right: list[int], batch_mode: str = "broadcast") -> CaseSpec:
        return CaseSpec(
            id=case_id,
            operator=OperatorSpec(name="custom.matmul"),
            invocation=InvocationSpec(api="torch.matmul", api_type="function"),
            generation=GenerationSpec(metadata={"operator_facts": {"contract": {"family": "matmul", "left": "input", "right": "other", "output_dtype": "input", "batch_mode": batch_mode}}}),
            parameters=[_parameter("input", left), _parameter("other", right)],
        )

    broadcast = expand_cases([case(785, [1, 2, 3, 4], [5, 1, 4, 6])])[0].metadata["resolved_operator_contract"]
    mismatch = expand_cases([case(786, [2, 3, 4], [5, 4, 6])])[0].metadata["resolved_operator_contract"]
    bmm_mismatch = expand_cases([case(787, [1, 2, 4], [3, 4, 6], "equal")])[0].metadata["resolved_operator_contract"]

    assert broadcast["valid_matmul"] is True
    assert broadcast["batch_shape"] == [5, 2]
    assert broadcast["output_shape"] == [5, 2, 3, 6]
    assert mismatch["valid_batch_broadcast"] is False
    assert mismatch["matmul_failure_reason"] == "batch_broadcast_mismatch"
    assert "output_shape" not in mismatch
    assert bmm_mismatch["valid_batch_broadcast"] is False
    assert bmm_mismatch["matmul_failure_reason"] == "batch_dimensions_mismatch"
    assert "output_shape" not in bmm_mismatch


def test_reduce_contract_validates_multi_dim_negative_and_duplicate_dimensions():
    def case(case_id: int, dim: object, keepdim: bool = False) -> CaseSpec:
        return CaseSpec(
            id=case_id,
            operator=OperatorSpec(name="torch.sum"),
            invocation=InvocationSpec(api="torch.sum", api_type="function"),
            parameters=[_parameter("input", [2, 3, 4]), ParameterSpec(name="dim", kind=ParameterKind.ATTRIBUTE_TUPLE, dtypes=["int64"], values=dim), ParameterSpec(name="keepdim", kind=ParameterKind.ATTRIBUTE, dtypes=["bool"], values=keepdim)],
        )

    legal = expand_cases([case(782, [-1, 0], True)])[0].metadata["resolved_operator_contract"]
    duplicate = expand_cases([case(783, [0, -3])])[0].metadata["resolved_operator_contract"]
    out_of_range = expand_cases([case(784, [3])])[0].metadata["resolved_operator_contract"]

    assert legal["valid_reduce_dims"] is True
    assert legal["reduced_dimensions"] == [2, 0]
    assert legal["output_shape"] == [1, 3, 1]
    assert duplicate["valid_reduce_dims"] is False
    assert duplicate["reduce_failure_reason"] == "duplicate_reduce_dim"
    assert "output_shape" not in duplicate
    assert out_of_range["valid_reduce_dims"] is False
    assert out_of_range["reduce_failure_reason"] == "reduce_dim_out_of_range"
    assert "output_shape" not in out_of_range


def test_view_contract_distinguishes_contiguous_and_non_contiguous_inputs():
    contiguous = CaseSpec(
        id=780,
        operator=OperatorSpec(name="torch.view"),
        invocation=InvocationSpec(api="torch.view", api_type="function"),
        parameters=[_parameter("input", [2, 3]), ParameterSpec(name="shape", kind=ParameterKind.ATTRIBUTE_TUPLE, dtypes=["int64"], values=[3, 2])],
    )
    non_contiguous = contiguous.model_copy(update={"id": 781, "parameters": [contiguous.parameters[0].model_copy(update={"metadata": {"shape_policy": {"non_contiguous": True}}}), contiguous.parameters[1]]})

    legal = expand_cases([contiguous])[0].metadata["resolved_operator_contract"]
    invalid = expand_cases([non_contiguous])[0].metadata["resolved_operator_contract"]

    assert legal["valid_view"] is True
    assert legal["input_contiguous"] is True
    assert legal["output_shape"] == [3, 2]
    assert invalid["valid_view"] is False
    assert invalid["input_contiguous"] is False
    assert invalid["view_failure_reason"] == "input_not_contiguous"
    assert "output_shape" not in invalid


def test_invalid_conv_contract_does_not_emit_output_shape():
    case = CaseSpec(id=778, operator=OperatorSpec(name="torch.conv2d"), invocation=InvocationSpec(api="torch.conv2d", api_type="function"), generation=GenerationSpec(), parameters=[_parameter("input", [1, 3, 5, 5]), _parameter("weight", [4, 2, 3, 3]), ParameterSpec(name="groups", kind=ParameterKind.ATTRIBUTE, values=2)])
    contract = expand_cases([case])[0].metadata["resolved_operator_contract"]
    assert contract["valid_groups"] is False
    assert "output_shape" not in contract


def test_attention_contract_reports_each_compatibility_dimension():
    case = CaseSpec(id=779, operator=OperatorSpec(name="torch.scaled_dot_product_attention"), invocation=InvocationSpec(api="torch.scaled_dot_product_attention", api_type="function"), parameters=[_parameter("query", [1, 2, 3, 4]), _parameter("key", [1, 3, 5, 6]), _parameter("value", [1, 3, 6, 7]), _parameter("attn_mask", [1, 1, 3, 5])])
    contract = expand_cases([case])[0].metadata["resolved_operator_contract"]
    assert contract["qk_embedding_compatible"] is True
    assert contract["kv_sequence_compatible"] is True
    assert contract["head_compatible"] is True
    assert contract["mask_broadcast_compatible"] is True
    assert contract["valid_attention"] is True


def test_complex_contract_invalid_variants_cover_conv_norm_attention():
    conv = CaseSpec(id=770, operator=OperatorSpec(name="torch.conv2d"), invocation=InvocationSpec(api="torch.conv2d", api_type="function"), generation=GenerationSpec(invalid_count=1), parameters=[_parameter("input", [1, 3, 8, 8]), _parameter("weight", [4, 3, 3, 3]), _parameter("bias", [4])])
    norm = CaseSpec(id=771, operator=OperatorSpec(name="torch.layer_norm"), invocation=InvocationSpec(api="torch.layer_norm", api_type="function"), generation=GenerationSpec(invalid_count=1), parameters=[_parameter("input", [2, 3, 4]), _parameter("normalized_shape", [4]), _parameter("weight", [4]), _parameter("bias", [4])])
    attention = CaseSpec(id=772, operator=OperatorSpec(name="torch.scaled_dot_product_attention"), invocation=InvocationSpec(api="torch.scaled_dot_product_attention", api_type="function"), generation=GenerationSpec(invalid_count=1), parameters=[_parameter("query", [1, 2, 3, 4]), _parameter("key", [1, 2, 5, 4]), _parameter("value", [1, 2, 5, 4])])

    generated = expand_cases([conv, norm, attention])

    assert generated[1].metadata["contract_invalid_reason"] == "conv_channel_mismatch"
    assert generated[1].parameters[1].metadata["selected_invalid_value"] == [4, 4, 3, 3]
    assert generated[3].metadata["contract_invalid_reason"] == "normalized_shape_mismatch"
    assert generated[3].parameters[1].metadata["selected_invalid_value"] == [5]
    assert generated[5].metadata["contract_invalid_reason"] == "attention_head_mismatch"
    assert generated[5].parameters[1].metadata["selected_invalid_value"] == [1, 3, 5, 4]
    assert generated[5].parameters[1].shape.dims == [1, 3, 5, 4]
    contract = generated[5].metadata["resolved_operator_contract"]
    assert contract["head_compatible"] is False
    assert contract["valid_attention"] is False
    assert "output_shape" not in contract
