"""mypy plugin facade (Phase 5, #199): re-exports + plugin() entry. Solvers
live in _types/_matmul/_reshape/_reductions/_builders/_ops/_broadcast/_linal
g/_indexing/_heads/_hooks; see those modules. No __all__ (explicit re-
exports only).
"""
from __future__ import annotations


from mypy.plugin import Plugin

# Phase 5 (#199) facade: eager submodule imports + explicit re-exports.
from . import (  # noqa: F401
    _broadcast, _broadcast_core, _builders, _dtypes, _einsum, _errors, _exprs, _fft, _forms, _frames,
    _heads, _hooks,
    _indexing, _linalg, _matmul, _ops, _reshape, _reductions, _state, _trace, _types, _unions,
    _unknowns,
)
from ._state import (  # noqa: F401
    STATE
)
from ._errors import (  # noqa: F401
    NUMTY_ANNOTATION, NUMTY_AXIS, NUMTY_BAIL, NUMTY_BROADCAST, NUMTY_CALLSITE, NUMTY_INTERNAL,
    NUMTY_SHAPE, NUMTY_UNSOLVED, _CANARY_BY_MODULE, _CANARY_FULLNAMES, _MIN_NUMPY_VERSION,
    _TESTED_MYPY_MAJOR, _TESTED_MYPY_MINOR, _TESTED_MYPY_VERSION, _bail, _mypy_version_warning,
    _numpy_version_warning, _numty_debug, _parse_version_tuple, _swallowed, _warn_startup,
)
from ._frames import (  # noqa: F401
    POLARS_FRAME_TO_NUMPY, POLARS_SERIES_TO_NUMPY, frame_to_numpy_hook
)
from ._trace import (  # noqa: F401
    _numty_strict, _numty_trace_path, _trace_event
)
from ._dtypes import (  # noqa: F401
    _DATETIME_FULLNAMES, _DTYPE_BASE, _DTYPE_INFO, _NBIT_MODULES, _NBIT_TAG_BY_BYTES,
    _NAME_OF_KIND_SIZE, _UFUNC_ABS, _UFUNC_BINARY_BOOL, _UFUNC_BINARY_FLOAT, _UFUNC_BINARY_PROMOTE,
    _UFUNC_BOOL, _UFUNC_FLOAT, _UFUNC_KEEP, _UFUNC_NO_COMPLEX, _UFUNC_REAL, _WEAK_DEFAULT,
    _WEAK_SCALARS, _datetime_string_kind, _promote_strong, _promote_weak,
)
from ._forms import (  # noqa: F401
    _DivForm, _LinForm, _MonoForm, _UNKNOWN_PREFIX, _canon_div, _coeff_vars_to_form, _eval_form,
    _form_has_unknown, _form_mul, _form_never_size, _form_to_coeff_vars, _forms_provably_ne,
    _is_unknown_fullname, _lin_add, _lin_scale, _mono_normalize,
)
from ._unknowns import (  # noqa: F401
    _fresh_unknown, _is_same_unknown, _unknown_display_name, _unknown_ids, _unknown_names, _unknown_per_line
)
from ._exprs import (  # noqa: F401
    _EXPR_PREFIX, _LATE_PREFIXES, _expr_constituents, _expr_forms, _expr_ids, _expr_str, _expr_var,
    _form_from_fullname, _form_of, _implicit_scope_vars, _materialize_dim, _mono_str, _ns_of_ctx,
    _normalize_ns, _resolve_dim, _seen_vars, _short_var, _current_scope_vars, _iter_tvarlikes,
)
from ._fft import (  # noqa: F401
    _complex_dtype, _fft_axis_n, _fft_pair, _fft_s_axes, _fftn_solve, _rfft_complex_input,
    _rfft_size, _rfft_solve, fft2_function_hook, fft_function_hook, fftn_function_hook,
    fftshift_function_hook, irfft_function_hook, rfft2_function_hook, rfft_function_hook,
    rfftn_function_hook,
)
from ._einsum import (  # noqa: F401
    _einsum_signature_hook, _parse_einsum_term, einsum_function_hook
)
from ._linalg import (  # noqa: F401
    _cross_int_value, _linalg_cond, _linalg_cross_signature_hook, _linalg_det, _linalg_dtype,
    _linalg_eig, _linalg_eig_names, _linalg_flag, _linalg_lstsq, _linalg_matrix_power, _linalg_norm,
    _linalg_pinv, _linalg_qr, _linalg_real_name, _linalg_same_shape, _linalg_slogdet, _linalg_solve,
    _linalg_solve_dtype, _linalg_svd, _linalg_tensorsolve, _linalg_tuple, _norm_ord_fits,
    _norm_signature_hook, _reject_float16_linalg, cross_function_hook, linalg_cross_function_hook,
    linalg_function_hook, _dtype_base, _kind_of, _maybe_scalar, _scalar_of_dtype, _string_value,
)
from ._indexing import (  # noqa: F401
    _consume_one, _consume_prefix, _consume_suffix, _getitem_types, _index_pairs, _is_scalar_target,
    _parse_index_item, _parse_index_items, _solve_getitem, _solve_getitem_unpack, _take_in_bounds,
    _take_index_values, _take_indices, _take_solve, getitem_method_hook, setitem_method_hook,
    take_along_axis_function_hook, take_function_hook, take_method_hook,
)
from ._broadcast import (  # noqa: F401
    _apply_ndmin, _asarray_array_type, _broadcast_dims, _broadcast_dims_unpack, _broadcast_result,
    _broadcast_shapes, _broadcast_to_compatible, _broadcast_to_types, _clip_types, _combine_specs,
    _compare_types, _display_element_kind, _is_bool_operand, _is_builtin_int, _is_plain_bool,
    _logical_types, _merge_shape_lists, _operand_spec, _smallest_float, _solve_asarray,
    _solve_broadcast_to, _solve_clip, _solve_round, _solve_truediv_pair, _solve_ufunc,
    _solve_ufunc_binary_bool, _solve_ufunc_binary_float, _solve_where, _specs_dtype, _truediv_types,
    _ufunc_bool_types, _ufunc_call_solve, _ufunc_float_types, _ufunc_name, _ufunc_types,
    _where_types, angle_function_hook, asarray_function_hook, broadcast_method_hook,
    broadcast_to_function_hook, clip_function_hook, clip_method_hook, compare_method_hook,
    divmod_function_hook, divmod_method_hook, floordiv_mod_pow_method_hook, inplace_method_hook,
    isclose_function_hook, logical_method_hook, round_function_hook, round_method_hook,
    subtract_method_hook, truediv_method_hook, ufunc_call_hook, ufunc_function_hook,
    where_function_hook,
)
from ._heads import (  # noqa: F401
    _FAM_HEAD_DTYPES, _FamBail, _FamFail, _arr_head, _fam_applied, _fam_arg, _fam_bare, _fam_bind_words,
    _fam_ensure_implicit, _fam_expr, _fam_head, _fam_string, _is_fam_int_name,
)
from ._types import (  # noqa: F401
    _int_value, _dim_value, _dim_eq, _dims_provably_ne, _dim_str, _dim_add_const, _dim_add2,
    _dim_mul_const, _dim_div, _mul_dims, _is_bare_name_dim, _linear_product, _reshape_total_form,
    _check_dim_eq, _shape_dims, _shape_dims_unpack, _fold_int, _fold_len_arg, _fold_shape_expr,
    _ndarray_parts, _parts_or_unpack, _has_unpack, _union_members, _distribute_pairings,
    _union_results, _distribute_solve, _split_flat, _shape_tvar_formals, _strip_optional,
    _sequence_element, _align_pair, _pairs_for_actual, _conflict_pairs, _emit_call_site_conflicts,
    _callsite_formula_return, _callsite_expr_bindings, _substitute_return_exprs, _fixed_shape,
    _shape_note_str, _rank_mismatch_notes, _reported_as_operator, _mentions_ndarray,
    _info_mentions_array, _note_rank_mismatches, _ndarray_type, _single_shape_arg,
    _single_shape_expr, _shape_from_arg, _int_shape_from_arg, _builder_fold_dim_expr,
    NDARRAY_FULLNAME, INT_FULLNAME, TUPLE_FULLNAME, FLOAT64_FULLNAME, DTYPE_FULLNAME, STR_FULLNAME,
    _SEQUENCE_FULLNAMES, _OPERATOR_DUNDERS, _enum_member_count, _fold_dim_expr, _has_unknown_shape,
    _unpack_shape,
)
from ._builders import (  # noqa: F401
    _operands_have_known_shape, _builder_shape, _has_literal_dim, _has_int_dim, _has_mixed_operand,
    _is_degraded_operand, _undegraded_args, _guard_mixed, _default_float_dtype, _dims_fit_intp,
    zeros_like_hook, _default_dtype, _fill_dtype_name, _fold_single, _explicit_dtype,
    _instance_fullname, _scalar_ret_name, _named_dtype, _apply_out_dtype, like_function_hook,
    _like_types, copy_function_hook, ascontiguousarray_function_hook, full_function_hook,
    eye_function_hook, _builder_single_dim, linspace_function_hook, identity_function_hook,
    tri_function_hook, _arange_symbolic, _arange_symbolic_kw, arange_function_hook,
    _comprehension_parts, _sequence_parts, _is_dynamic_sequence, _is_empty_sequence, _concat_axis,
    _single_axis_value, _element_expr_type, _dynamic_sequence_element, concatenate_function_hook,
    _stack_promote, _concat_merge, _stack_solve, vstack_function_hook, hstack_function_hook,
    dstack_function_hook, column_stack_function_hook, _dynamic_explicit_dtype, _concat_dynamic,
    _stack_dynamic, stack_function_hook, _SHAPE_KEEPERS, INTP_MAX, _BUILTIN_DTYPE, _DTYPE_STRINGS,
)
from ._matmul import (  # noqa: F401
    _matmul_inner, _matmul_result, _matmul_unpack_dims, _matmul_promoted, _operand_parts,
    _solve_matmul, _positional_actuals, _matmul_has_dims, matmul_hook, _two_arrays, _contract_pair,
    _promoted_scalar, dot_function_hook, dot_method_hook, matmul_method_hook, rmatmul_method_hook,
    imatmul_method_hook, inner_function_hook, _total_size, outer_function_hook, kron_function_hook,
    vdot_function_hook, _tensordot_axes, tensordot_function_hook,
)
from ._ops import (  # noqa: F401
    _check_each_axis, _roll_shift_matches, _same_shape_hook, flip_function_hook,
    fliplr_function_hook, flipud_function_hook, _rank_check_hook, trace_function_hook,
    trace_method_hook, diag_function_hook, diagonal_function_hook, diagonal_method_hook,
    corrcoef_function_hook, cov_function_hook, roll_function_hook, gradient_function_hook,
    _gradient_result, _parse_int_pair, rot90_function_hook, _convolve_solve, _convolve_same_size,
    _convolve_valid_size, convolve_function_hook, correlate_function_hook, _hook_pair, _tile_reps,
    tile_function_hook, _repeat_count, _repeat_display, _repeat_display_size, repeat_function_hook,
    repeat_method_hook, _pad_side, _pad_widths, _pad_side_fallback, _pad_side_fallback_single,
    pad_function_hook, split_function_hook, array_split_function_hook, _select_index_elem,
    _nonzero_like, nonzero_function_hook, count_nonzero_function_hook, flatnonzero_function_hook,
    argwhere_function_hook, unique_function_hook, isin_function_hook, _search_values_dims,
    searchsorted_function_hook, sliding_window_view_function_hook, atleast_function_hook,
    _fft_seq_arg, _fft_seq_entry,
)
from ._reshape import (  # noqa: F401
    _dims_str, _reshape_new_dims, _solve_reshape_unpack, _reshape_fold_arithmetic,
    _reshape_dynamic_dims, _solve_reshape_dynamic, _solve_reshape, _all_positionals,
    _all_positionals_expanded, _formal_pair, _fold_bool, _bool_value, _fold_flag, _is_none_type,
    _parse_axis_arg, _normalize_axis, _normalize_axis_0d, _nbit_bytes, _dtype_kind_size,
    _dtype_fullname, _solve_transpose, _apply_permutation, _solve_transpose_unpack, _datetime_kind,
    _construct_datetime_dtype, _weak_name, _scalar_key, _promote_dtypes, _promote_all,
    _solve_squeeze, _solve_expand_dims, reshape_method_hook, reshape_function_hook,
    transpose_method_hook, transpose_function_hook, _swapaxes_solve, swapaxes_function_hook,
    swapaxes_method_hook, _moveaxis_permutation, _parse_shape_like, moveaxis_function_hook,
    rollaxis_function_hook, _matrix_transpose_solve, matrix_transpose_function_hook,
    mt_attribute_hook, permute_dims_function_hook, append_function_hook, insert_function_hook,
    delete_function_hook, squeeze_method_hook, squeeze_function_hook, expand_dims_function_hook,
    REDUCTION_DTYPES, _check_unique_axes, _construct_dtype, _exact_element, _int_or_none,
    _is_mixed_shape, _is_plain_int, _literal_positions, _mixed_parts, _mixed_result, _movable_parts,
    _ndarrays_in, _picture_complete, _positionals_only, _unknown_axis_parts, _widen_literal_dims,
)
from ._reductions import (  # noqa: F401
    _reduction_out_dtype, _solve_reduction, _reduction_inputs_bool, _reduction_bool_types,
    allany_method_hook, allany_function_hook, _solve_reduction_unpack, _reduction_inputs,
    _reduction_dtype_only, _reduction_types, reduction_method_hook, _ufunc_reduce_types,
    _ufunc_reduce_inputs, _ufunc_accumulate_types, _ufunc_accumulate_inputs, _ufunc_outer_shape,
    _ufunc_outer_types, _ufunc_outer_inputs, _ufunc_any_dtype, _reduceat_length,
    _ufunc_reduceat_types, _ufunc_reduceat_inputs, ufunc_method_hook, _enum_member_value,
    _fold_index_bound, _final_slice_call, _solve_argdrop, _solve_same_or_flat, _kth_in_bounds,
    _arg_inputs, arg_method_hook, arg_function_hook, _parse_size_arg, rng_method_hook,
    legacy_rand_hook, _percentile_q_dims, percentile_function_hook, reduction_function_hook,
    _same_dims, _check_average_weights, _average_weighted_dtype, average_function_hook,
    _flatten_total_dim, _cumsum_solve, cumsum_function_hook, cumsum_method_hook, _ravel_solve,
    ravel_method_hook, flatten_method_hook, ravel_function_hook, diff_function_hook,
    _promoted_result_type, ARG_DROP, ARG_SAME_INT, ARG_SAME, RNG_FLOAT_OPS,
)
from ._broadcast_core import (  # noqa: F401
    _broadcast_types, _merge_known_pair, _merge_unpack_dims, _same_unpack, _solve_broadcast,
    _solve_broadcast_scalar, _solve_broadcast_unpack,
)
from ._hooks import (  # noqa: F401
    NumtyPlugin, _reshape_signature_hook, _broadcast_signature_hook, _matmul_signature_hook,
    identity_method_hook, _identity_type, astype_method_hook, _astype_type, _has_expr_var,
    _subst_expr_dim, _subst_attr_type, substituted_attribute_hook, shape_attribute_hook,
    _shape_type, t_attribute_hook, _t_type, len_function_hook, _numpy_function_hooks,
    _NUMPY_FUNCTION_HOOKS,
)
def _installed_numpy_version() -> str | None:
    """Installed numpy version, or None when it cannot be determined."""
    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("numpy")
    except Exception:
        _swallowed(None, "_installed_numpy_version")
        return None


def __getattr__(name: str) -> int:
    # Phase 4 (#198): _recheck_depth lives in STATE (int field).
    if name == "_recheck_depth":
        return STATE.recheck_depth
    raise AttributeError(f"module 'numty_plugin' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), "_recheck_depth"])


def _installed_numpy_warning() -> str | None:
    """Loud warning when the installed numpy leaves the supported range."""
    try:
        numpy_version = _installed_numpy_version()
    except Exception:
        _swallowed(None, "_installed_numpy_warning")
        return None
    if numpy_version is None:
        return None
    return _numpy_version_warning(numpy_version)


def plugin(version: str) -> type[Plugin]:
    for warning in (_mypy_version_warning(version), _installed_numpy_warning()):
        if warning is not None:
            _warn_startup(warning)
    return NumtyPlugin


