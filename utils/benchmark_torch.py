import torch

def batched_dot_mul_sum(a, b):
    '''computes batched dot by mulitplying and summing'''
    return a.mul(b).sum(-1)

def batched_dot_bmm(a, b):
    '''computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)

# input for benchmarking
x = torch.randn(10000, 64)

# ensure both functions compute the same output
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

import timeit

t0 = timeit.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')

import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

print(t0.timeit(100))
print(t1.timeit(100))

num_threads = torch.get_num_threads()
print(f'Benchmarking on {num_threads} threads')

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using mul and sum')

t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x},
    num_threads=num_threads,
    label='Multithreaded batch dot',
    sub_label='Implemented using bmm')

print(t0.timeit(100))
print(t1.timeit(100))


x = torch.randn(10000, 1024, device='cuda')

t0 = timeit.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = timeit.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

# Ran each twice to show difference before/after warm-up
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'mul_sum(x, x):  {t0.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')
print(f'bmm(x, x):      {t1.timeit(100) / 100 * 1e6:>5.1f} us')

t0 = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='from __main__ import batched_dot_mul_sum',
    globals={'x': x})

t1 = benchmark.Timer(
    stmt='batched_dot_bmm(x, x)',
    setup='from __main__ import batched_dot_bmm',
    globals={'x': x})

# Run only once since benchmark module does warm-up for us
print(t0.timeit(100))
print(t1.timeit(100))

m0 = t0.blocked_autorange()
m1 = t1.blocked_autorange()

print(m0)
print(m1)

print(f"Mean:   {m0.mean * 1e6:6.2f} us")
print(f"Median: {m0.median * 1e6:6.2f} us")

from itertools import product

# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [1, 64, 1024, 10000]
for b, n in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = 'Batched dot'
    sub_label = f'[{b}, {n}]'
    x = torch.ones((b, n))
    for num_threads in [1, 4, 16, 32]:
        results.append(benchmark.Timer(
            stmt='batched_dot_mul_sum(x, x)',
            setup='from __main__ import batched_dot_mul_sum',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='mul/sum',
        ).blocked_autorange(min_run_time=1))
        results.append(benchmark.Timer(
            stmt='batched_dot_bmm(x, x)',
            setup='from __main__ import batched_dot_bmm',
            globals={'x': x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='bmm',
        ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.print()

compare.trim_significant_figures()
compare.colorize()
compare.print()

import pickle

ab_test_results = []
for env in ('environment A: mul/sum', 'environment B: bmm'):
    for b, n in ((1, 1), (1024, 10000), (10000, 1)):
        x = torch.ones((b, n))
        dot_fn = (batched_dot_mul_sum if env == 'environment A: mul/sum' else batched_dot_bmm)
        m = benchmark.Timer(
            stmt='batched_dot(x, x)',
            globals={'x': x, 'batched_dot': dot_fn},
            num_threads=1,
            label='Batched dot',
            description=f'[{b}, {n}]',
            env=env,
        ).blocked_autorange(min_run_time=1)
        ab_test_results.append(pickle.dumps(m))

ab_results = [pickle.loads(i) for i in ab_test_results]
compare = benchmark.Compare(ab_results)
compare.trim_significant_figures()
compare.colorize()
compare.print()

# And just to show that we can round trip all of the results from earlier:
round_tripped_results = pickle.loads(pickle.dumps(results))
assert(str(benchmark.Compare(results)) == str(benchmark.Compare(round_tripped_results)))

from torch.utils.benchmark import Fuzzer, FuzzedParameter, FuzzedTensor, ParameterAlias

# Generates random tensors with 128 to 10000000 elements and sizes k0 and k1 chosen from a
# ``loguniform`` distribution in [1, 10000], 40% of which will be discontiguous on average.
example_fuzzer = Fuzzer(
    parameters = [
        FuzzedParameter('k0', minval=1, maxval=10000, distribution='loguniform'),
        FuzzedParameter('k1', minval=1, maxval=10000, distribution='loguniform'),
    ],
    tensors = [
        FuzzedTensor('x', size=('k0', 'k1'), min_elements=128, max_elements=10000000, probability_contiguous=0.6)
    ],
    seed=0,
)

results = []
for tensors, tensor_params, params in example_fuzzer.take(10):
    # description is the column label
    sub_label=f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    results.append(benchmark.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='mul/sum',
    ).blocked_autorange(min_run_time=1))
    results.append(benchmark.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='bmm',
    ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.print()

from torch.utils.benchmark.op_fuzzers import binary

results = []
for tensors, tensor_params, params in binary.BinaryOpFuzzer(seed=0).take(10):
    sub_label=f"{params['k0']:<6} x {params['k1']:<4} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
    results.append(benchmark.Timer(
        stmt='batched_dot_mul_sum(x, x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='mul/sum',
    ).blocked_autorange(min_run_time=1))
    results.append(benchmark.Timer(
        stmt='batched_dot_bmm(x, x)',
        setup='from __main__ import batched_dot_bmm',
        globals=tensors,
        label='Batched dot',
        sub_label=sub_label,
        description='bmm',
    ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize(rowwise=True)
compare.print()

batched_dot_src = """\
/* ---- Python ---- */
// def batched_dot_mul_sum(a, b):
//     return a.mul(b).sum(-1)

torch::Tensor batched_dot_mul_sum_v0(
    const torch::Tensor a,
    const torch::Tensor b) {
  return a.mul(b).sum(-1);
}

torch::Tensor batched_dot_mul_sum_v1(
    const torch::Tensor& a,
    const torch::Tensor& b) {
  return a.mul(b).sum(-1);
}
"""


# PyTorch makes it easy to test our C++ implementations by providing a utility
# to JIT compile C++ source into Python extensions:
import os
from torch.utils import cpp_extension
cpp_lib = cpp_extension.load_inline(
    name='cpp_lib',
    cpp_sources=batched_dot_src,
    extra_cflags=['-O3'],
    extra_include_paths=[
        # `load_inline` needs to know where to find ``pybind11`` headers.
        os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    ],
    functions=['batched_dot_mul_sum_v0', 'batched_dot_mul_sum_v1']
)

# `load_inline` will create a shared object that is loaded into Python. When we collect
# instruction counts Timer will create a subprocess, so we need to re-import it. The
# import process is slightly more complicated for C extensions, but that's all we're
# doing here.
module_import_str = f"""\
# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
import importlib.util
spec = importlib.util.spec_from_file_location("cpp_lib", {repr(cpp_lib.__file__)})
cpp_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpp_lib)"""

import textwrap
def pretty_print(result):
    """Import machinery for ``cpp_lib.so`` can get repetitive to look at."""
    print(repr(result).replace(textwrap.indent(module_import_str, "  "), "  import cpp_lib"))


t_baseline = benchmark.Timer(
    stmt='batched_dot_mul_sum(x, x)',
    setup='''\
from __main__ import batched_dot_mul_sum
x = torch.randn(2, 2)''')

t0 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v0(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

t1 = benchmark.Timer(
    stmt='cpp_lib.batched_dot_mul_sum_v1(x, x)',
    setup=f'''\
{module_import_str}
x = torch.randn(2, 2)''')

# Moving to C++ did indeed reduce overhead, but it's hard to tell which
# calling convention is more efficient. v1 (call with references) seems to
# be a bit faster, but it's within measurement error.
pretty_print(t_baseline.blocked_autorange())
pretty_print(t0.blocked_autorange())
pretty_print(t1.blocked_autorange())

# Let's use ``Callgrind`` to determine which is better.
stats_v0 = t0.collect_callgrind()
stats_v1 = t1.collect_callgrind()

pretty_print(stats_v0)
pretty_print(stats_v1)

# `.as_standardized` removes file names and some path prefixes, and makes
# it easier to read the function symbols.
stats_v0 = stats_v0.as_standardized()
stats_v1 = stats_v1.as_standardized()

# `.delta` diffs the instruction counts, and `.denoise` removes several
# functions in the Python interpreter that are known to have significant
# jitter.
delta = stats_v1.delta(stats_v0).denoise()

# `.transform` is a convenience API for transforming function names. It is
# useful for increasing cancelation when ``diff-ing`` instructions, as well as
# just generally improving readability.
replacements = (
    ("???:void pybind11", "pybind11"),
    ("batched_dot_mul_sum_v0", "batched_dot_mul_sum_v1"),
    ("at::Tensor, at::Tensor", "..."),
    ("at::Tensor const&, at::Tensor const&", "..."),
    ("auto torch::detail::wrap_pybind_function_impl_", "wrap_pybind_function_impl_"),
)
for before, after in replacements:
    delta = delta.transform(lambda l: l.replace(before, after))

# We can use print options to control how much of the function to display.
torch.set_printoptions(linewidth=160)

# Once parsed, the instruction counts make clear that passing `a` and `b`
# by reference is more efficient as it skips some ``c10::TensorImpl`` bookkeeping
# for the intermediate Tensors, and is also works better with ``pybind11``. This
# is consistent with our noisy wall time observations.
print(delta)


