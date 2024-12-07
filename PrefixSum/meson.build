project(
    'PrefixSum',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

executable('PrefixSum',
    [
        'src/PrefixSum.cpp',
        xxd.process('src/PrefixSum.cl'),
    ],
    extra_files: [
        'src/PrefixSum.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
    ],
)
