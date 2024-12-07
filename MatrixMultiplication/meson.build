project(
    'MatrixMultiplication',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

executable('MatrixMultiplication',
    [
        'src/MatrixMultiplication.cpp',
        xxd.process('src/MatrixMultiplication.cl'),
    ],
    extra_files: [
        'src/MatrixMultiplication.cl',
    ],
    dependencies : [
        meson.get_compiler('c').find_library('cblas'),
        core_dep,
        opencl_dep,
    ],
)
