project(
    'Mandelbrot',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

executable('Mandelbrot',
    [
        'src/Mandelbrot.cpp',
        xxd.process('src/Mandelbrot.cl'),
    ],
    extra_files: [
        'src/Mandelbrot.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
    ],
)
