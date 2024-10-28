project(
    'OpenCLExercise2_Mandelbrot',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

executable('OpenCLExercise2_Mandelbrot',
    [
        'src/OpenCLExercise2_Mandelbrot.cpp',
        xxd.process('src/OpenCLExercise2_Mandelbrot.cl'),
    ],
    extra_files: [
        'src/OpenCLExercise2_Mandelbrot.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
    ],
)
