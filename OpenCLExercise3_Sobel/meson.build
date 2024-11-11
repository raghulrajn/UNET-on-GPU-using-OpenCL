project(
    'OpenCLExercise3_Sobel',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

executable('OpenCLExercise3_Sobel',
    [
        'src/OpenCLExercise3_Sobel.cpp',
        xxd.process('src/OpenCLExercise3_Sobel.cl'),
    ],
    extra_files: [
        'src/OpenCLExercise3_Sobel.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
    ],
)

configure_file(
  input : 'Valve.pgm',
  output : 'Valve.pgm',
  copy : true,
)
