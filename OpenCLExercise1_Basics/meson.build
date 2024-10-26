project(
    'OpenCLExercise1_Basics',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

executable('OpenCLExercise1_Basics',
    [
        'src/OpenCLExercise1_Basics.cpp',
        xxd.process('src/OpenCLExercise1_Basics.cl'),
    ],
    extra_files: [
        'src/OpenCLExercise1_Basics.cl',
    ],
    dependencies : [
        core_dep,
        opencl_dep,
    ],
)
