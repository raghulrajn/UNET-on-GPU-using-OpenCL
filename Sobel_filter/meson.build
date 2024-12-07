project(
    'Sobel',
    ['cpp', 'c'],
    default_options : [
        'buildtype=debugoptimized',
        'optimization=3',
        'cpp_std=c++14',
        'warning_level=3',
    ],
)

subdir('lib')

executable('Sobel',
    [
        'src/Sobel.cpp',
        xxd.process('src/Sobel.cl'),
    ],
    extra_files: [
        'src/Sobel.cl',
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
