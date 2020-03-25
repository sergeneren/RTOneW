# RTOneW
RTOneW is an implementation of Peter Shirley's "Ray Tracing in One Weekend" books using Cuda language and Qt5 gui   

![My image](https://github.com/sergeneren/RTOneW/blob/master/img/Main.JPG)


### Installing

RTOneW install is configured for windows and vcpkg

- Clone [vcpkg](https://github.com/Microsoft/vcpkg)  
- Install Qt5 base with vcpkg  "vcpkg.exe install Qt5-base:X64-windows" 
- Clone RTOneW 
- With cmake gui select source dir as source and create a build dir

- Configure with x64 platform and choose to specify toolchain platform
![My image](https://github.com/sergeneren/RTOneW/blob/master/img/step_1.JPG)


- Select your vcpkg_directory/scripts/buildsytems/vcpkg.cmake file 
![My image](https://github.com/sergeneren/RTOneW/blob/master/img/step_2.JPG)


- Configure and generate visual studio solution. Built files will be placed under: BUILD_DIR/RTOneW/bin


## Author

* **Sergen Eren** - [My website](https://sergeneren.com) - [Vimeo](https://vimeo.com/sergeneren)

## Status
:red_circle: This project is closed to development and maintenance 

## License
This project is licensed under GNU General Public License v3.0

## Acknowledgments
* [Ray Tracing in One Weekend](http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html) - *Big thanks to Peter Shirley*
* [Accelerated Ray Tracing in One Weekend in CUDA](https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/) - *Roger Allen's first cuda implementation* 
