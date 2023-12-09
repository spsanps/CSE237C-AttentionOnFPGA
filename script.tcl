############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 2013 Xilinx Inc. All rights reserved.
############################################################
open_project hls_project
set_top attention
add_files attention.h
add_files attention.cpp
add_files dotProd.h
add_files dotProd.cpp
add_files QKV.h
add_files QKV.cpp
add_files QKVProj.h
add_files QKVProj.cpp
add_files weights16.h
add_files weights32.h
add_files weights64.h
## add_files -tb test.cpp
open_solution "solution1"
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

